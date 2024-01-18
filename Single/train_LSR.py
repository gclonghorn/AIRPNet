# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import argparse
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import ImageFolder
from losses import LSR_Loss
import logging
import numpy as np
import PIL.Image as Image
from torchvision.transforms import ToPILImage
from pytorch_msssim import ms_ssim
from typing import Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from util import DWT,IWT,setup_logger
from LIH import Model as hide_model
from LSR import Model as restore_model
from tqdm import tqdm
import lpips






def downsample(hr,scale):
    lr = F.interpolate(hr, scale_factor=1.0/scale, mode='bicubic')
    lr = F.interpolate(lr, scale_factor=scale, mode='bicubic')
    return lr

def guass_blur(hr,k_sz,sigma):
    transform = transforms.GaussianBlur(kernel_size=k_sz,sigma=sigma)
    return transform(hr)
    
def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.cpu().clamp_(0, 1).squeeze())


def compute_metrics(
        a: Union[np.array, Image.Image],
        b: Union[np.array, Image.Image],
        max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    return optimizer



def train_one_epoch(
    hide_model, denoise_model, criterion, train_dataloader, hide_optimizer, denoise_optimizer, epoch, logger_train, tb_logger, args
):
    if args.finetune:
        hide_model.train()
    else:
         for param in hide_model.parameters():
            param.requires_grad=False
    denoise_model.train()
 
    device = next(hide_model.parameters()).device
    dwt = DWT()
    iwt = IWT()

    for i, d in enumerate(train_dataloader):
        batch_size = d.shape[0]
        sp1 = int(round(batch_size/2*0.25))
        sp2 = int(round(batch_size/2*0.25))
        sp3 = int(batch_size/2-sp1-sp2)

        d = d.to(device)  #[16,3,224,224]
        cover_img = d[d.shape[0] // 2:, :, :, :]  #[8,3,224,224]
        secret_img = d[:d.shape[0] // 2, :, :, :]
        ns,bs,ss = torch.split(secret_img,[sp1,sp2,sp3],dim=0)
        #noise
        noiselvl = np.random.uniform(0,55,size=1) #random noise level
        noise = torch.cuda.FloatTensor(ns.size()).normal_(mean=0, std=noiselvl[0] / 255.) 
        noise_secret = ns + noise 
        #blur
        blur_secret = guass_blur(bs,2*random.randint(0,11)+3,random.uniform(0.1,2))
        scalelvl = random.choice([2,4])
        #lr
        lr_secret = downsample(ss,scalelvl)  
        degrade_secret_img = torch.cat([noise_secret,blur_secret,lr_secret],dim=0)

        input_cover = dwt(cover_img)
        input_secret = dwt(degrade_secret_img)
        
        if args.finetune:
             hide_optimizer.zero_grad()
        denoise_optimizer.zero_grad()
        #################
        # hide#
        #################

        output_steg, output_z = hide_model(input_cover,input_secret)
        steg_img = iwt(output_steg)

        #################
        #denoise#
        #################
        steg_clean = denoise_model(steg_img,sp1=sp1,sp2=sp2,sp3=sp3)
        output_clean = dwt(steg_clean)

        #################
        #reveal#
        #################
        output_z_guass = gauss_noise(output_z.shape)
        cover_rev, secret_rev= hide_model(output_clean, output_z_guass,rev=True)
        rec_img = iwt(secret_rev)
        
        #loss
        out_criterion = criterion(secret_img,cover_img,steg_clean,rec_img,args.sweight,args.cweight,args.pweight_c)
        loss = out_criterion['loss']
        loss.backward()
        denoise_optimizer.step()
        if args.finetune:
            hide_optimizer.step()

        if i % 10 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {loss.item():.3f} |'
        
            )
    tb_logger.add_scalar('{}'.format('[train]: loss'), loss.item(), epoch)


def test_epoch(args,epoch, test_dataloader, hide_model, denoise_model,logger_val,criterion,lpips_fn):
    dwt = DWT()
    iwt = IWT()
    hide_model.eval()
    device = next(hide_model.parameters()).device
    psnrc_n = AverageMeter()
    ssimc_n = AverageMeter()
    psnrs_n = AverageMeter()
    ssims_n = AverageMeter()
    psnrc_b= AverageMeter()
    ssimc_b = AverageMeter()
    psnrs_b = AverageMeter()
    ssims_b = AverageMeter()
    psnrc_s= AverageMeter()
    ssimc_s = AverageMeter()
    psnrs_s = AverageMeter()
    ssims_s = AverageMeter()
    psnrori_n = AverageMeter()
    ssimori_n = AverageMeter()
    psnrori_b = AverageMeter()
    ssimori_b = AverageMeter()
    psnrori_s = AverageMeter()
    ssimori_s = AverageMeter()
    lpips_n = AverageMeter()
    lpips_b = AverageMeter()
    lpips_s = AverageMeter()
    lpipsori_n = AverageMeter()
    lpipsori_b = AverageMeter()
    lpipsori_s = AverageMeter()
    lpips_ns = AverageMeter()
    lpips_bs = AverageMeter()
    lpips_ss = AverageMeter()
    loss = AverageMeter()
    i=0
    
    with torch.no_grad():
        for idx,d in tqdm(enumerate(test_dataloader)):
            d = d.to(device)
            cover_img = d[d.shape[0] // 2:, :, :, :]  #[1,3,224,224]
            secret_img = d[:d.shape[0] // 2, :, :, :]
            #degrade
            noise = torch.cuda.FloatTensor(secret_img.size()).normal_(mean=0, std=25 / 255.) 
            noise_secret = secret_img + noise 
            blur_secret = guass_blur(secret_img,15,args.std)
            scalelvl = 4
            lr_secret = downsample(secret_img,scalelvl)  
            degrade_secret_img = torch.cat([noise_secret,blur_secret,lr_secret],dim=0)

            input_cover = dwt(cover_img)
            input_secret = dwt(degrade_secret_img)
            
           #################
            # hide#
            #################

            output_steg, output_z = hide_model(input_cover,input_secret)
            steg_img = iwt(output_steg)
            nsteg_ori,bsteg_ori,ssteg_ori = torch.split(steg_img,1,dim=0)    

            #################
            #denoise#
            #################
            steg_clean = denoise_model(steg_img,sp1=1,sp2=1,sp3=1)
            nsteg,bsteg,ssteg = torch.split(steg_clean,1,dim=0)
            output_clean = dwt(steg_clean)
 
            
            #################
            #reveal#
            #################
            output_z_guass = gauss_noise(output_z.shape)
            cover_rev, secret_rev= hide_model(output_clean, output_z_guass,rev=True)
            rec_img = iwt(secret_rev)
            nrec,brec,srec = torch.split(rec_img,1,dim=0)
            
            out_criterion = criterion(secret_img,cover_img,steg_clean,rec_img,args.sweight,args.cweight,args.pweight_c)
            loss.update(out_criterion["loss"])
            
            lpipsn = lpips_fn.forward(cover_img,nsteg)
            lpips_n.update(lpipsn.mean().item())
            lpipsb = lpips_fn.forward(cover_img,bsteg)
            lpips_b.update(lpipsb.mean().item())
            lpipss = lpips_fn.forward(cover_img,ssteg)
            lpips_s.update(lpipss.mean().item())
            
            lpipsn = lpips_fn.forward(cover_img,nsteg_ori)
            lpipsori_n.update(lpipsn.mean().item())
            lpipsb = lpips_fn.forward(cover_img,bsteg_ori)
            lpipsori_b.update(lpipsb.mean().item())
            lpipss = lpips_fn.forward(cover_img,ssteg_ori)
            lpipsori_s.update(lpipss.mean().item())
            
            lpipsn = lpips_fn.forward(secret_img,nrec)
            lpips_ns.update(lpipsn.mean().item())
            lpipsb = lpips_fn.forward(secret_img,brec)
            lpips_bs.update(lpipsb.mean().item())
            lpipss = lpips_fn.forward(secret_img,srec)
            lpips_ss.update(lpipss.mean().item())
            
            

            #################
            #reveal wo processing#
            #################
            cover_rev2, secret_rev2= hide_model(output_steg, output_z_guass,rev=True)
            rec_img_ori = iwt(secret_rev2)
            nori,bori,sori = torch.split(rec_img_ori,1,dim=0)
            
            
            #compute psnr and save image
            save_dir = os.path.join('experiments', args.experiment,'images')
            
            secret_img = torch2img(secret_img)
            cover_img = torch2img(cover_img)

            denoise_steg_img = torch2img(nsteg)
            deblur_steg_img=  torch2img(bsteg)
            sr_steg_img=  torch2img(ssteg)

            noise_steg_img = torch2img(nsteg_ori)
            blur_steg_img=  torch2img(bsteg_ori)
            lr_steg_img=  torch2img(ssteg_ori)

            noise_secret_img = torch2img(noise_secret)
            blur_secret_img = torch2img(blur_secret)
            lr_secret_img = torch2img(lr_secret)

            denoise_secret_img = torch2img(nrec)
            deblur_secret_img = torch2img(brec)
            sr_secret_img = torch2img(srec)
           
            ndenoise_secret_img = torch2img(nori)
            ndeblur_secret_img = torch2img(bori)
            nsr_secret_img = torch2img(sori)
            
            p1, m1 = compute_metrics(denoise_secret_img, secret_img)
            psnrs_n.update(p1)
            ssims_n.update(m1)
            p2, m2 = compute_metrics(denoise_steg_img, cover_img)
            psnrc_n.update(p2)
            ssimc_n.update(m2)
            p3, m3 = compute_metrics(noise_steg_img, cover_img)
            psnrori_n.update(p3)
            ssimori_n.update(m3)

            p1, m1 = compute_metrics(deblur_secret_img, secret_img)
            psnrs_b.update(p1)
            ssims_b.update(m1)
            p2, m2 = compute_metrics(deblur_steg_img, cover_img)
            psnrc_b.update(p2)
            ssimc_b.update(m2)
            p3, m3 = compute_metrics(blur_steg_img, cover_img)
            psnrori_b.update(p3)
            ssimori_b.update(m3)

            p1, m1 = compute_metrics(sr_secret_img, secret_img)
            psnrs_s.update(p1)
            ssims_s.update(m1)
            p2, m2 = compute_metrics(sr_steg_img, cover_img)
            psnrc_s.update(p2)
            ssimc_s.update(m2)
            p3, m3 = compute_metrics(lr_steg_img, cover_img)
            psnrori_s.update(p3)
            ssimori_s.update(m3)
          

            if args.save_img:
                denoise_rec_dir = os.path.join(save_dir,'rec','denoise')
                if not os.path.exists(denoise_rec_dir):
                    os.makedirs(denoise_rec_dir)
                deblur_rec_dir = os.path.join(save_dir,'rec','deblur')
                if not os.path.exists(deblur_rec_dir):
                    os.makedirs(deblur_rec_dir)
                sr_rec_dir = os.path.join(save_dir,'rec','sr')
                if not os.path.exists(sr_rec_dir):
                    os.makedirs(sr_rec_dir)

                secret_dir = os.path.join(save_dir,'secret')
                if not os.path.exists(secret_dir):
                    os.makedirs(secret_dir)

                noise_secret_dir = os.path.join(save_dir,'degrade_secret','noise')
                if not os.path.exists(noise_secret_dir):
                    os.makedirs(noise_secret_dir)
                blur_secret_dir = os.path.join(save_dir,'degrade_secret','blur')
                if not os.path.exists(blur_secret_dir):
                    os.makedirs(blur_secret_dir)
                lr_secret_dir = os.path.join(save_dir,'degrade_secret','lr')
                if not os.path.exists(lr_secret_dir):
                    os.makedirs(lr_secret_dir)

                cover_dir = os.path.join(save_dir,'cover')
                if not os.path.exists(cover_dir):
                    os.makedirs(cover_dir)
                
                noise_stego_dir = os.path.join(save_dir,'stego','noise')
                if not os.path.exists(noise_stego_dir):
                    os.makedirs(noise_stego_dir)
                blur_stego_dir = os.path.join(save_dir,'stego','blur')
                if not os.path.exists(blur_stego_dir):
                    os.makedirs(blur_stego_dir)
                lr_stego_dir = os.path.join(save_dir,'stego','lr')
                if not os.path.exists(lr_stego_dir):
                    os.makedirs(lr_stego_dir)
                
                denoise_stego_dir = os.path.join(save_dir,'processed_stego','denoise')
                if not os.path.exists(denoise_stego_dir):
                    os.makedirs(denoise_stego_dir)
                deblur_stego_dir = os.path.join(save_dir,'processed_stego','deblur')
                if not os.path.exists(deblur_stego_dir):
                    os.makedirs(deblur_stego_dir)
                sr_stego_dir = os.path.join(save_dir,'processed_stego','sr')
                if not os.path.exists(sr_stego_dir):
                    os.makedirs(sr_stego_dir)

                denoise_steg_img.save(os.path.join(denoise_stego_dir,'%03d.png' % i))
                deblur_steg_img.save(os.path.join(deblur_stego_dir,'%03d.png' % i))
                sr_steg_img.save(os.path.join(sr_stego_dir,'%03d.png' % i))

                denoise_secret_img.save(os.path.join(denoise_rec_dir,'%03d.png' % i))
                deblur_secret_img.save(os.path.join(deblur_rec_dir,'%03d.png' % i))
                sr_secret_img.save(os.path.join(sr_rec_dir,'%03d.png' % i))
    
                secret_img.save(os.path.join(secret_dir,'%03d.png' % i))
                cover_img.save(os.path.join(cover_dir,'%03d.png' % i))

                noise_steg_img.save(os.path.join(noise_stego_dir,'%03d.png' % i))
                blur_steg_img.save(os.path.join(blur_stego_dir,'%03d.png' % i))
                lr_steg_img.save(os.path.join(lr_stego_dir,'%03d.png' % i))

                noise_secret_img.save(os.path.join(noise_secret_dir,'%03d.png' % i))
                blur_secret_img.save(os.path.join(blur_secret_dir,'%03d.png' % i))
                lr_secret_img.save(os.path.join(lr_secret_dir,'%03d.png' % i))

            

                i=i+1


    logger_val.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tLPIPS_N: {lpips_n.avg:.6f} |"
        f"\tLPIPS_B: {lpips_b.avg:.6f} |"
        f"\tLPIPS_S: {lpips_s.avg:.6f} |"
        f"\tLPIPSORI_N: {lpipsori_n.avg:.6f} |"
        f"\tLPIPSORI_B: {lpipsori_b.avg:.6f} |"
        f"\tLPIPSORI_S: {lpipsori_s.avg:.6f} |"
        f"\tLPIPS_NS: {lpips_ns.avg:.6f} |"
        f"\tLPIPS_BS: {lpips_bs.avg:.6f} |"
        f"\tLPIPS_SS: {lpips_ss.avg:.6f} |"
        f"\tPSNRC_N: {psnrc_n.avg:.6f} |"
        f"\tPSNRS_N: {psnrs_n.avg:.6f} |"
        f"\tSSIMC_N: {ssimc_n.avg:.6f} |"
        f"\tSSIMS_N: {ssims_n.avg:.6f} |"
        f"\tPSNRORI_N: {psnrori_n.avg:.6f} |"
        f"\tSSIMORI_N: {ssimori_n.avg:.6f} |"
        f"\tPSNRC_B: {psnrc_b.avg:.6f} |"
        f"\tPSNRS_B: {psnrs_b.avg:.6f} |"
        f"\tSSIMC_B: {ssimc_b.avg:.6f} |"
        f"\tSSIMS_B: {ssims_b.avg:.6f} |"
        f"\tPSNRORI_B: {psnrori_b.avg:.6f} |"
        f"\tSSIMORI_B: {ssimori_b.avg:.6f} |"
        f"\tPSNRC_S: {psnrc_s.avg:.6f} |"
        f"\tPSNRS_S: {psnrs_s.avg:.6f} |"
        f"\tSSIMC_S: {ssimc_s.avg:.6f} |"
        f"\tSSIMS_S: {ssims_s.avg:.6f} |"
        f"\tPSNRORI_S: {psnrori_s.avg:.6f} |"
        f"\tSSIMORI_S: {ssimori_s.avg:.6f} |"
    )

    return loss.avg

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        dest_filename = filename.replace(filename.split('/')[-1], "_checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, dest_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-d_test", "--test_dataset", type=str, required=True, help="Testing dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=2,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(224,224),
        help="Size of the training patches to be cropped (default: %(default)s)",
    ),
    parser.add_argument(
        "--test-patch-size",
        type=int,
        nargs=2,
        default=(1024, 1024),
        help="Size of the testing patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--hide_checkpoint", type=str, help="Path to a LIH checkpoint"),
    parser.add_argument("--checkpoint", type=str, help="Path to a LSR checkpoint"),
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    ),
    parser.add_argument(
        "--val_freq", type=int,  default=30, help="how often should an evaluation be performed"
    ),
    parser.add_argument(
        "--channels_in", type=int,  default=12, help="channels into punet"
    ),
    parser.add_argument(
        "--num_step", type=int,  default=12, help="num of lifting steps in LIH"
    ),
    parser.add_argument(
        "--klvl", type=int,  default=3, help="num of scales in LSR"
    ),
    parser.add_argument(
        "--mid", type=int,  default=2,help="middle_blk_num in SRM"
    ),
    parser.add_argument(
        "--enc", default = [2,2,4], nargs='+', type=int, help="enc_blk_num in SRM"
    ),
    parser.add_argument(
        "--dec", default = [2,2,2], nargs='+', type=int, help="dec_blk_num in SRM"
    ),
    parser.add_argument(
        "--save_img", action="store_true", default=False, help="Save model to disk"
    )
    parser.add_argument(
        "--sp1", type=int,  default=2, help='num of noisy samples for training in one batch'
    ),
    parser.add_argument(
        "--sp2", type=int,  default=2,help='num of blur samples for training in one batch'
    ),
    parser.add_argument(
        "--sp3", type=int,  default=4,help='num of lr samples for training in one batch'
    ),
    parser.add_argument(
        "--test", action="store_true", default=False, help="test"
    ),
    parser.add_argument(
        "--finetune", action="store_true", default=False, help="train LIH and LSR in an endtoend manner"
    ),
    parser.add_argument(
        "--std", type=float,  default=1.6, help="Standard deviation"
    ),
    parser.add_argument(
        "--sweight", type=float,  default=2, help="weight of restoration loss"
    ),
    parser.add_argument(
        "--cweight", type=float,  default=1, help="weight of security loss"
    ),
    parser.add_argument(
        "--pweight_c", type=float,  default=0.01,help="weight of perceptual loss"
    ),
    parser.add_argument(
        "--lfrestore", action="store_true", default=True, help="Save model to disk"
    ),
    parser.add_argument(
        "--steps", type=int,  default=4, help="num of wlblocks in each scale of LSR"
    ),
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    if not os.path.exists(os.path.join('experiments', args.experiment)):
        os.makedirs(os.path.join('experiments', args.experiment))

    setup_logger('train', os.path.join('experiments', args.experiment), 'train_' + args.experiment,
                      level=logging.INFO,
                      screen=True, tofile=True)
    setup_logger('val', os.path.join('experiments', args.experiment), 'val_' + args.experiment,
                      level=logging.INFO,
                      screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')

    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('experiments', args.experiment, 'checkpoints'))


    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.test_patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="", transform=train_transforms)
    test_dataset = ImageFolder(args.test_dataset, split="", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    #denoise net
    hide_net = hide_model(pin_ch=args.channels_in,uin_ch=args.channels_in,num_step=args.num_step)
    hide_net = hide_net.to(device)

    denoise_net = restore_model(steps = args.steps,klvl=args.klvl,mid=args.mid,enc=args.enc,dec=args.dec,lfrestore=args.lfrestore)
    denoise_net = denoise_net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        denoise_net = CustomDataParallel(denoise_net)
    logger_train.info(args)
    logger_train.info(denoise_net)   

    #load hide net
    state_dicts = torch.load(args.hide_checkpoint, map_location=device)   
    hide_net.load_state_dict(state_dicts['state_dict'])

    hide_optimizer = configure_optimizers(hide_net, args)
    denoise_optimizer = configure_optimizers(denoise_net, args)

    criterion = LSR_Loss()
    lpips_fn = lpips.LPIPS(net='alex',version='0.1')
    lpips_fn.cuda()
    last_epoch = 0
    loss = float("inf")
    best_loss = float("inf")
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint= torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        denoise_net.load_state_dict(checkpoint["state_dict"])
        denoise_optimizer.load_state_dict(checkpoint["optimizer"])
        denoise_optimizer.param_groups[0]['lr'] = args.learning_rate
    
    if not args.test:
        for epoch in range(last_epoch, args.epochs):
            logger_train.info(f"Learning rate: {denoise_optimizer.param_groups[0]['lr']}")
            train_one_epoch(
                hide_net,
                denoise_net,
                criterion,
                train_dataloader,
                hide_optimizer,
                denoise_optimizer,
                epoch,
                logger_train,
                tb_logger,
                args
            )
            if epoch % args.val_freq == 0:
                loss = test_epoch(args, epoch, test_dataloader, hide_net, denoise_net,logger_val,criterion,lpips_fn)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": denoise_net.state_dict(),
                        "loss": loss,
                        "optimizer": denoise_optimizer.state_dict(),
                    },
                    is_best,
                    os.path.join('experiments', args.experiment, 'checkpoints', "net_checkpoint.pth.tar")
                )
                if args.finetune:
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": hide_net.state_dict(),
                            "best_loss":best_loss,
                            "optimizer": hide_optimizer.state_dict(),
                        },
                        is_best,
                        os.path.join('experiments', args.experiment, 'hide_checkpoints', "net_checkpoint.pth.tar")
                    )
                if is_best:
                    logger_val.info('best checkpoint saved.')
    else:
        loss = test_epoch(args, 0, test_dataloader, hide_net, denoise_net,logger_val,criterion,lpips_fn)


if __name__ == "__main__":
    main(sys.argv[1:])
