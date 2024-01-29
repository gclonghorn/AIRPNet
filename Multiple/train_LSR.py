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
from LIH import LIH_stage1,LIH_stage2
from LSR import LSR
from GM import GM
from tqdm import tqdm
import lpips


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = 0.01 * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
                


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def downsample(hr,scale):
    lr = F.interpolate(hr, scale_factor=1.0/scale, mode='bicubic')
    lr = F.interpolate(lr, scale_factor=scale, mode='bicubic')
    return lr

def guass_blur(hr,k_sz,sigma):
    transform = transforms.GaussianBlur(kernel_size=k_sz,sigma=sigma)
    return transform(hr)
    


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

    a = torch.from_numpy(a.copy()).float().unsqueeze(0).cuda()
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0).cuda()
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
    hide_model1, hide_model2, hide_model3, denoise_model, criterion, train_dataloader, hide_optimizer1,  hide_optimizer2,  denoise_optimizer, epoch, logger_train, tb_logger, args
):
    if args.finetune:
        hide_model1.train()
        hide_model2.train()
        for param in hide_model3.parameters():
            param.requires_grad=False
    else:
        for param in hide_model1.parameters():
            param.requires_grad=False
        for param in hide_model2.parameters():
            param.requires_grad=False
        for param in hide_model3.parameters():
            param.requires_grad=False
            
    denoise_model.train()
   
    device = next(hide_model1.parameters()).device
    dwt = DWT()
    iwt = IWT()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        cover = d[:d.shape[0] // 3, :, :, :]  
        secret_1 = d[d.shape[0] // 3: 2 * (d.shape[0] // 3), :, :, :]
        secret_2 = d[2 * (d.shape[0] // 3): 3 * (d.shape[0] // 3), :, :, :]
        ns,bs,ss = torch.split(secret_1,[args.sp1,args.sp2,args.sp3],dim=0)
        noiselvl = np.random.uniform(0,55,size=1)
        noise = torch.cuda.FloatTensor(ns.size()).normal_(mean=0, std=noiselvl[0] / 255.)  
        noise_secret = ns + noise 
        blur_secret = guass_blur(bs,2*random.randint(0,11)+3,random.uniform(0.1,2))
        scalelvl = random.choice([2,4])
        lr_secret = downsample(ss,scalelvl)  
        input_secret_1 = torch.cat([noise_secret,blur_secret,lr_secret],dim=0)
        
        ns,bs,ss = torch.split(secret_2,[args.sp1,args.sp2,args.sp3],dim=0)
        noiselvl = np.random.uniform(0,55,size=1) 
        noise = torch.cuda.FloatTensor(ns.size()).normal_(mean=0, std=noiselvl[0] / 255.) 
        noise_secret = ns + noise 
        blur_secret = guass_blur(bs,2*random.randint(0,11)+3,random.uniform(0.1,2))
        scalelvl = random.choice([2,4])
        lr_secret = downsample(ss,scalelvl)  
        input_secret_2 = torch.cat([noise_secret,blur_secret,lr_secret],dim=0)

        cover_dwt = dwt(cover)
        secret_dwt_1 = dwt(input_secret_1)
        secret_dwt_2 = dwt(input_secret_2)

        if args.finetune:
            hide_optimizer1.zero_grad()
            hide_optimizer2.zero_grad()
        denoise_optimizer.zero_grad()    
        #################
        # hide secret 1#
        #################
        steg_dwt_1, z_dwt_1 = hide_model1(cover_dwt,secret_dwt_1)
        #get steg 1
        steg_1 = iwt(steg_dwt_1)

        #################
        #hide secret 2#
        #################
        if args.guiding_map:
            if args.update_gm:
                imp = hide_model3(cover, input_secret_1, steg_1)
            else:
                imp  = torch.zeros(cover.shape).cuda()
            imp_dwt = dwt(imp)
            steg_dwt_1 = torch.cat((steg_dwt_1,imp_dwt), 1) 
            
        output_dwt_2, z_dwt_2 = hide_model2(steg_dwt_1,secret_dwt_2)
      
        #get steg 2
        steg_dwt_2 = output_dwt_2.narrow(1, 0, 12)
        steg_2 = iwt(steg_dwt_2)   
        
        #################
        #    denoise    #
        #################  
         
        steg_2_clean = denoise_model(steg_2,args.sp1,args.sp2,args.sp3)
        steg_dwt_2_clean = dwt(steg_2_clean)
        output_dwt_2_clean =  torch.cat((steg_dwt_2_clean, gauss_noise(z_dwt_1.shape)), 1) 

        #################
        #reveal secret 2#
        #################
        z_guass_1 = gauss_noise(z_dwt_1.shape) 
        z_guass_2 = gauss_noise(z_dwt_2.shape) 
        if args.guiding_map:
            output_rev_dwt_1, secret_rev_dwt_2= hide_model2(output_dwt_2_clean, z_guass_2,rev=True) 
        else:
             output_rev_dwt_1, secret_rev_dwt_2= hide_model2(steg_dwt_2_clean, z_guass_2,rev=True) 
        steg_rev_dwt_1 = output_rev_dwt_1.narrow(1, 0, 12)
        steg_rev_1 = iwt(steg_rev_dwt_1)
        secret_rev_2 = iwt(secret_rev_dwt_2)

        #################
        #reveal secret 1#
        #################
        cover_rev_dwt_1, secret_rev_dwt_1= hide_model1(steg_rev_dwt_1, z_guass_1,rev=True)
        cover_rev_1 = iwt(cover_rev_dwt_1)
        secret_rev_1 = iwt(secret_rev_dwt_1)

        
        #loss
        out_criterian = criterion(secret_1,secret_2,cover,steg_2_clean,secret_rev_1,secret_rev_2,args.sweight1,args.sweight2,args.cweight2,\
            args.pweight_c,args.sp1,args.sp2,args.sp3)
        loss = out_criterian['loss']
        percep_loss = out_criterian['percep_loss']
        pixel_loss = out_criterian['pixel_loss']
        loss.backward()
        denoise_optimizer.step()
        if args.finetune:
            hide_optimizer1.step()
            hide_optimizer2.step()

        if i % 10 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {loss.item():.3f} |'
                f'\tpercep_Loss: {percep_loss.item():.3f} |'
                f'\tpixel_Loss: {pixel_loss.item():.3f} |'
        
            )
    tb_logger.add_scalar('{}'.format('[train]: loss'), loss.item(), epoch)


def test_epoch(args,epoch, test_dataloader, hide_model1, hide_model2,hide_model3, denoise_model,logger_val,tb_logger,criterion,lpips_fn):
    dwt = DWT()
    iwt = IWT()
    hide_model1.eval()
    hide_model2.eval()
    hide_model3.eval()
    denoise_model.eval()
    device = next(hide_model1.parameters()).device
    psnrc_b = AverageMeter()
    ssimc_b = AverageMeter()
    lpipsc_b = AverageMeter()
    psnrs_1_b = AverageMeter()
    ssims_1_b = AverageMeter()
    lpipss_1_b = AverageMeter()
    psnrs_2_b = AverageMeter()
    ssims_2_b = AverageMeter()
    lpipss_2_b = AverageMeter()
    psnrc_ori_1_b = AverageMeter()
    ssimc_ori_1_b = AverageMeter()
    lpipsc_ori_1_b = AverageMeter()
    psnrc_ori_2_b = AverageMeter()
    ssimc_ori_2_b = AverageMeter()
    lpipsc_ori_2_b = AverageMeter()
     
    psnrc_n = AverageMeter()
    ssimc_n = AverageMeter()
    lpipsc_n = AverageMeter()
    psnrs_1_n = AverageMeter()
    ssims_1_n = AverageMeter()
    lpipss_1_n = AverageMeter()
    psnrs_2_n = AverageMeter()
    ssims_2_n = AverageMeter()
    lpipss_2_n = AverageMeter()
    psnrc_ori_1_n = AverageMeter()
    ssimc_ori_1_n = AverageMeter()
    lpipsc_ori_1_n = AverageMeter()
    psnrc_ori_2_n = AverageMeter()
    ssimc_ori_2_n = AverageMeter()
    lpipsc_ori_2_n = AverageMeter()
    
    psnrc_l = AverageMeter()
    ssimc_l = AverageMeter()
    lpipsc_l = AverageMeter()
    psnrs_1_l = AverageMeter()
    ssims_1_l = AverageMeter()
    lpipss_1_l = AverageMeter()
    psnrs_2_l = AverageMeter()
    ssims_2_l = AverageMeter()
    lpipss_2_l = AverageMeter()
    psnrc_ori_1_l = AverageMeter()
    ssimc_ori_1_l = AverageMeter()
    lpipsc_ori_1_l = AverageMeter()
    psnrc_ori_2_l = AverageMeter()
    ssimc_ori_2_l = AverageMeter()
    lpipsc_ori_2_l = AverageMeter()
    loss = AverageMeter()
    
    i=0
    with torch.no_grad():
        for i, d in tqdm(enumerate(test_dataloader)):
            d = d.to(device)
            cover = d[:d.shape[0] // 3, :, :, :] 
            secret_1 = d[d.shape[0] // 3: 2 * (d.shape[0] // 3), :, :, :]
            secret_2 = d[2 * (d.shape[0] // 3): 3 * (d.shape[0] // 3), :, :, :]
            #degrade secret 1
            blur_secret_img = guass_blur(secret_1,15,args.std)
            input_secret_1_b = blur_secret_img
    
            noise = torch.cuda.FloatTensor(secret_1.size()).normal_(mean=0, std=25 / 255.)  
            noise_secret_img = secret_1 + noise 
            input_secret_1_n = noise_secret_img

            scalelvl = 4
            lr_secret_img = downsample(secret_1,scalelvl)
            input_secret_1_l = lr_secret_img
            
            # degrade secret 2
            blur_secret_img = guass_blur(secret_2,15,args.std)
            input_secret_2_b = blur_secret_img
             
            noise = torch.cuda.FloatTensor(secret_2.size()).normal_(mean=0, std=25/ 255.) 
            noise_secret_img = secret_2 + noise 
            input_secret_2_n = noise_secret_img

            scalelvl = 4
            lr_secret_img = downsample(secret_2,scalelvl)
            input_secret_2_l = lr_secret_img             

            input_secret_1 = torch.cat([input_secret_1_n,input_secret_1_b,input_secret_1_l],dim=0)
            input_secret_2 = torch.cat([input_secret_2_n,input_secret_2_b,input_secret_2_l],dim=0)
            cover_dwt = dwt(cover)
            secret_dwt_1 = dwt(input_secret_1)
            secret_dwt_2 = dwt(input_secret_2)

            #################
            # hide secret 1#
            #################
            steg_dwt_1, z_dwt_1 = hide_model1(cover_dwt,secret_dwt_1)
            #get steg 1
            steg_1 = iwt(steg_dwt_1)
            steg_1_n,steg_1_b,steg_1_l =  torch.split(steg_1,1,dim=0)

            #################
            #hide secret 2#
            #################
            if args.guiding_map:
                if args.update_gm:
                    imp_n = hide_model3(cover, input_secret_1_n, steg_1_n)
                    imp_b = hide_model3(cover, input_secret_1_b, steg_1_b)
                    imp_l = hide_model3(cover, input_secret_1_l, steg_1_l)
                else:
                    imp_n  = torch.zeros(cover.shape).cuda()
                    imp_b  = torch.zeros(cover.shape).cuda()
                    imp_l  = torch.zeros(cover.shape).cuda()
                imp_dwt = dwt(torch.cat([imp_n,imp_b,imp_l],dim=0))
                
                steg_dwt_1 = torch.cat((steg_dwt_1,imp_dwt), 1)  
                
            output_dwt_2, z_dwt_2 = hide_model2(steg_dwt_1,secret_dwt_2)
        
            #get steg 2
            steg_dwt_2 = output_dwt_2.narrow(1, 0, 12)
            steg_2 = iwt(steg_dwt_2)  
            steg_2_n,steg_2_b,steg_2_l =  torch.split(steg_2,1,dim=0) 
            
            #################
            #    denoise    #
            #################  
            
            steg_2_clean = denoise_model(steg_2,sp1=1,sp2=1,sp3=1)
            steg_2_clean_n,steg_2_clean_b,steg_2_clean_l = torch.split(steg_2_clean,1,dim=0)
            steg_dwt_2_clean = dwt(steg_2_clean)
            output_dwt_2_clean =  torch.cat((steg_dwt_2_clean, gauss_noise(z_dwt_1.shape)), 1) 

            #################
            #reveal secret 2#
            #################
            z_guass_1 = gauss_noise(z_dwt_1.shape) 
            z_guass_2 = gauss_noise(z_dwt_2.shape)
            if args.guiding_map:
                output_rev_dwt_1, secret_rev_dwt_2= hide_model2(output_dwt_2_clean, z_guass_2,rev=True) 
            else:
                output_rev_dwt_1, secret_rev_dwt_2= hide_model2(steg_dwt_2_clean, z_guass_2,rev=True) 
            steg_rev_dwt_1 = output_rev_dwt_1.narrow(1, 0, 12)
    
            secret_rev_2 = iwt(secret_rev_dwt_2)
            secret_rev_2_n, secret_rev_2_b, secret_rev_2_l = torch.split(secret_rev_2,1,dim=0)

            #################
            #reveal secret 1#
            #################
            cover_rev_dwt_1, secret_rev_dwt_1= hide_model1(steg_rev_dwt_1, z_guass_1,rev=True)
            secret_rev_1 = iwt(secret_rev_dwt_1)
            secret_rev_1_n, secret_rev_1_b, secret_rev_1_l = torch.split(secret_rev_1,1,dim=0)
            
            
            #loss
            out_criterian = criterion(secret_1,secret_2,cover,steg_2_clean,secret_rev_1,secret_rev_2,args.sweight1,args.sweight2,args.cweight2,\
                args.pweight_c)
            loss.update(out_criterian['loss'])
       
            lpips = lpips_fn.forward(cover,steg_2_clean_b)
            lpipsc_b.update(lpips.mean().item())
            lpips = lpips_fn.forward(cover,steg_1_b)
            lpipsc_ori_1_b.update(lpips.mean().item())
            lpips = lpips_fn.forward(cover,steg_2_b)
            lpipsc_ori_2_b.update(lpips.mean().item())
            lpips = lpips_fn.forward(secret_1,secret_rev_1_b)
            lpipss_1_b.update(lpips.mean().item())
            lpips = lpips_fn.forward(secret_2,secret_rev_2_b)
            lpipss_2_b.update(lpips.mean().item())
            
            lpips = lpips_fn.forward(cover,steg_2_clean_n)
            lpipsc_n.update(lpips.mean().item())
            lpips = lpips_fn.forward(cover,steg_1_n)
            lpipsc_ori_1_n.update(lpips.mean().item())
            lpips = lpips_fn.forward(cover,steg_2_n)
            lpipsc_ori_2_n.update(lpips.mean().item())
            lpips = lpips_fn.forward(secret_1,secret_rev_1_n)
            lpipss_1_n.update(lpips.mean().item())
            lpips = lpips_fn.forward(secret_2,secret_rev_2_n)
            lpipss_2_n.update(lpips.mean().item())
            
            lpips = lpips_fn.forward(cover,steg_2_clean_l)
            lpipsc_l.update(lpips.mean().item())
            lpips = lpips_fn.forward(cover,steg_1_l)
            lpipsc_ori_1_l.update(lpips.mean().item())
            lpips = lpips_fn.forward(cover,steg_2_l)
            lpipsc_ori_2_l.update(lpips.mean().item())
            lpips = lpips_fn.forward(secret_1,secret_rev_1_l)
            lpipss_1_l.update(lpips.mean().item())
            lpips = lpips_fn.forward(secret_2,secret_rev_2_l)
            lpipss_2_l.update(lpips.mean().item())
   
            
            #compute psnr and save image
            save_dir = os.path.join('experiments', args.experiment,'images')
            
            secret_img1 = torch2img(secret_1)
            secret_img2 = torch2img(secret_2)
            cover_img = torch2img(cover)

            steg_img_b  = torch2img(steg_2_clean_b)
            steg_img_ori_1_b = torch2img(steg_1_b)
            steg_img_ori_2_b = torch2img(steg_2_b)
            lq_secret_img_1_b = torch2img(input_secret_1_b)
            lq_secret_img_2_b = torch2img(input_secret_2_b)
            hq_secret_img_1_b= torch2img(secret_rev_1_b)
            hq_secret_img_2_b = torch2img(secret_rev_2_b)
            
            steg_img_n  = torch2img(steg_2_clean_n)
            steg_img_ori_1_n = torch2img(steg_1_n)
            steg_img_ori_2_n = torch2img(steg_2_n)
            lq_secret_img_1_n = torch2img(input_secret_1_n)
            lq_secret_img_2_n = torch2img(input_secret_2_n)
            hq_secret_img_1_n= torch2img(secret_rev_1_n)
            hq_secret_img_2_n = torch2img(secret_rev_2_n)
            
            steg_img_l  = torch2img(steg_2_clean_l)
            steg_img_ori_1_l = torch2img(steg_1_l)
            steg_img_ori_2_l = torch2img(steg_2_l)
            lq_secret_img_1_l = torch2img(input_secret_1_l)
            lq_secret_img_2_l = torch2img(input_secret_2_l)
            hq_secret_img_1_l= torch2img(secret_rev_1_l)
            hq_secret_img_2_l = torch2img(secret_rev_2_l)
            
            imp_img_b = torch2img(imp_b)
            
            p1, m1 = compute_metrics(hq_secret_img_1_b, secret_img1)
            psnrs_1_b.update(p1)
            ssims_1_b.update(m1)
            p1, m1 = compute_metrics(hq_secret_img_2_b, secret_img2)
            psnrs_2_b.update(p1)
            ssims_2_b.update(m1)
            p2, m2 = compute_metrics(steg_img_b, cover_img)
            psnrc_b.update(p2)
            ssimc_b.update(m2)
            p3, m3 = compute_metrics(steg_img_ori_1_b, cover_img)
            psnrc_ori_1_b.update(p3)
            ssimc_ori_1_b.update(m3)
            p3, m3 = compute_metrics(steg_img_ori_2_b, cover_img)
            psnrc_ori_2_b.update(p3)
            ssimc_ori_2_b.update(m3)
            
            p1, m1 = compute_metrics(hq_secret_img_1_n, secret_img1)
            psnrs_1_n.update(p1)
            ssims_1_n.update(m1)
            p1, m1 = compute_metrics(hq_secret_img_2_n, secret_img2)
            psnrs_2_n.update(p1)
            ssims_2_n.update(m1)
            p2, m2 = compute_metrics(steg_img_n, cover_img)
            psnrc_n.update(p2)
            ssimc_n.update(m2)
            p3, m3 = compute_metrics(steg_img_ori_1_n, cover_img)
            psnrc_ori_1_n.update(p3)
            ssimc_ori_1_n.update(m3)
            p3, m3 = compute_metrics(steg_img_ori_2_n, cover_img)
            psnrc_ori_2_n.update(p3)
            ssimc_ori_2_n.update(m3)
            
            p1, m1 = compute_metrics(hq_secret_img_1_l, secret_img1)
            psnrs_1_l.update(p1)
            ssims_1_l.update(m1)
            p1, m1 = compute_metrics(hq_secret_img_2_l, secret_img2)
            psnrs_2_l.update(p1)
            ssims_2_l.update(m1)
            p2, m2 = compute_metrics(steg_img_l, cover_img)
            psnrc_l.update(p2)
            ssimc_l.update(m2)
            p3, m3 = compute_metrics(steg_img_ori_1_l, cover_img)
            psnrc_ori_1_l.update(p3)
            ssimc_ori_1_l.update(m3)
            p3, m3 = compute_metrics(steg_img_ori_2_l, cover_img)
            psnrc_ori_2_l.update(p3)
            ssimc_ori_2_l.update(m3)
          
            if args.save_img:
                secret_dir1 = os.path.join(save_dir,'secret1')
                if not os.path.exists(secret_dir1):
                    os.makedirs(secret_dir1)   
                secret_dir2 = os.path.join(save_dir,'secret2')
                if not os.path.exists(secret_dir2):
                    os.makedirs(secret_dir2)
                cover_dir = os.path.join(save_dir,'cover')
                if not os.path.exists(cover_dir):
                    os.makedirs(cover_dir)
                    
                deblur_rec_dir1_b = os.path.join(save_dir,'rec1','deblur')
                if not os.path.exists(deblur_rec_dir1_b):
                    os.makedirs(deblur_rec_dir1_b)
                deblur_rec_dir2_b = os.path.join(save_dir,'rec2','deblur')
                if not os.path.exists(deblur_rec_dir2_b):
                    os.makedirs(deblur_rec_dir2_b)
                blur_secret_dir1_b = os.path.join(save_dir,'degrade_secret1','blur')
                if not os.path.exists(blur_secret_dir1_b):
                    os.makedirs(blur_secret_dir1_b)
                blur_secret_dir2_b = os.path.join(save_dir,'degrade_secret2','blur')
                if not os.path.exists(blur_secret_dir2_b):
                    os.makedirs(blur_secret_dir2_b)                           
                blur_stego_dir1_b = os.path.join(save_dir,'stego1','blur')
                if not os.path.exists(blur_stego_dir1_b):
                    os.makedirs(blur_stego_dir1_b)
                blur_stego_dir2_b = os.path.join(save_dir,'stego2','blur')
                if not os.path.exists(blur_stego_dir2_b):
                    os.makedirs(blur_stego_dir2_b)
                deblur_stego_dir2_b = os.path.join(save_dir,'processed_stego2','blur')
                if not os.path.exists(deblur_stego_dir2_b):
                    os.makedirs(deblur_stego_dir2_b)
                    
                denoise_rec_dir1_n = os.path.join(save_dir,'rec1','denoise')
                if not os.path.exists(denoise_rec_dir1_n):
                    os.makedirs(denoise_rec_dir1_n)
                denoise_rec_dir2_n = os.path.join(save_dir,'rec2','denoise')
                if not os.path.exists(denoise_rec_dir2_n):
                    os.makedirs(denoise_rec_dir2_n)
                noise_secret_dir1_n = os.path.join(save_dir,'degrade_secret1','noise')
                if not os.path.exists(noise_secret_dir1_n):
                    os.makedirs(noise_secret_dir1_n)
                noise_secret_dir2_n = os.path.join(save_dir,'degrade_secret2','noise')
                if not os.path.exists(noise_secret_dir2_n):
                    os.makedirs(noise_secret_dir2_n)                           
                noise_stego_dir1_n = os.path.join(save_dir,'stego1','noise')
                if not os.path.exists(noise_stego_dir1_n):
                    os.makedirs(noise_stego_dir1_n)
                noise_stego_dir2_n = os.path.join(save_dir,'stego2','noise')
                if not os.path.exists(noise_stego_dir2_n):
                    os.makedirs(noise_stego_dir2_n)
                denoise_stego_dir2_n = os.path.join(save_dir,'processed_stego2','denoise')
                if not os.path.exists(denoise_stego_dir2_n):
                    os.makedirs(denoise_stego_dir2_n)
                    
                sr_rec_dir1_l = os.path.join(save_dir,'rec1','sr')
                if not os.path.exists(sr_rec_dir1_l):
                    os.makedirs(sr_rec_dir1_l)
                sr_rec_dir2_l = os.path.join(save_dir,'rec2','sr')
                if not os.path.exists(sr_rec_dir2_l):
                    os.makedirs(sr_rec_dir2_l)
                lr_secret_dir1_l = os.path.join(save_dir,'degrade_secret1','lr')
                if not os.path.exists(lr_secret_dir1_l):
                    os.makedirs(lr_secret_dir1_l)
                lr_secret_dir2_l = os.path.join(save_dir,'degrade_secret2','lr')
                if not os.path.exists(lr_secret_dir2_l):
                    os.makedirs(lr_secret_dir2_l)                           
                lr_stego_dir1_l = os.path.join(save_dir,'stego1','lr')
                if not os.path.exists(lr_stego_dir1_l):
                    os.makedirs(lr_stego_dir1_l)
                lr_stego_dir2_l = os.path.join(save_dir,'stego2','lr')
                if not os.path.exists(lr_stego_dir2_l):
                    os.makedirs(lr_stego_dir2_l)
                sr_stego_dir2_l = os.path.join(save_dir,'processed_stego2','sr')
                if not os.path.exists(sr_stego_dir2_l):
                    os.makedirs(sr_stego_dir2_l)
                    
                imp_dir_b = os.path.join(save_dir,'imp','blur')
                if not os.path.exists(imp_dir_b):
                    os.makedirs(imp_dir_b)
          
                secret_img1.save(os.path.join(secret_dir1,'%03d.png' % i))
                secret_img2.save(os.path.join(secret_dir2,'%03d.png' % i))
                cover_img.save(os.path.join(cover_dir,'%03d.png' % i))
                
                steg_img_b.save(os.path.join(deblur_stego_dir2_b,'%03d.png' % i))  
                hq_secret_img_1_b.save(os.path.join(deblur_rec_dir1_b,'%03d.png' % i))
                hq_secret_img_2_b.save(os.path.join(deblur_rec_dir2_b,'%03d.png' % i))
                steg_img_ori_1_b.save(os.path.join(blur_stego_dir1_b,'%03d.png' % i))
                steg_img_ori_2_b.save(os.path.join(blur_stego_dir2_b,'%03d.png' % i))
                lq_secret_img_1_b.save(os.path.join(blur_secret_dir1_b,'%03d.png' % i))
                lq_secret_img_2_b.save(os.path.join(blur_secret_dir2_b,'%03d.png' % i))
                
                steg_img_n.save(os.path.join(denoise_stego_dir2_n,'%03d.png' % i))  
                hq_secret_img_1_n.save(os.path.join(denoise_rec_dir1_n,'%03d.png' % i))
                hq_secret_img_2_n.save(os.path.join(denoise_rec_dir2_n,'%03d.png' % i))
                steg_img_ori_1_n.save(os.path.join(noise_stego_dir1_n,'%03d.png' % i))
                steg_img_ori_2_n.save(os.path.join(noise_stego_dir2_n,'%03d.png' % i))
                lq_secret_img_1_n.save(os.path.join(noise_secret_dir1_n,'%03d.png' % i))
                lq_secret_img_2_n.save(os.path.join(noise_secret_dir2_n,'%03d.png' % i))
                
                steg_img_l.save(os.path.join(sr_stego_dir2_l,'%03d.png' % i))  
                hq_secret_img_1_l.save(os.path.join(sr_rec_dir1_l,'%03d.png' % i))
                hq_secret_img_2_l.save(os.path.join(sr_rec_dir2_l,'%03d.png' % i))
                steg_img_ori_1_l.save(os.path.join(lr_stego_dir1_l,'%03d.png' % i))
                steg_img_ori_2_l.save(os.path.join(lr_stego_dir2_l,'%03d.png' % i))
                lq_secret_img_1_l.save(os.path.join(lr_secret_dir1_l,'%03d.png' % i))
                lq_secret_img_2_l.save(os.path.join(lr_secret_dir2_l,'%03d.png' % i))
                
                imp_img_b.save(os.path.join(imp_dir_b,'%03d.png' % i))

                i=i+1


    logger_val.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tPSNR_C_b: {psnrc_b.avg:.6f} |"
        f"\tSSIM_C_b: {ssimc_b.avg:.6f} |"
        f"\tLPIPS_C_b: {lpipsc_b.avg:.6f} |" 
        f"\tPSNR_S1_b: {psnrs_1_b.avg:.6f} |"
        f"\tSSIM_S1_b: {ssims_1_b.avg:.6f} |"
        f"\tLPIPS_S1_b: {lpipss_1_b.avg:.6f} |"
        f"\tPSNR_S2_b: {psnrs_2_b.avg:.6f} |"
        f"\tSSIM_S2_b: {ssims_2_b.avg:.6f} |"
        f"\tLPIPS_S2_b: {lpipss_2_b.avg:.6f} |" 
        f"\tPSNR_CORI1_b: {psnrc_ori_1_b.avg:.6f} |"
        f"\tSSIM_CORI1_b: {ssimc_ori_1_b.avg:.6f} |"
        f"\tLPIPS_CORI1_b: {lpipsc_ori_1_b.avg:.6f} |"
        f"\tPSNR_CORI2_b: {psnrc_ori_2_b.avg:.6f} |"
        f"\tSSIM_CORI2_b: {ssimc_ori_2_b.avg:.6f} |"
        f"\tLPIPS_CORI2_b: {lpipsc_ori_2_b.avg:.6f} |"

        f"\tPSNR_C_n: {psnrc_n.avg:.6f} |"
        f"\tSSIM_C_n: {ssimc_n.avg:.6f} |"
        f"\tLPIPS_C_n: {lpipsc_n.avg:.6f} |" 
        f"\tPSNR_S1_n: {psnrs_1_n.avg:.6f} |"
        f"\tSSIM_S1_n: {ssims_1_n.avg:.6f} |"
        f"\tLPIPS_S1_n: {lpipss_1_n.avg:.6f} |"
        f"\tPSNR_S2_n: {psnrs_2_n.avg:.6f} |"
        f"\tSSIM_S2_n: {ssims_2_n.avg:.6f} |"
        f"\tLPIPS_S2_n: {lpipss_2_n.avg:.6f} |"
        f"\tPSNR_CORI1_n: {psnrc_ori_1_n.avg:.6f} |"
        f"\tSSIM_CORI1_n: {ssimc_ori_1_n.avg:.6f} |"
        f"\tLPIPS_CORI1_n: {lpipsc_ori_1_n.avg:.6f} |"
        f"\tPSNR_CORI2_n: {psnrc_ori_2_n.avg:.6f} |"
        f"\tSSIM_CORI2_n: {ssimc_ori_2_n.avg:.6f} |"
        f"\tLPIPS_CORI2_n: {lpipsc_ori_2_n.avg:.6f} |"
        
        f"\tPSNR_C_l: {psnrc_l.avg:.6f} |"
        f"\tSSIM_C_l: {ssimc_l.avg:.6f} |"
        f"\tLPIPS_C_l: {lpipsc_l.avg:.6f} |" 
        f"\tPSNR_S1_l: {psnrs_1_l.avg:.6f} |"
        f"\tSSIM_S1_l: {ssims_1_l.avg:.6f} |"
        f"\tLPIPS_S1_l: {lpipss_1_l.avg:.6f} |"
        f"\tPSNR_S2_l: {psnrs_2_l.avg:.6f} |"
        f"\tSSIM_S2_l: {ssims_2_l.avg:.6f} |"
        f"\tLPIPS_S2_l: {lpipss_2_l.avg:.6f} |"
        f"\tPSNR_CORI1_l: {psnrc_ori_1_l.avg:.6f} |"
        f"\tSSIM_CORI1_l: {ssimc_ori_1_l.avg:.6f} |"
        f"\tLPIPS_CORI1_l: {lpipsc_ori_1_l.avg:.6f} |"
        f"\tPSNR_CORI2_l: {psnrc_ori_2_l.avg:.6f} |"
        f"\tSSIM_CORI2_l: {ssimc_ori_2_l.avg:.6f} |"
        f"\tLPIPS_CORI2_l: {lpipsc_ori_2_l.avg:.6f} |"
    )
    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    return loss.avg

                   
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        dest_filename = filename.replace(".pth.tar", "_checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, dest_filename)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

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
        default=3,
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
    parser.add_argument(
        "--finetune", action="store_true", default=False, help="train LIH and LSR in an endtoend manner"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a LSR checkpoint")
    parser.add_argument("--hide_checkpoint1", type=str, help="Path to a LIH1")
    parser.add_argument("--hide_checkpoint2", type=str, help="Path to a LIH2")
    parser.add_argument("--hide_checkpoint3", type=str, help="Path to a GM")
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    )
    parser.add_argument(
        "--channels_in", type=int,  default=3,
    )
    parser.add_argument(
        "--val_freq", type=int,  default=30,
    )
    parser.add_argument(
        "--klvl", type=int,  default=3,
    )
    parser.add_argument(
        "--steps", type=int,  default=4, help="num of wlblocks in each scale of LSR"
    )
    parser.add_argument(
        "--num_step", type=int,  default=12, help="num of lifting steps in LIH"
    )
    parser.add_argument(
        "--mid", type=int,  default=6,
    )
    parser.add_argument(
        "--enc", default = [2,2,4], nargs='+', type=int,
    )
    parser.add_argument(
        "--dec", default = [2,2,2], nargs='+', type=int,
    )
    parser.add_argument(
        "--save_img", action="store_true", default=False, help="Save images "
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="test"
    )
    parser.add_argument(
        "--std", type=float,  default=1.6,
    )
    parser.add_argument(
        "--sweight1", type=float,  default=2,
    )
    parser.add_argument(
        "--sweight2", type=float,  default=2,
    )
    parser.add_argument(
        "--cweight1", type=float,  default=0,
    )
    parser.add_argument(
        "--cweight2", type=float,  default=1,
    )
    parser.add_argument(
        "--pweight_c", type=float,  default=0,
    )
    parser.add_argument(
        "--lfrestore", action="store_true", default=True, help="Whether to use LPM"
    )
    parser.add_argument("--update_gm", type=bool, default=True, help="Whether to update parameters of guiding module")
    parser.add_argument("--guiding_map", type=bool, default=True, help="Whether to use guiding module")
    parser.add_argument("--sp1", default = 1,type=int,help="num of noisy samples in one batch")
    parser.add_argument("--sp2", default = 1,type=int,help="num of blur samples in one batch")
    parser.add_argument("--sp3", default = 1,type=int,help="num of lr samples in one batch")
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
        
    if not os.path.exists(os.path.join('experiments', args.experiment, 'hide_checkpoints')):
        os.makedirs(os.path.join('experiments', args.experiment, 'hide_checkpoints'))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)


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
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        drop_last=True
    )

    hide_net1 = LIH_stage1(args.num_step)
    hide_net1 = hide_net1.to(device)
    hide_net2 = LIH_stage2(args.num_step,guiding_map=args.guiding_map)
    hide_net2 = hide_net2.to(device)
    hide_net3 = GM()
    hide_net3 = hide_net3.to(device)
    denoise_net = LSR(steps = args.steps,klvl=args.klvl,mid=args.mid,enc=args.enc,dec=args.dec,lfrestore=args.lfrestore)
    denoise_net = denoise_net.to(device)
    init_model(denoise_net)

    if args.cuda and torch.cuda.device_count() > 1:
        denoise_net = CustomDataParallel(denoise_net)
    logger_train.info(args)
    
    lpips_fn = lpips.LPIPS(net='alex',version='0.1')
    lpips_fn.cuda()

    state_dicts1 = torch.load(args.hide_checkpoint1, map_location=device)  
    hide_net1.load_state_dict(state_dicts1['state_dict'])
    
    state_dicts2 = torch.load(args.hide_checkpoint2, map_location=device)  
    hide_net2.load_state_dict(state_dicts2['state_dict'])
    
    state_dicts3 = torch.load(args.hide_checkpoint3, map_location=device)  
    hide_net3.load_state_dict(state_dicts3['state_dict'])

    hide_optimizer1 = configure_optimizers(hide_net1, args)
    hide_optimizer2 = configure_optimizers(hide_net2, args)
    denoise_optimizer = configure_optimizers(denoise_net, args)

    criterion = LSR_Loss()
    
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
                hide_net1,
                hide_net2,
                hide_net3,
                denoise_net,
                criterion,
                train_dataloader,
                hide_optimizer1,
                hide_optimizer2,
                denoise_optimizer,
                epoch,
                logger_train,
                tb_logger,
                args
            )
            if epoch % args.val_freq == 0:
                loss = test_epoch(args, epoch, test_dataloader, hide_net1, hide_net2, hide_net3,denoise_net,logger_val,tb_logger,criterion,loss_fn)   
            if epoch % 100 == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": denoise_net.state_dict(),
                        "best_loss":best_loss,
                        "optimizer": denoise_optimizer.state_dict(),
                    },
                    False,
                    os.path.join('experiments', args.experiment, 'checkpoints', "net_checkpoint_"+str(epoch)+".pth.tar")
                )
                if args.finetune:
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": hide_net1.state_dict(),
                            "best_loss":best_loss,
                            "optimizer": hide_optimizer1.state_dict(),
                        },
                        False,
                        os.path.join('experiments', args.experiment, 'hide_checkpoints', "net1_checkpoint_"+str(epoch)+".pth.tar")
                    )
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": hide_net2.state_dict(),
                            "best_loss":best_loss,
                            "optimizer": hide_optimizer2.state_dict(),
                        },
                        False,
                        os.path.join('experiments', args.experiment, 'hide_checkpoints', "net2_checkpoint_"+str(epoch)+".pth.tar")
                    )
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": denoise_net.state_dict(),
                        "best_loss":best_loss,
                        "optimizer": denoise_optimizer.state_dict(),
                    },
                    is_best,
                    os.path.join('experiments', args.experiment, 'checkpoints', "net_checkpoint.pth.tar")
                )
                if args.finetune:
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": hide_net1.state_dict(),
                            "best_loss":best_loss,
                            "optimizer": hide_optimizer1.state_dict(),
                        },
                        is_best,
                        os.path.join('experiments', args.experiment, 'hide_checkpoints', "net1_checkpoint.pth.tar")
                    )
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": hide_net2.state_dict(),
                            "best_loss":best_loss,
                            "optimizer": hide_optimizer2.state_dict(),
                        },
                        is_best,
                        os.path.join('experiments', args.experiment, 'hide_checkpoints', "net2_checkpoint.pth.tar")
                    )
                if is_best:
                    logger_val.info('best checkpoint saved.')
    else:
        loss = test_epoch(args, 0, test_dataloader, hide_net1, hide_net2, hide_net3,denoise_net,logger_val,tb_logger,criterion,lpips_fn)
     
if __name__ == "__main__":
    main(sys.argv[1:])
