
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
from losses import LIH_Loss,imp_loss

import logging
import numpy as np
import PIL.Image as Image
from torchvision.transforms import ToPILImage
from pytorch_msssim import ms_ssim
from typing import Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from LIH import LIH_stage1,LIH_stage2
from GM import GM
from util import DWT,IWT,setup_logger
from tqdm import tqdm






def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = 0.01 * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
                
def init_net3(mod):
    for key, param in mod.named_parameters():
        if param.requires_grad:
            param.data = 0.1 * torch.randn(param.data.shape).cuda()
            

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


def configure_optimizers(net, lr):

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    return optimizer


def train_one_epoch(
    model1, model2, model3, criterion, train_dataloader, hide_optimizer1, hide_optimizer2, imp_optimizer,  epoch, logger_train, tb_logger, args
):
    model1.train()
    model2.train()
    model3.train()
    device = next(model1.parameters()).device
    dwt = DWT()
    iwt = IWT()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        cover = d[:d.shape[0] // 3, :, :, :]  
        secret_1 = d[d.shape[0] // 3: 2 * (d.shape[0] // 3), :, :, :]
        secret_2 = d[2 * (d.shape[0] // 3): 3 * (d.shape[0] // 3), :, :, :]

        noiselvl = np.random.uniform(0, 50, size=1)
        noise = torch.cuda.FloatTensor(secret_1.size()).normal_(mean=0, std=noiselvl[0] / 255.)
        noise_secret_img_1 = secret_1 + noise

        noiselvl = np.random.uniform(0, 50, size=1)
        noise = torch.cuda.FloatTensor(secret_2.size()).normal_(mean=0, std=noiselvl[0] / 255.)
        noise_secret_img_2 = secret_2 + noise

        blur_secret_img_1 = guass_blur(noise_secret_img_1, 2 * random.randint(0, 11) + 3,random.uniform(0.1,2))
        blur_secret_img_2 = guass_blur(noise_secret_img_2, 2 * random.randint(0, 11) + 3,random.uniform(0.1,2))

        scalelvl = random.choice([2, 4])
        lr_secret_img_1 = downsample(blur_secret_img_1, scalelvl)
        lr_secret_img_2 = downsample(blur_secret_img_2, scalelvl)

        input_secret_1 = lr_secret_img_1
        input_secret_2 = lr_secret_img_2        

        cover_dwt = dwt(cover)
        secret_dwt_1 = dwt(input_secret_1)
        secret_dwt_2 = dwt(input_secret_2)

        hide_optimizer1.zero_grad()
        hide_optimizer2.zero_grad()
        if args.update_gm:
            imp_optimizer.zero_grad()

        #################
        # hide secret 1#
        #################
        steg_dwt_1, z_dwt_1 = model1(cover_dwt,secret_dwt_1)
        #get steg 1
        steg_1 = iwt(steg_dwt_1)

        #################
        #hide secret 2#
        #################
        if args.guiding_map:
            if args.update_gm:
                imp = model3(cover, input_secret_1, steg_1)
            else:
                imp  = torch.zeros(cover.shape).cuda()
            imp_dwt = dwt(imp)
            
            steg_dwt_1 = torch.cat((steg_dwt_1,imp_dwt), 1)  #24
            
        output_dwt_2, z_dwt_2 = model2(steg_dwt_1,secret_dwt_2)
        #get steg 2
        steg_dwt_2 = output_dwt_2.narrow(1, 0, 12)
        steg_2 = iwt(steg_dwt_2)        

        #################
        #reveal secret 2#
        #################
        z_guass_1 = gauss_noise(z_dwt_1.shape) #12
        z_guass_2 = gauss_noise(z_dwt_2.shape) #12
        z_guass_3 = gauss_noise(z_dwt_2.shape) #12
        if args.guiding_map:
            output_rev_dwt_1, secret_rev_dwt_2= model2(torch.cat((steg_dwt_2,z_guass_3),1), z_guass_2,rev=True) 
        else:
             output_rev_dwt_1, secret_rev_dwt_2= model2(steg_dwt_2, z_guass_2,rev=True) 
        steg_rev_dwt_1 = output_rev_dwt_1.narrow(1, 0, 12)
        steg_rev_1 = iwt(steg_rev_dwt_1)
        secret_rev_2 = iwt(secret_rev_dwt_2)

        #################
        #reveal secret 1#
        #################
        cover_rev_dwt_1, secret_rev_dwt_1= model1(steg_rev_dwt_1, z_guass_1,rev=True)
        cover_rev_1 = iwt(cover_rev_dwt_1)
        secret_rev_1 = iwt(secret_rev_dwt_1)
        

        #loss
        steg_dwt_1_low = steg_dwt_1.narrow(1, 0, 3)
        steg_dwt_2_low = steg_dwt_2.narrow(1, 0, 3)
        cover_dwt_low = cover_dwt.narrow(1, 0, 3)

        out_criterian = criterion(input_secret_1,input_secret_2,secret_rev_1,secret_rev_2, \
        cover,steg_1,steg_2, \
        steg_dwt_1_low,steg_dwt_2_low,cover_dwt_low, \
        args.rec_weight_1,args.rec_weight_2,args.guide_weight_1,args.guide_weight_2, \
        args.freq_weight_1,args.freq_weight_2)
        hide_loss = out_criterian['hide_loss']
        if args.update_gm:
            impmap_loss = imp_loss(imp, cover - steg_1)
            total_loss = hide_loss+0.01*impmap_loss
        total_loss = hide_loss 
        total_loss.backward()
        hide_optimizer1.step()
        hide_optimizer2.step()
        if args.update_gm:
            imp_optimizer.step()
        if i % 10 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\thide loss: {hide_loss.item():.3f} |'
        
            )
    tb_logger.add_scalar('{}'.format('[train]: hide_loss'), hide_loss.item(), epoch)


def test_epoch(args, epoch, test_dataloader, model1, model2, model3, criterion, logger_val, tb_logger):
    dwt = DWT()
    iwt = IWT()
    model1.eval()
    model2.eval()
    model3.eval()
    device = next(model1.parameters()).device

    psnr_cover_1 = AverageMeter()
    psnr_cover_2 = AverageMeter()
    psnr_secret_1 = AverageMeter()
    psnr_secret_2 = AverageMeter()
    ssim_cover_1 = AverageMeter()
    ssim_cover_2 = AverageMeter()
    ssim_secret_1 = AverageMeter()
    ssim_secret_2 = AverageMeter()
    loss = AverageMeter()

    with torch.no_grad():
        for i, d in tqdm(enumerate(test_dataloader)):
            d = d.to(device)
            cover = d[:d.shape[0] // 3, :, :, :]
            secret_1 = d[d.shape[0] // 3: 2 * (d.shape[0] // 3), :, :, :]
            secret_2 = d[2 * (d.shape[0] // 3): 3 * (d.shape[0] // 3), :, :, :]

            def degrade_image(image):
                noise_level = 15
                noise = torch.cuda.FloatTensor(image.size()).normal_(mean=0, std=noise_level / 255.)
                noisy_image = image + noise

                blurred_image = guass_blur(noisy_image, 11, 1.2)

                downsampled_image = downsample(blurred_image, 2)

                return downsampled_image

            input_secret_1 = degrade_image(secret_1)
            input_secret_2 = degrade_image(secret_2)

            cover_dwt = dwt(cover)
            secret_dwt_1 = dwt(input_secret_1)
            secret_dwt_2 = dwt(input_secret_2)

            #################
            # hide secret 1 #
            #################
            steg_dwt_1, z_dwt_1 = model1(cover_dwt, secret_dwt_1)
            steg_1 = iwt(steg_dwt_1)

            #################
            # hide secret 2 #
            #################
            if args.guiding_map:
                if args.update_gm:
                    imp = model3(cover, input_secret_1, steg_1)
                else:
                    imp = torch.zeros(cover.shape).cuda()
                imp_dwt = dwt(imp)
                steg_dwt_1 = torch.cat((steg_dwt_1, imp_dwt), 1)
            output_dwt_2, z_dwt_2 = model2(steg_dwt_1, secret_dwt_2)

            steg_dwt_2 = output_dwt_2.narrow(1, 0, 12)
            steg_2 = iwt(steg_dwt_2)

            #################
            # reveal secret 2 #
            #################
            z_guass_1 = gauss_noise(z_dwt_1.shape)
            z_guass_2 = gauss_noise(z_dwt_2.shape)
            z_guass_3 = gauss_noise(z_dwt_2.shape)
            if args.guiding_map:
                output_rev_dwt_1, secret_rev_dwt_2 = model2(torch.cat((steg_dwt_2, z_guass_3), 1), z_guass_2, rev=True)
            else:
                output_rev_dwt_1, secret_rev_dwt_2 = model2(steg_dwt_2, z_guass_2, rev=True)
            steg_rev_dwt_1 = output_rev_dwt_1.narrow(1, 0, 12)
            steg_rev_1 = iwt(steg_rev_dwt_1)
            secret_rev_2 = iwt(secret_rev_dwt_2)

            #################
            # reveal secret 1 #
            #################
            cover_rev_dwt_1, secret_rev_dwt_1 = model1(steg_rev_dwt_1, z_guass_1, rev=True)
            cover_rev_1 = iwt(cover_rev_dwt_1)
            secret_rev_1 = iwt(secret_rev_dwt_1)

            steg_dwt_1_low = steg_dwt_1.narrow(1, 0, 3)
            steg_dwt_2_low = steg_dwt_2.narrow(1, 0, 3)
            cover_dwt_low = cover_dwt.narrow(1, 0, 3)

            out_criterian = criterion(input_secret_1, input_secret_2, secret_rev_1, secret_rev_2,
                                      cover, steg_1, steg_2,
                                      steg_dwt_1_low, steg_dwt_2_low, cover_dwt_low,
                                      args.rec_weight_1, args.rec_weight_2, args.guide_weight_1, args.guide_weight_2,
                                      args.freq_weight_1, args.freq_weight_2)
            loss.update(out_criterian['hide_loss'])

            cover_img = torch2img(cover)

            input_secret_img_1 = torch2img(input_secret_1)
            steg_img_1 = torch2img(steg_1)
            secret_rev_img_1 = torch2img(secret_rev_1)
            p1, m1 = compute_metrics(secret_rev_img_1, input_secret_img_1)
            psnr_secret_1.update(p1)
            ssim_secret_1.update(m1)
            p2, m2 = compute_metrics(steg_img_1, cover_img)
            psnr_cover_1.update(p2)
            ssim_cover_1.update(m2)

            input_secret_img_2 = torch2img(input_secret_2)
            steg_img_2 = torch2img(steg_2)
            secret_rev_img_2 = torch2img(secret_rev_2)
            p1, m1 = compute_metrics(secret_rev_img_2, input_secret_img_2)
            psnr_secret_2.update(p1)
            ssim_secret_2.update(m1)
            p2, m2 = compute_metrics(steg_img_2, cover_img)
            psnr_cover_2.update(p2)
            ssim_cover_2.update(m2)

            if args.save_images:
                save_dir = os.path.join('experiments', args.experiment, 'images')
                os.makedirs(save_dir, exist_ok=True)
                cover_dir = os.path.join(save_dir, 'cover')
                secret1_dir = os.path.join(save_dir, 'secret1')
                stego1_dir = os.path.join(save_dir, 'stego1')
                rev1_dir = os.path.join(save_dir, 'rev1')
                secret2_dir = os.path.join(save_dir, 'secret2')
                stego2_dir = os.path.join(save_dir, 'stego2')
                rev2_dir = os.path.join(save_dir, 'rev2')

                os.makedirs(cover_dir, exist_ok=True)
                os.makedirs(secret1_dir, exist_ok=True)
                os.makedirs(stego1_dir, exist_ok=True)
                os.makedirs(rev1_dir, exist_ok=True)
                os.makedirs(secret2_dir, exist_ok=True)
                os.makedirs(stego2_dir, exist_ok=True)
                os.makedirs(rev2_dir, exist_ok=True)

                cover_img.save(os.path.join(save_dir, 'cover', f'{i:03d}.png'))
                input_secret_img_1.save(os.path.join(save_dir, 'secret1', f'{i:03d}.png'))
                steg_img_1.save(os.path.join(save_dir, 'stego1', f'{i:03d}.png'))
                secret_rev_img_1.save(os.path.join(save_dir, 'rev1', f'{i:03d}.png'))
                input_secret_img_2.save(os.path.join(save_dir, 'secret2', f'{i:03d}.png'))
                steg_img_2.save(os.path.join(save_dir, 'stego2', f'{i:03d}.png'))
                secret_rev_img_2.save(os.path.join(save_dir, 'rev2', f'{i:03d}.png'))

    logger_val.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tPSNR Cover 1: {psnr_cover_1.avg:.6f} |"
        f"\tSSIM Cover 1: {ssim_cover_1.avg:.6f} |"
        f"\tPSNR Cover 2: {psnr_cover_2.avg:.6f} |"
        f"\tSSIM Cover 2: {ssim_cover_2.avg:.6f} |"
        f"\tPSNR Secret 1: {psnr_secret_1.avg:.6f} |"
        f"\tSSIM Secret 1: {ssim_secret_1.avg:.6f} |"
        f"\tPSNR Secret 2: {psnr_secret_2.avg:.6f} |"
        f"\tSSIM Secret 2: {ssim_secret_2.avg:.6f} |"
    )

    return loss.avg


    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        dest_filename = filename.replace(".pth.tar", "_checkpoint_best_loss.pth.tar")
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
        "-lr1",
        "--learning-rate-1",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-lr2",
        "--learning-rate-2",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-lr3",
        "--learning-rate-3",
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
        "--batch-size", type=int, default=24, help="Batch size (default: %(default)s)"
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
        default=(224, 224),
        help="Size of the training patches to be cropped (default: %(default)s)",
    )
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
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint-1", type=str, help="Path to a checkpoint1"),
    parser.add_argument("--checkpoint-2", type=str, help="Path to a checkpoint2"),
    parser.add_argument("--checkpoint-3", type=str, help="Path to a checkpoint3"),
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    ),
    parser.add_argument("--channel-in", type=int, default= 12,help="channels into punet"),
    parser.add_argument("--num-steps", type=int, help="num of WLBlocks in LIH"),
    parser.add_argument("--rec-weight-1", default = 3.0,type=float),
    parser.add_argument("--rec-weight-2", default = 3.0,type=float),
    parser.add_argument("--guide-weight-1", default = 1.0,type=float),
    parser.add_argument("--guide-weight-2", default = 1.0,type=float),
    parser.add_argument("--freq-weight-1", default = 0,type=float),
    parser.add_argument("--freq-weight-2", default = 0,type=float),
    parser.add_argument("--data-type", default = [1,2,3], nargs='+', type=int),
    parser.add_argument("--val-freq", default = 30, type=int),
    parser.add_argument(
        "--save-images", action="store_true", default=False, help="Save images to disk"
    ),
    parser.add_argument("--nrate", default = 0.6,type=float),
    parser.add_argument("--lrate", default = 0.2,type=float),
    parser.add_argument("--brate", default = 0.2,type=float),
    parser.add_argument("--test_type1", default = 2,type=int,help='test type of 1st secret blur=1,noise=2,lr=3'),
    parser.add_argument("--test_type2", default = 2,type=int,help='test type of 2nd secret blur=1,noise=2,lr=3'),
    parser.add_argument("--update_gm", type=bool, default=True, help="Whether to update parameters of guiding module")
    parser.add_argument("--guiding_map", type=bool, default=True, help="Whether to use guiding module")
    parser.add_argument("--test", action="store_true", help="test")
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

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size),transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.test_patch_size),transforms.ToTensor()]
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
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    net1 = LIH_stage1(args.num_steps)
    net1 = net1.to(device)
    net2 = LIH_stage2(args.num_steps,guiding_map=args.guiding_map)
    net2 = net2.to(device)
    net3 = GM()
    net3 = net3.to(device)
    
    init_model(net1)
    init_model(net2)
    init_net3(net3)

    if args.cuda and torch.cuda.device_count() > 1:
        net1 = CustomDataParallel(net1)
        net2 = CustomDataParallel(net2)
        net3 = CustomDataParallel(net3)
    logger_train.info(args)

    optimizer1 = configure_optimizers(net1, args.learning_rate_1)
    optimizer2 = configure_optimizers(net2, args.learning_rate_2)
    optimizer3 = configure_optimizers(net3, args.learning_rate_3)
    criterion = LIH_Loss()
    
    last_epoch = 0
    loss = float("inf")
    if args.checkpoint_1:  # load from previous checkpoint
        print("Loading", args.checkpoint_1)
        checkpoint_1= torch.load(args.checkpoint_1, map_location=device)
        last_epoch = checkpoint_1["epoch"] + 1
        best_loss = checkpoint_1["best_loss"]
        net1.load_state_dict(checkpoint_1["state_dict"])
        optimizer1.load_state_dict(checkpoint_1["optimizer"])
        optimizer1.param_groups[0]['lr'] = args.learning_rate_1
    
    if args.checkpoint_2:
        print("Loading", args.checkpoint_2)   
        checkpoint_2= torch.load(args.checkpoint_2, map_location=device)
        net2.load_state_dict(checkpoint_2["state_dict"])
        optimizer2.load_state_dict(checkpoint_2["optimizer"])
        optimizer2.param_groups[0]['lr'] = args.learning_rate_2
    
    if args.checkpoint_3:
        print("Loading", args.checkpoint_3)   
        checkpoint_3= torch.load(args.checkpoint_3, map_location=device)
        net3.load_state_dict(checkpoint_3["state_dict"])
        optimizer3.load_state_dict(checkpoint_3["optimizer"])
        optimizer3.param_groups[0]['lr'] = args.learning_rate_3

    
    if not args.test:
        best_loss = float("inf")
        for epoch in range(last_epoch, args.epochs):
            logger_train.info(f"Learning rate1: {optimizer1.param_groups[0]['lr']}")
            logger_train.info(f"Learning rate2: {optimizer2.param_groups[0]['lr']}")
            logger_train.info(f"Learning rate3: {optimizer3.param_groups[0]['lr']}")
            train_one_epoch(
                net1,
                net2,
                net3,
                criterion,
                train_dataloader,
                optimizer1,
                optimizer2,
                optimizer3,
                epoch,
                logger_train,
                tb_logger,
                args
            )
            if epoch % args.val_freq == 0:
                loss = test_epoch(args, epoch, test_dataloader, net1, net2, net3, criterion,logger_val,tb_logger)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net1.state_dict(),
                        "optimizer": optimizer1.state_dict(),
                        "best_loss":best_loss,
                    },
                    is_best,
                    os.path.join('experiments', args.experiment, 'checkpoints', "net_1.pth.tar")
                )
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net2.state_dict(),
                        "optimizer": optimizer2.state_dict(),
                    },
                    is_best,
                    os.path.join('experiments', args.experiment, 'checkpoints', "net_2.pth.tar")
                )
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net3.state_dict(),
                        "optimizer": optimizer3.state_dict(),
                    },
                    is_best,
                    os.path.join('experiments', args.experiment, 'checkpoints', "net_3.pth.tar")
                )
                if is_best:
                    logger_val.info('best checkpoint saved.')
    else:
        loss = test_epoch(args, 0, test_dataloader, net1, net2, net3, criterion,logger_val,tb_logger)


if __name__ == "__main__":
    main(sys.argv[1:])
