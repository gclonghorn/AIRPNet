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

def guass_blur(hr,k_sz):
    transform = transforms.GaussianBlur(kernel_size=k_sz)
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

        p = np.array([args.brate,args.nrate,args.lrate])
        type = np.random.choice(args.data_type,p=p.ravel())
        type1 = type
        type2 = type

        if type1 == 1:
            #blur
            blur_secret_img = guass_blur(secret_1,2*random.randint(0,11)+3)
            input_secret_1 = blur_secret_img
        
        elif type1 == 2:
            #add noise
            noiselvl = np.random.uniform(0,55,size=1) 
            noise = torch.cuda.FloatTensor(secret_1.size()).normal_(mean=0, std=noiselvl[0] / 255.)  
            noise_secret_img = secret_1 + noise
            input_secret_1 = noise_secret_img

        else:
            #down sample to low resolution
            scalelvl = random.choice([2,4])
            lr_secret_img = downsample(secret_1,scalelvl)
            input_secret_1 = lr_secret_img

        if type2 == 1:
            #blur
            blur_secret_img = guass_blur(secret_2,2*random.randint(0,11)+3)
            input_secret_2 = blur_secret_img
        
        elif type2 == 2:
            #add noise
            noiselvl = np.random.uniform(0,55,size=1) 
            noise = torch.cuda.FloatTensor(secret_2.size()).normal_(mean=0, std=noiselvl[0] / 255.) 
            noise_secret_img = secret_2 + noise 
            input_secret_2 = noise_secret_img

        else:
            #down sample to low resolution
            scalelvl = random.choice([2,4])
            lr_secret_img = downsample(secret_2,scalelvl)
            input_secret_2 = lr_secret_img             

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


def test_epoch(args,epoch, test_dataloader, model1, model2,model3, criterion,logger_val, tb_logger):
    dwt = DWT()
    iwt = IWT()
    model1.eval()
    model2.eval()
    model3.eval()
    device = next(model1.parameters()).device
    psnrc_1_b = AverageMeter()
    psnrs_1_b = AverageMeter()
    ssimc_1_b = AverageMeter()
    ssims_1_b = AverageMeter()
    psnrc_2_b = AverageMeter()
    psnrs_2_b = AverageMeter()
    ssimc_2_b = AverageMeter()
    ssims_2_b = AverageMeter()
    
    psnrc_1_n = AverageMeter()
    psnrs_1_n = AverageMeter()
    ssimc_1_n = AverageMeter()
    ssims_1_n = AverageMeter()
    psnrc_2_n = AverageMeter()
    psnrs_2_n = AverageMeter()
    ssimc_2_n = AverageMeter()
    ssims_2_n = AverageMeter()

    psnrc_1_l = AverageMeter()
    psnrs_1_l = AverageMeter()
    ssimc_1_l = AverageMeter()
    ssims_1_l = AverageMeter()
    psnrc_2_l = AverageMeter()
    psnrs_2_l = AverageMeter()
    ssimc_2_l = AverageMeter()
    ssims_2_l = AverageMeter()
    loss = AverageMeter()   

    i=0
    with torch.no_grad():
        for i, d in tqdm(enumerate(test_dataloader)):
            d = d.to(device)
            cover = d[:d.shape[0] // 3, :, :, :]  
            secret_1 = d[d.shape[0] // 3: 2 * (d.shape[0] // 3), :, :, :]
            secret_2 = d[2 * (d.shape[0] // 3): 3 * (d.shape[0] // 3), :, :, :]

            p = np.array([args.brate,args.nrate,args.lrate])
         
            blur_secret_img = guass_blur(secret_1,15)
            input_secret_1_b = blur_secret_img
            
            noise = torch.cuda.FloatTensor(secret_1.size()).normal_(mean=0, std=25 / 255.)  
            noise_secret_img = secret_1 + noise 
            input_secret_1_n = noise_secret_img

            scalelvl = 4
            lr_secret_img = downsample(secret_1,scalelvl)
            input_secret_1_l = lr_secret_img
            
            blur_secret_img = guass_blur(secret_2,15)
            input_secret_2_b = blur_secret_img
             
            #add noise
            noise = torch.cuda.FloatTensor(secret_2.size()).normal_(mean=0, std=25/ 255.) 
            noise_secret_img = secret_2 + noise 
            input_secret_2_n = noise_secret_img

            #down sample to low resolution
            scalelvl = 4
            lr_secret_img = downsample(secret_2,scalelvl)
            input_secret_2_l = lr_secret_img             

            cover_dwt = dwt(cover)
            secret_dwt_1_b = dwt(input_secret_1_b)
            secret_dwt_2_b = dwt(input_secret_2_b)
            secret_dwt_1_n = dwt(input_secret_1_n)
            secret_dwt_2_n = dwt(input_secret_2_n)
            secret_dwt_1_l = dwt(input_secret_1_l)
            secret_dwt_2_l = dwt(input_secret_2_l)

            #################
            # hide secret 1#
            #################
            steg_dwt_1_b, z_dwt_1 = model1(cover_dwt,secret_dwt_1_b)
            steg_dwt_1_n, z_dwt_1 = model1(cover_dwt,secret_dwt_1_n)
            steg_dwt_1_l, z_dwt_1 = model1(cover_dwt,secret_dwt_1_l)
            #get steg 1
            steg_1_b = iwt(steg_dwt_1_b)
            steg_1_n = iwt(steg_dwt_1_n)
            steg_1_l = iwt(steg_dwt_1_l)

            #################
            #hide secret 2#
            #################
            if args.guiding_map:
                    if args.update_gm:
                        imp_b = model3(cover, input_secret_1_b, steg_1_b)
                        imp_n = model3(cover, input_secret_1_n, steg_1_n)
                        imp_l = model3(cover, input_secret_1_l, steg_1_l)
                    else:
                        imp_b  = torch.zeros(cover.shape).cuda()
                        imp_n  = torch.zeros(cover.shape).cuda()
                        imp_l  = torch.zeros(cover.shape).cuda()
                    imp_dwt_b = dwt(imp_b)
                    imp_dwt_n = dwt(imp_n)
                    imp_dwt_l = dwt(imp_l)
                    
                    steg_dwt_1_b = torch.cat((steg_dwt_1_b,imp_dwt_b), 1)       
                    steg_dwt_1_n = torch.cat((steg_dwt_1_n,imp_dwt_n), 1)       
                    steg_dwt_1_l = torch.cat((steg_dwt_1_l,imp_dwt_l), 1)               
            output_dwt_2_b, z_dwt_2 = model2(steg_dwt_1_b,secret_dwt_2_b)
            output_dwt_2_n, z_dwt_2 = model2(steg_dwt_1_n,secret_dwt_2_n)
            output_dwt_2_l, z_dwt_2 = model2(steg_dwt_1_l,secret_dwt_2_l)
        
            #get steg 2
            steg_dwt_2_b = output_dwt_2_b.narrow(1, 0, 12)
            steg_2_b = iwt(steg_dwt_2_b)        
            steg_dwt_2_n = output_dwt_2_n.narrow(1, 0, 12)
            steg_2_n = iwt(steg_dwt_2_n)     
            steg_dwt_2_l = output_dwt_2_l.narrow(1, 0, 12)
            steg_2_l = iwt(steg_dwt_2_l)     

            #################
            #reveal secret 2#
            #################
            z_guass_1 = gauss_noise(z_dwt_1.shape) 
            z_guass_2 = gauss_noise(z_dwt_2.shape) 
            z_guass_3 = gauss_noise(z_dwt_2.shape) 
            if args.guiding_map:
                output_rev_dwt_1_b, secret_rev_dwt_2_b= model2(torch.cat((steg_dwt_2_b,z_guass_3),1), z_guass_2,rev=True) 
                output_rev_dwt_1_n, secret_rev_dwt_2_n= model2(torch.cat((steg_dwt_2_n,z_guass_3),1), z_guass_2,rev=True) 
                output_rev_dwt_1_l, secret_rev_dwt_2_l= model2(torch.cat((steg_dwt_2_l,z_guass_3),1), z_guass_2,rev=True) 
            else:
                output_rev_dwt_1_b, secret_rev_dwt_2_b= model2(steg_dwt_2_b, z_guass_2,rev=True) 
                output_rev_dwt_1_n, secret_rev_dwt_2_n= model2(steg_dwt_2_n, z_guass_2,rev=True) 
                output_rev_dwt_1_l, secret_rev_dwt_2_l= model2(steg_dwt_2_l, z_guass_2,rev=True) 
            steg_rev_dwt_1_b = output_rev_dwt_1_b.narrow(1, 0, 12)
            steg_rev_dwt_1_n = output_rev_dwt_1_n.narrow(1, 0, 12)
            steg_rev_dwt_1_l = output_rev_dwt_1_l.narrow(1, 0, 12)
            steg_rev_1_b = iwt(steg_rev_dwt_1_b)
            secret_rev_2_b = iwt(secret_rev_dwt_2_b)
            steg_rev_1_n = iwt(steg_rev_dwt_1_n)
            secret_rev_2_n = iwt(secret_rev_dwt_2_n)
            steg_rev_1_l = iwt(steg_rev_dwt_1_l)
            secret_rev_2_l = iwt(secret_rev_dwt_2_l)

            #################
            #reveal secret 1#
            #################
            cover_rev_dwt_1_b, secret_rev_dwt_1_b= model1(steg_rev_dwt_1_b, z_guass_1,rev=True)
            cover_rev_1_b = iwt(cover_rev_dwt_1_b)
            secret_rev_1_b = iwt(secret_rev_dwt_1_b)
            
            cover_rev_dwt_1_n, secret_rev_dwt_1_n= model1(steg_rev_dwt_1_n, z_guass_1,rev=True)
            cover_rev_1_n = iwt(cover_rev_dwt_1_n)
            secret_rev_1_n = iwt(secret_rev_dwt_1_n)
            
            cover_rev_dwt_1_l, secret_rev_dwt_1_l= model1(steg_rev_dwt_1_l, z_guass_1,rev=True)
            cover_rev_1_l = iwt(cover_rev_dwt_1_l)
            secret_rev_1_l= iwt(secret_rev_dwt_1_l)

            #loss
            steg_dwt_1_low_b = steg_dwt_1_b.narrow(1, 0, 3)
            steg_dwt_2_low_b = steg_dwt_2_b.narrow(1, 0, 3)
            cover_dwt_low = cover_dwt.narrow(1, 0, 3)

            out_criterian_b = criterion(input_secret_1_b,input_secret_2_b,secret_rev_1_b,secret_rev_2_b, \
            cover,steg_1_b,steg_2_b, \
            steg_dwt_1_low_b,steg_dwt_2_low_b,cover_dwt_low, \
            args.rec_weight_1,args.rec_weight_2,args.guide_weight_1,args.guide_weight_2, \
            args.freq_weight_1,args.freq_weight_2)
            loss.update(out_criterian_b['hide_loss'])

            steg_dwt_1_low_n = steg_dwt_1_n.narrow(1, 0, 3)
            steg_dwt_2_low_n = steg_dwt_2_n.narrow(1, 0, 3)
            cover_dwt_low = cover_dwt.narrow(1, 0, 3)

            out_criterian_n = criterion(input_secret_1_n,input_secret_2_n,secret_rev_1_n,secret_rev_2_n, \
            cover,steg_1_n,steg_2_n, \
            steg_dwt_1_low_n,steg_dwt_2_low_n,cover_dwt_low, \
            args.rec_weight_1,args.rec_weight_2,args.guide_weight_1,args.guide_weight_2, \
            args.freq_weight_1,args.freq_weight_2)
            loss.update(out_criterian_n['hide_loss'])

            steg_dwt_1_low_l = steg_dwt_1_l.narrow(1, 0, 3)
            steg_dwt_2_low_l = steg_dwt_2_l.narrow(1, 0, 3)
            cover_dwt_low = cover_dwt.narrow(1, 0, 3)

            out_criterian_l = criterion(input_secret_1_l,input_secret_2_l,secret_rev_1_l,secret_rev_2_l, \
            cover,steg_1_l,steg_2_l, \
            steg_dwt_1_low_l,steg_dwt_2_low_l,cover_dwt_low, \
            args.rec_weight_1,args.rec_weight_2,args.guide_weight_1,args.guide_weight_2, \
            args.freq_weight_1,args.freq_weight_2)
            loss.update(out_criterian_l['hide_loss'])
            
            save_dir = os.path.join('experiments', args.experiment,'images')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cover_img = torch2img(cover)

            input_secret_img_1_b = torch2img(input_secret_1_b)
            steg_img_1_b = torch2img(steg_1_b)
            secret_rev_img_1_b= torch2img(secret_rev_1_b)
            p1, m1 = compute_metrics(secret_rev_img_1_b, input_secret_img_1_b)
            psnrs_1_b.update(p1)
            ssims_1_b.update(m1)
            p2, m2 = compute_metrics(steg_img_1_b, cover_img)
            ssimc_1_b.update(m2)
            psnrc_1_b.update(p2)

            input_secret_img_2_b = torch2img(input_secret_2_b)
            steg_img_2_b = torch2img(steg_2_b)
            secret_rev_img_2_b= torch2img(secret_rev_2_b)
            p1, m1 = compute_metrics(secret_rev_img_2_b, input_secret_img_2_b)
            psnrs_2_b.update(p1)
            ssims_2_b.update(m1)
            p2, m2 = compute_metrics(steg_img_2_b, cover_img)
            psnrc_2_b.update(p2)
            ssimc_2_b.update(m2)
            
            input_secret_img_1_n = torch2img(input_secret_1_n)
            steg_img_1_n = torch2img(steg_1_n)
            secret_rev_img_1_n= torch2img(secret_rev_1_n)
            p1, m1 = compute_metrics(secret_rev_img_1_n, input_secret_img_1_n)
            psnrs_1_n.update(p1)
            ssims_1_n.update(m1)
            p2, m2 = compute_metrics(steg_img_1_n, cover_img)
            ssimc_1_n.update(m2)
            psnrc_1_n.update(p2)

            input_secret_img_2_n = torch2img(input_secret_2_n)
            steg_img_2_n = torch2img(steg_2_n)
            secret_rev_img_2_n= torch2img(secret_rev_2_n)
            p1, m1 = compute_metrics(secret_rev_img_2_n, input_secret_img_2_n)
            psnrs_2_n.update(p1)
            ssims_2_n.update(m1)
            p2, m2 = compute_metrics(steg_img_2_n, cover_img)
            psnrc_2_n.update(p2)
            ssimc_2_n.update(m2)
            
            
            input_secret_img_1_l = torch2img(input_secret_1_l)
            steg_img_1_l = torch2img(steg_1_l)
            secret_rev_img_1_l= torch2img(secret_rev_1_l)
            p1, m1 = compute_metrics(secret_rev_img_1_l, input_secret_img_1_l)
            psnrs_1_l.update(p1)
            ssims_1_l.update(m1)
            p2, m2 = compute_metrics(steg_img_1_l, cover_img)
            ssimc_1_l.update(m2)
            psnrc_1_l.update(p2)

            input_secret_img_2_l = torch2img(input_secret_2_l)
            steg_img_2_l = torch2img(steg_2_l)
            secret_rev_img_2_l= torch2img(secret_rev_2_l)
            p1, m1 = compute_metrics(secret_rev_img_2_l, input_secret_img_2_l)
            psnrs_2_l.update(p1)
            ssims_2_l.update(m1)
            p2, m2 = compute_metrics(steg_img_2_l, cover_img)
            psnrc_2_l.update(p2)
            ssimc_2_l.update(m2)

            
            if args.save_images:
                cover_dir = os.path.join(save_dir,'cover')
                if not os.path.exists(cover_dir):
                    os.makedirs(cover_dir)
                stego_dir_1_b = os.path.join(save_dir,'stego1','blur')
                if not os.path.exists(stego_dir_1_b):
                    os.makedirs(stego_dir_1_b)
                secret_dir_1_b = os.path.join(save_dir,'secret1','blur')
                if not os.path.exists(secret_dir_1_b):
                    os.makedirs(secret_dir_1_b)
                rev_dir_1_b = os.path.join(save_dir,'rev1','blur')
                if not os.path.exists(rev_dir_1_b):
                    os.makedirs(rev_dir_1_b)
                stego_dir_2_b = os.path.join(save_dir,'stego2','blur')
                if not os.path.exists(stego_dir_2_b):
                    os.makedirs(stego_dir_2_b)
                secret_dir_2_b = os.path.join(save_dir,'secret2','blur')
                if not os.path.exists(secret_dir_2_b):
                    os.makedirs(secret_dir_2_b)
                rev_dir_2_b = os.path.join(save_dir,'rev2','blur')
                if not os.path.exists(rev_dir_2_b):
                    os.makedirs(rev_dir_2_b)  
                    
                stego_dir_1_n = os.path.join(save_dir,'stego1','noise')
                if not os.path.exists(stego_dir_1_n):
                    os.makedirs(stego_dir_1_n)
                secret_dir_1_n = os.path.join(save_dir,'secret1','noise')
                if not os.path.exists(secret_dir_1_n):
                    os.makedirs(secret_dir_1_n)
                rev_dir_1_n = os.path.join(save_dir,'rev1','noise')
                if not os.path.exists(rev_dir_1_n):
                    os.makedirs(rev_dir_1_n)
                stego_dir_2_n = os.path.join(save_dir,'stego2','noise')
                if not os.path.exists(stego_dir_2_n):
                    os.makedirs(stego_dir_2_n)
                secret_dir_2_n = os.path.join(save_dir,'secret2','noise')
                if not os.path.exists(secret_dir_2_n):
                    os.makedirs(secret_dir_2_n)
                rev_dir_2_n = os.path.join(save_dir,'rev2','noise')
                if not os.path.exists(rev_dir_2_n):
                    os.makedirs(rev_dir_2_n)    
                    
    
                stego_dir_1_l = os.path.join(save_dir,'stego1','lr')
                if not os.path.exists(stego_dir_1_l):
                    os.makedirs(stego_dir_1_l)
                secret_dir_1_l = os.path.join(save_dir,'secret1','lr')
                if not os.path.exists(secret_dir_1_l):
                    os.makedirs(secret_dir_1_l)
                rev_dir_1_l = os.path.join(save_dir,'rev1','lr')
                if not os.path.exists(rev_dir_1_l):
                    os.makedirs(rev_dir_1_l)
                stego_dir_2_l = os.path.join(save_dir,'stego2','lr')
                if not os.path.exists(stego_dir_2_l):
                    os.makedirs(stego_dir_2_l)
                secret_dir_2_l = os.path.join(save_dir,'secret2','lr')
                if not os.path.exists(secret_dir_2_l):
                    os.makedirs(secret_dir_2_l)
                rev_dir_2_l = os.path.join(save_dir,'rev2','lr')
                if not os.path.exists(rev_dir_2_l):
                    os.makedirs(rev_dir_2_l)      
        
              
                cover_img.save(os.path.join(cover_dir,'%03d.png' % i))
                input_secret_img_1_b.save(os.path.join(secret_dir_1_b,'%03d.png' % i))
                steg_img_1_b.save(os.path.join(stego_dir_1_b,'%03d.png' % i))
                secret_rev_img_1_b.save(os.path.join(rev_dir_1_b,'%03d.png' % i))
                input_secret_img_2_b.save(os.path.join(secret_dir_2_b,'%03d.png' % i))
                steg_img_2_b.save(os.path.join(stego_dir_2_b,'%03d.png' % i))
                secret_rev_img_2_b.save(os.path.join(rev_dir_2_b,'%03d.png' % i))
                
                input_secret_img_1_n.save(os.path.join(secret_dir_1_n,'%03d.png' % i))
                steg_img_1_n.save(os.path.join(stego_dir_1_n,'%03d.png' % i))
                secret_rev_img_1_n.save(os.path.join(rev_dir_1_n,'%03d.png' % i))
                input_secret_img_2_n.save(os.path.join(secret_dir_2_n,'%03d.png' % i))
                steg_img_2_n.save(os.path.join(stego_dir_2_n,'%03d.png' % i))
                secret_rev_img_2_n.save(os.path.join(rev_dir_2_n,'%03d.png' % i))
                
                input_secret_img_1_l.save(os.path.join(secret_dir_1_l,'%03d.png' % i))
                steg_img_1_l.save(os.path.join(stego_dir_1_l,'%03d.png' % i))
                secret_rev_img_1_l.save(os.path.join(rev_dir_1_l,'%03d.png' % i))
    
                input_secret_img_2_l.save(os.path.join(secret_dir_2_l,'%03d.png' % i))
                steg_img_2_l.save(os.path.join(stego_dir_2_l,'%03d.png' % i))
                secret_rev_img_2_l.save(os.path.join(rev_dir_2_l,'%03d.png' % i))
             
            i=i+1


    logger_val.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tPSNRC_1_b: {psnrc_1_b.avg:.6f} |"
        f"\tSSIMC_1_b: {ssimc_1_b.avg:.6f} |"
        f"\tPSNRS_1_b: {psnrs_1_b.avg:.6f} |" 
        f"\tSSIMS_1_b: {ssims_1_b.avg:.6f} |"
        f"\tPSNRC_2_b: {psnrc_2_b.avg:.6f} |"
        f"\tSSIMC_2_b: {ssimc_2_b.avg:.6f} |"
        f"\tPSNRS_2_b: {psnrs_2_b.avg:.6f} |" 
        f"\tSSIMS_2_b: {ssims_2_b.avg:.6f} |"
        f"\tPSNRC_1_n: {psnrc_1_n.avg:.6f} |"
        f"\tSSIMC_1_n: {ssimc_1_n.avg:.6f} |"
        f"\tPSNRS_1_n: {psnrs_1_n.avg:.6f} |" 
        f"\tSSIMS_1_n: {ssims_1_n.avg:.6f} |"
        f"\tPSNRC_2_n: {psnrc_2_n.avg:.6f} |"
        f"\tSSIMC_2_n: {ssimc_2_n.avg:.6f} |"
        f"\tPSNRS_2_n: {psnrs_2_n.avg:.6f} |" 
        f"\tSSIMS_2_n: {ssims_2_n.avg:.6f} |"
        f"\tPSNRC_1_l: {psnrc_1_l.avg:.6f} |"
        f"\tSSIMC_1_l: {ssimc_1_l.avg:.6f} |"
        f"\tPSNRS_1_l: {psnrs_1_l.avg:.6f} |" 
        f"\tSSIMS_1_l: {ssims_1_l.avg:.6f} |"
        f"\tPSNRC_2_l: {psnrc_2_l.avg:.6f} |"
        f"\tSSIMC_2_l: {ssimc_2_l.avg:.6f} |"
        f"\tPSNRS_2_l: {psnrs_2_l.avg:.6f} |" 
        f"\tSSIMS_2_l: {ssims_2_l.avg:.6f} |"
    )

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
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint_3["net"].items():
        #     name = k[7:]  # 移除 `module.`
        #     new_state_dict[name] = v
        # net3.load_state_dict(new_state_dict)
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
