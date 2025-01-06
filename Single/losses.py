import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


class LIH_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)
    def forward(self, secret, cover , stego ,rec, steg_low, cover_low,rec_weight,guide_weight,freq_weight):
        N, _, H, W = secret.size()
        out = {}
        guide_loss = self.mse(stego,cover)
        reconstruction_loss = self.mse(rec,secret)
        freq_loss = self.mse(steg_low,cover_low)
        hide_loss = rec_weight*reconstruction_loss  + freq_weight*freq_loss  +guide_weight*guide_loss
        out['g_loss'] = guide_loss
        out['r_loss'] = reconstruction_loss
        out['f_loss'] = freq_loss
        out['hide_loss'] = hide_loss
        return out


class LSR_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)
        self.vgg_loss =  VGGLoss(3,1,True)
    def forward(self,secret_img,cover_img,steg_clean,steg_ori,rec_img,sweight,cweight,pweight_c,finetune):
        N, _, H, W = secret_img.size()
        out = {}
        lossc = self.mse(cover_img,steg_clean)
        lossc_ori = self.mse(cover_img,steg_ori)
        losss = self.mse(secret_img,rec_img)
        percep_lossc = self.vgg_loss(cover_img,steg_clean)
        if finetune:
            loss =lossc*cweight + sweight*losss+ pweight_c*percep_lossc+lossc_ori*cweight
            out['pixel_loss'] = cweight*lossc + sweight*losss+lossc_ori*cweight
            out['percep_loss'] = pweight_c*percep_lossc
            out['loss'] = loss
        else:
            loss =lossc*cweight + sweight*losss+ pweight_c*percep_lossc
            out['pixel_loss'] = cweight*lossc + sweight*losss
            out['percep_loss'] = pweight_c*percep_lossc
            out['loss'] = loss
        return out

class VGGLoss(nn.Module):
    """
    Part of pre-trained VGG16. This is used in case we want perceptual loss instead of Mean Square Error loss.
    See for instance https://arxiv.org/abs/1603.08155
    """
    def __init__(self, block_no: int, layer_within_block: int, use_batch_norm_vgg: bool):
        super(VGGLoss, self).__init__()
        if use_batch_norm_vgg:
            vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        else:
            vgg16 = torchvision.models.vgg16(pretrained=True)
        curr_block = 1
        curr_layer = 1
        layers = []
        for layer in vgg16.features.children():
            layers.append(layer.to('cuda:0'))
            if curr_block == block_no and curr_layer == layer_within_block:
                break
            if isinstance(layer, nn.MaxPool2d):
                curr_block += 1
                curr_layer = 1
            else:
                curr_layer += 1

        self.vgg_loss = nn.Sequential(*layers)
        self.criterion = torch.nn.MSELoss(reduce=True, size_average=False).to('cuda:0')

    def forward(self, source,target):
        return self.criterion(self.vgg_loss(source),self.vgg_loss(target))
