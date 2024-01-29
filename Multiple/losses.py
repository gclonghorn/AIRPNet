import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


class LIH_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)
        self.vgg_loss =  VGGLoss(3,1,False)
    def forward(self,input_secret_1,input_secret_2,secret_rev_1,secret_rev_2, \
        cover,steg_1,steg_2, \
        steg_dwt_1_low,steg_dwt_2_low,cover_dwt_low, \
        rec_weight_1,rec_weight_2,guide_weight_1,guide_weight_2, \
        freq_weight_1,freq_weight_2):

        N, _, H, W = input_secret_1.size()
        out = {}
        guide_loss_1 = self.mse(steg_1,cover)
        guide_loss_2 = self.mse(steg_2,cover)
        reconstruction_loss_1 = self.mse(secret_rev_1,input_secret_1)
        reconstruction_loss_2 = self.mse(secret_rev_2,input_secret_2)
        freq_loss_1 = self.mse(steg_dwt_1_low,cover_dwt_low)
        freq_loss_2 = self.mse(steg_dwt_2_low,cover_dwt_low)
        hide_loss = rec_weight_1*reconstruction_loss_1+rec_weight_2*reconstruction_loss_2 \
         + freq_weight_1*freq_loss_1 + freq_weight_2*freq_loss_2 \
         +guide_weight_1*guide_loss_1+guide_weight_2*guide_loss_2
        out['hide_loss'] = hide_loss
        return out


class LSR_Loss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)
        self.vgg_loss =  VGGLoss(3,1,True)
    def forward(self,secret_img1,secret_img2,cover_img,steg_clean2,rec_img1,rec_img2,sweight1,sweight2,cweight2,
        pweight_c):
        N, _, H, W = secret_img1.size()
        out = {}
       
        lossc2 = self.mse(cover_img,steg_clean2)
        losss1 = self.mse(secret_img1,rec_img1)
        losss2 = self.mse(secret_img2,rec_img2)
        percep_losss1 = self.vgg_loss(secret_img1,rec_img1)
        percep_losss2 = self.vgg_loss(secret_img2,rec_img2)
        loss = cweight2*lossc2+ sweight1*losss1 + sweight2*losss2 + pweight_c*percep_losss1 +pweight_c*percep_losss2 
        out['pixel_loss'] = cweight2*lossc2+ sweight1*losss1+sweight2*losss2
        out['percep_loss'] = pweight_c*percep_losss1 +pweight_c*percep_losss2
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


def imp_loss(output, resi):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, resi)
    return loss
 