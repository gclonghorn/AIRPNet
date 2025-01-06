import torch
from torch import nn
import math
from util import waveletDecomp,initialize_weights
from SRM import SRM
from LPM import LPM



class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

    def inverse(self, *inputs):
        for module in reversed(self._modules.values()):
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5



class LiftingStep(nn.Module):
    def __init__(self, pin_ch, uin_ch, ):    
        super(LiftingStep, self).__init__()
        self.predictor = ResidualDenseBlock_out(pin_ch, uin_ch)
        self.updator = ResidualDenseBlock_out(uin_ch, pin_ch)

    def forward(self, xc, xd):
        Fxc = self.predictor(xc)
        xd = - Fxc + xd
        Fxd = self.updator(xd)
        xc = xc + Fxd

        return xc, xd

    def inverse(self, xc, xd):
        Fxd = self.updator(xd)
        xc = xc - Fxd
        Fxc = self.predictor(xc)
        xd = xd + Fxc

        return xc, xd


class LINN(nn.Module):
    def __init__(self, pin_ch, uin_ch, num_step):  
        super(LINN, self).__init__()
       
        self.layers = []
        for _ in range(num_step):
            self.layers.append(LiftingStep(pin_ch, uin_ch))
        self.net = mySequential(*self.layers)


    def forward(self, xc, xd):
        for i in range(len(self.net)):
            outc, outd = self.net[i].forward(xc, xd)
        return outc, outd

    def inverse(self, xc, xd):
        for i in reversed(range(len(self.net))):
            outc, outd = self.net[i].inverse(xc, xd)
        return outc, outd


class Model(nn.Module):
    def __init__(self, steps=1, klvl=3, mid=2,enc=[2,2,4],dec=[2,2,2],lfrestore=True,width=32):
        super(Model, self).__init__()
        pin_chs = 3
        uint_chs = 9
        nstep = steps
        self.innlayers = []
        for ii in range(klvl):
            dilate = 2 ** ii
            if ii > 1:
                dilate = 2
            self.innlayers.append(LINN(pin_ch=pin_chs, uin_ch=uint_chs,num_step=nstep))
        self.innnet = mySequential(*self.innlayers)
        #denoise net
        self.ddnlayers = []
        for ii in range(klvl):
            self.ddnlayers.append(SRM(img_channel=9,width=width,middle_blk_num=mid,enc_blk_nums=enc,dec_blk_nums=dec))

        self.ddnnet = mySequential(*self.ddnlayers)
        self.split = waveletDecomp()
        self.LFRestoreBlock = LPM()
        self.lfrestore = lfrestore

    def forward(self,x):
        xc, xd, xc_, xd_  = [], [], [], []

        for i in range(len(self.innnet)):
           
            if i == 0:
                xcc , xdd = self.split.forward(x)
                tmpxc, tmpxd = self.innnet[i].forward(xcc,xdd)
            else:
                xcc, xdd = self.split.forward(xc[i - 1])
                tmpxc, tmpxd = self.innnet[i].forward(xcc,xdd)

            xc.append(tmpxc)
            xd.append(tmpxd)
            tmpxd_ = self.ddnnet[i].forward(xd[i])

            xd_.append(tmpxd_)
            if self.lfrestore:
                if i == len(self.innnet)-1:

                    tmpxc_ = self.LFRestoreBlock(tmpxc)
                    xc_.append(tmpxc_)
                else:
                    xc_.append(tmpxc)
            else:
                xc_.append(tmpxc)
        for i in reversed(range(len(self.innnet))):
            if i > 0:
                tmpxc,tmpxd = self.innnet[i].inverse(xc_[i], xd_[i])
                xc_[i - 1] = self.split.inverse(tmpxc,tmpxd)
            else:
                tmpxc,tmpxd = self.innnet[i].inverse(xc_[i], xd_[i])
                out = self.split.inverse(tmpxc,tmpxd)

        return out


