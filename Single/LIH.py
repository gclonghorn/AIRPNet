import torch
from torch import nn
import math
from util import initialize_weights
import pdb

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


class WLBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(WLBlock, self).__init__()
        self.net = ResidualDenseBlock_out(in_ch,out_ch)

    def forward(self, x):
        out = self.net(x)
        return out

class LiftingStep(nn.Module):
    def __init__(self, pin_ch,  uin_ch):    
        super(LiftingStep, self).__init__()
    
        self.predictor = WLBlock(pin_ch, uin_ch)
        self.updator = WLBlock(uin_ch, pin_ch)

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
    
class LIH(nn.Module):
    def __init__(self, pin_ch, uin_ch, num_step):  
        super(LIH, self).__init__()
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
    def __init__(self, pin_ch, uin_ch, num_step=1):
        super(Model, self).__init__()
 
        self.model = LIH(pin_ch, uin_ch, num_step)

    def forward(self, xc, xd, rev=False):

        if not rev:
            outc, outd = self.model.forward(xc, xd)

        else:
            outc, outd = self.model.inverse(xc, xd)

        return outc, outd