import torch
from torch import nn
import math
from util import initialize_weights

      
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

    def inverse(self, *inputs,rev):
        for module in reversed(self._modules.values()):
            if type(inputs) == tuple:
                inputs = module(*inputs,rev)
            else:
                inputs = module(inputs,rev)
        return inputs


class ResidualDenseBlock(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5




class WLBlock(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock,  guiding_map=True):
        super().__init__()
        self.split_len1 = 12
        self.split_len2 = 12   
        if guiding_map:
            self.imp = 12
        else:
            self.imp = 0
        self.predictor = subnet_constructor(self.split_len1+self.imp , self.split_len2)
        self.updator = subnet_constructor(self.split_len1 , self.split_len2+self.imp)


    def forward(self, x1, x2, rev=False):
        if not rev:
            Fx1 = self.predictor(x1)
            x2 = - Fx1 + x2
            Fx2 = self.updator(x2)
            x1 = x1 + Fx2
        else:  

            Fxd = self.updator(x2)
            x1 = x1 - Fxd
            Fxc = self.predictor(x1)
            x2 = x2 + Fxc
        return x1, x2

class LIH_stage1(nn.Module):
    def __init__(self,num_step,guiding_map=False):
        super(LIH_stage1, self).__init__()
        self.net = nn.Sequential(*[WLBlock(guiding_map=guiding_map) for _ in range(num_step)])

    def forward(self, x1, x2 ,rev=False):
        if not rev:
            for layer in self.net:
                x1, x2 = layer(x1, x2)
        else:
            for layer in reversed(self.net):
                x1, x2 = layer(x1, x2, rev=True)
        return x1, x2
    
class LIH_stage2(nn.Module):
    def __init__(self,num_step,guiding_map=True):
        super(LIH_stage2, self).__init__()
        self.net = nn.Sequential(*[WLBlock(guiding_map=guiding_map) for _ in range(num_step)])

    def forward(self, x1, x2 ,rev=False):
        if not rev:
            for layer in self.net:
                x1, x2 = layer(x1, x2)
        else:
            for layer in reversed(self.net):
                x1, x2 = layer(x1, x2, rev=True)
        return x1, x2
    
    