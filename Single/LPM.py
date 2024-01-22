import torch 
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
from util import initialize_weights

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionBlock(nn.Module):
    def __init__(self, input=3, output=3, bias=True):
        super(AttentionBlock, self).__init__()

        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.senet = SELayer(channel=input + 2 * 32)
        initialize_weights([self.conv3], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x = torch.cat((x, x1, x2), 1)
        x = self.senet(x)
        x3 = self.conv3(x)
        return x3

class LPM(nn.Module):
    def __init__(self, in_channel=3,att_channel=3, width=16, bias=True):
        super(LPM, self).__init__()
        self.attn_block = AttentionBlock()
        self.conv1 = nn.Conv2d(in_channel, width, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(width+att_channel,width , 3, 1, 1, bias=bias)
        self.prelu1 = nn.PReLU()
        self.conv3 = nn.Conv2d(width+att_channel,width , 3, 1, 1, bias=bias)
        self.prelu2 = nn.PReLU()
        self.conv4 = nn.Conv2d(width,width, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(width,in_channel, 1, 1, 0, bias=bias)
    def forward(self,x):
        imp_map = self.attn_block(x)
        x1 = self.conv1(x)
        x2 = self.prelu1(self.conv2(torch.cat((x1,imp_map),1)))
        x2 = x2 + x1
        x3 = self.prelu2(self.conv3(torch.cat((x2,imp_map),1)))
        x3 = x3 + x1
        x4 = self.conv4(x3)
        x4 = x4 +x1
        x5 = self.conv5(x4)
        return x5 


        
