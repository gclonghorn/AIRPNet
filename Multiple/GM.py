import torch
import torch.nn as nn
from util import initialize_weights


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z



class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

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


class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()

        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.senet = SELayer(channel=input + 4 * 32)
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x = torch.cat((x, x1, x2, x3, x4), 1)
        x = self.senet(x)
        x5 = self.conv5(x)
        return x5


class PrepareBlock(nn.Module):
    def __init__(self, input, bias=True):
        super(PrepareBlock, self).__init__()

        self.conv1 = nn.Conv2d(input, 64, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 64, 64, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 64, 64, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.senet = SELayer(channel=input + 2 * 64)
        initialize_weights([self.conv3], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x = torch.cat((x, x1, x2), 1)
        x = self.senet(x)
        x3 = self.conv3(x)
        return x3


class GM(nn.Module):
    def __init__(self):
        super(GM, self).__init__()
        self.rrdb = ResidualDenseBlock_out(input=64 + 64 + 64, output=3)
        self.s_cover = PrepareBlock(input=3)
        self.s_secret = PrepareBlock(input=3)
        self.s_steg = PrepareBlock(input=3)
        self.nonlocal_block = NONLocalBlock2D(3, sub_sample=True, bn_layer=True)

    def forward(self, cover, secret, steg):

        x_cover = self.s_cover(cover)
        x_steg = self.s_steg(steg)
        x_secret = self.s_secret(secret)
        x = torch.cat((x_cover, x_steg), 1)
        x = torch.cat((x, x_secret), 1)
        x = self.rrdb(x)

        return x


