import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import Operations_1 as Operations, OPS


## Operation layer
class OperationLayer(nn.Module):
    def __init__(self, C):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        for o in Operations:
            op = OPS[o](C)
            self._ops.append(op)
        self._out = nn.Sequential(nn.Conv2d(C*len(Operations), C, 1, padding=0, bias=False), nn.ReLU())

    def forward(self, x, weights):
        weights = weights.transpose(1,0)
        states = []
        for w, op in zip(weights, self._ops):
            states.append(op(x)*w.view([-1, 1, 1, 1]))
        return self._out(torch.cat(states[:], dim=1))

## a Group of operation layers
class GroupOLs(nn.Module):
    def __init__(self, steps, C):
        super(GroupOLs, self).__init__()  #step=2
        self._steps = steps
        self._ops = nn.ModuleList()
        for _ in range(self._steps):
            op = OperationLayer(C)
            self._ops.append(op)

    def forward(self, s0, weights):
        for i in range(self._steps):
            s0 = self._ops[i](s0, weights[:, i, :])
        return s0


## Operation-wise Attention Layer (OWAL)
class OperationAttn(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OperationAttn, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.ca_fc = nn.Sequential(
                    nn.Linear(channel, self.output*2),
                    nn.ReLU(),
                    nn.Linear(self.output*2, self.k*self.num_ops))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y


class TaskAdaptor(nn.Module):
    def __init__(self, in_size ,out_size,semodule=None):
        super(TaskAdaptor, self).__init__()
        self.se = semodule
        self.conv1 =  nn.Conv2d(in_size, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.nonlinear1 =  nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.nonlinear2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.nonlinear3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(64, out_size, kernel_size=1, stride=1, padding=0, bias=False)

        self.shortcut = nn.Sequential()
        if in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
            )

    def forward(self, x):
        out = self.nonlinear1(self.conv1(x))
        out = self.nonlinear2(self.conv2(out))
        out = self.nonlinear3(self.conv3(out))
        out = self.conv4(out)
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) 
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=1):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return x * self.se(x)


class hswish(nn.Module): 
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_size,out_size,semodule):
        super(Block, self).__init__()
        self.se = semodule
        self.conv1 =  nn.Conv2d(in_size, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.nolinear1 =  nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.nonlinear2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.nolinear3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(64, out_size, kernel_size=1, stride=1, padding=0, bias=False)

        self.shortcut = nn.Sequential()
        if in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
            )

    def forward(self, x):
        out = self.nonlinear1(self.conv1(x))
        out = self.nonlinear2(self.conv2(x))
        out = self.nolinear3(self.conv3(out))
        out = self.conv4(out)
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) 
        return out

        
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma




class Encoder(nn.Module):
    def __init__(self, img_channel=9, width=32, enc_blk_nums=[],split1=1,split2=2):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
       

    def forward(self,inp):
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        return x,encs


class Decoder(nn.Module):

    def __init__(self, img_channel=9, width=32, middle_blk_num=1,dec_blk_nums=[]):
        super().__init__()

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        chan = width * 2**len(dec_blk_nums)
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )


    def forward(self, inp, x, encs,H ,W):
        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

class Middle(nn.Module):
    def __init__(self,chan=32,middle_blk_num=2):
        super().__init__()
        self.middle_blks = \
        nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

    def forward(self,x):
        x = self.middle_blks(x)
        return x

class SRM(nn.Module):

    def __init__(self, img_channel=9, width=16, middle_blk_num=2, enc_blk_nums=[2,2,4], dec_blk_nums=[2,2,2]):
        super().__init__()
        self.encoder = Encoder(img_channel,width,enc_blk_nums)
        self.decoder = Decoder(img_channel,width,middle_blk_num,dec_blk_nums)
        self.num_task=3
        self.steps=2
        self.task_adaptor = TaskAdaptor(in_size=width * 2**len(dec_blk_nums),out_size= self.num_task,semodule= SeModule(self.num_task))
        self.opertaion_attn = OperationAttn(self.num_task, self.steps,len(Operations))
        self.operation_layer = GroupOLs(steps=self.steps,C=width * 2**len(dec_blk_nums))
        self.padder_size = 2 ** len(enc_blk_nums)
    def forward(self,inp):
        B, C, H, W = inp.shape  # 3,9,512,512
        inp = self.check_image_size(inp)
        inp_enc,encs = self.encoder(inp)  # 3, 256, 64, 64
        task_logit = self.task_adaptor(inp_enc) #[3,3] [b,num_task] 
        weights = self.opertaion_attn(task_logit) #[3,3,8] [b,num_steps,num_operations]
        inp_mid = self.operation_layer(inp_enc, weights) #3,256,4,4
        inp_dec = self.decoder(inp,inp_mid,encs,H,W)  #3,9,512,512
        return inp_dec 
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x




if __name__ == "__main__" :
    inp = torch.randn(3,9,1024,1024)
    model = SRM()
    ret = model(inp)




    

