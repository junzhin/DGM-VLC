
import os
import sys
import functools
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from models.resnet import ResnetBlock


# Source: https://github.com/tobran/DF-GAN/blob/64fc9c9eb1d99cfde0f8215fc4388588734b047c/code/models/GAN.py

class NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
        super(NetG, self).__init__()
        self.ngf = ngf
        # input noise (batch_size, 100)
        self.fc = nn.Linear(nz, ngf*8*4*4)
        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        in_out_pairs = get_G_in_out_chs(ngf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.GBlocks.append(
                G_Block(cond_dim+nz, in_ch, out_ch, upsample=True))
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, c):  # x=noise, c=ent_emb
        # concat noise and sentence
        out = self.fc(noise)
        out = out.view(noise.size(0), 8*self.ngf, 4, 4)
        cond = torch.cat((noise, c), dim=1)
        # fuse text and visual features
        for GBlock in self.GBlocks:
            out = GBlock(out, cond)
        # convert to RGB image
        out = self.to_rgb(out)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf, imsize=128, ch_size=3):
        super(NetD, self).__init__()
        self.conv_img = nn.Conv2d(ch_size, ndf, 3, 1, 1)
        # build DBlocks
        self.DBlocks = nn.ModuleList([])
        in_out_pairs = get_D_in_out_chs(ndf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.DBlocks.append(D_Block(in_ch, out_ch))

    def forward(self, x):
        out = self.conv_img(x)
        for DBlock in self.DBlocks:
            out = DBlock(out)
        return out


class NetC(nn.Module):
    def __init__(self, ndf, cond_dim=256):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf*8+cond_dim, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):
        y = y.view(-1, self.cond_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out


class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, x, y):
        if self.upsample == True:
            x = F.interpolate(x, scale_factor=2)
        return self.shortcut(x) + self.residual(x, y)


class D_Block(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        # return x + res
        return x + self.gamma*res


class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return h


class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs



#------------------------------------------------------------------------------

class G_Block3D(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample, nz = 0):
        super(G_Block3D, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv3d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK3D(cond_dim, in_ch, nz = nz)
        self.fuse2 = DFBLK3D(cond_dim, out_ch, nz = nz)
        if self.learnable_sc:
            self.c_sc = nn.Conv3d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, x, y):
        if self.upsample:
            x = F.interpolate(x, scale_factor=(2, 2, 2),
                              mode='trilinear', align_corners=True)
        return self.shortcut(x) + self.residual(x, y)
   
class DFBLK3D(nn.Module):
    def __init__(self, cond_dim, in_ch, nz = 0):
        super(DFBLK3D, self).__init__()
        self.affine0 = Affine3D(cond_dim, in_ch, nz = nz)
        self.affine1 = Affine3D(cond_dim, in_ch, nz = nz)

    def forward(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return h


class Affine3D(nn.Module):
    def __init__(self, cond_dim, num_features, nz = 0):
        super(Affine3D, self).__init__()
        self.cond_dim = cond_dim
        self.num_features = num_features
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self.nz = nz
        self.clip_text_em_dim = 512

        self.linear = nn.Linear(self.clip_text_em_dim + self.nz, cond_dim)


        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None): 
        y_reduce = self.linear(y) 
        weight = self.fc_gamma(y_reduce)
        bias = self.fc_beta(y_reduce)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        # Expand weight and bias to match the dimensions of the 3D volume
        size = x.size()
        # print('= x.size(): ',  x.size())
        weight = weight.view(size[0], -1, 1, 1, 1).expand(size)
        bias = bias.view(size[0], -1, 1, 1, 1).expand(size)
        return weight * x + bias


def get_G_in_out_chs(nf, imsize):
    # Assuming imsize is a tuple of (D, H, W)
    layer_num = int(np.log2(max(imsize)))- 1
    channel_nums = [nf * min(2 ** idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = list(zip(channel_nums[:-1], channel_nums[1:]))
    print('in_out_pairs: ', in_out_pairs)
    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    # Assuming imsize is a tuple of (D, H, W)
    layer_num = int(np.log2(max(imsize))) - 1
    channel_nums = [nf * min(2 ** idx, 8) for idx in range(layer_num)]
    in_out_pairs = list(zip(channel_nums[:-1], channel_nums[1:]))
    print('in_out_pairs: ', in_out_pairs)
    return in_out_pairs


#------------------------------------------------------------------------------


class ResnetGenerator_with_text_embeddings3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect', cond_dim = 256, imsize=(200, 200, 64)):
        assert (n_blocks >= 0)
        super(ResnetGenerator_with_text_embeddings3, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nz = 100

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        # Encoder
        encoder_layers = [
            nn.ReplicationPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder_layers += [
                nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                          stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        self.encoder = nn.Sequential(*encoder_layers)

        # ResNet Blocks
        resnet_blocks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resnet_blocks.append(ResnetBlock(ngf * mult, padding_type=padding_type,
                                 norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))

        self.res_blocks = nn.Sequential(*resnet_blocks)

        # ResNet Blocks with GBlocks connect points - 修改点-------------------------
        self.GBlocks = nn.ModuleList([])
        # You need to define this function
        in_out_pairs = [(256, 256), (256, 256),
                        (256, 256),(256, 256),(256, 256), (256, 256)]
        print('-------------------------->in_out_pairs: ', in_out_pairs)                        
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.GBlocks.append(
                G_Block3D(cond_dim + self.nz, in_ch, out_ch, upsample=False,nz = self.nz))  

            #没有加噪声是时候使用
            # self.nz = 0
            # self.GBlocks.append(G_Block3D(cond_dim, in_ch, out_ch, upsample=False,nz = 0) )  

                


        # Decoder
        decoder_layers = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder_layers += [
                nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), kernel_size=3,
                                   stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        decoder_layers += [
            nn.ReplicationPad3d(3),
            nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, input, prompts=None, opt= None):
        # Pass input through encoder
        encoded = self.encoder(input)
        # Pass encoded features through ResNet blocks


        if prompts is not None:
            if self.nz != 0:
                noise_vector = torch.randn(prompts.shape[0], self.nz).to(opt.device)
                # print('noise_vector: ', noise_vector.device)
                # print('prompts: ', prompts.device) 
                prompts = torch.cat([noise_vector,prompts], dim = 1).to(opt.device)
            for GBlock in self.GBlocks:
                encoded = GBlock(encoded, prompts)
            embeddings = encoded
        else:
            embeddings = self.res_blocks(encoded)

 
        return self.decoder(embeddings)


if __name__ == "__main__":
    import torch
    # Instantiate the model
    net = ResnetGenerator_with_text_embeddings3(input_nc=1, output_nc=1, ngf=64, norm_layer=get_norm_layer(norm_type="instance"),
                                                use_dropout=True, n_blocks=9, padding_type='reflect', cond_dim=512, imsize=(200, 200, 64))

    # Create dummy input data
    input_tensor = torch.rand(1, 1, 200, 200, 64)
    prompts_tensor = torch.rand(1, 512)

    # Perform a forward pass
    output = net(input_tensor, prompts_tensor)

    print(f"Output shape: {output.shape}")
