import torch
import torch.nn as nn 
import functools
import monai 
from models.resnet import ResnetBlock 

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        print('++++++++++++++++++++++++self.loss: ', self.loss)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


'''
define the correlation coefficient loss
'''


def Cor_CoeLoss(y_pred, y_target):
    x = y_pred
    y = y_target
    x_var = x - torch.mean(x)
    y_var = y - torch.mean(y)
    r_num = torch.sum(x_var * y_var)
    r_den = torch.sqrt(torch.sum(x_var ** 2)) * \
        torch.sqrt(torch.sum(y_var ** 2))
    r = r_num / r_den

    # return 1 - r  # best are 0
    return 1 - r**2  # abslute constrain


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult,
                                  padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReplicationPad3d(3)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
    

# 定义3D交叉注意力层 3

class CrossAttentionLayer3Dver2(nn.Module):
    def __init__(self, feature_dim, text_embedding_dim):
        super(CrossAttentionLayer3Dver2, self).__init__()
        # 特征维度，例如图像的通道数
        self.feature_dim = feature_dim
        # 文本嵌入的维度
        self.text_embedding_dim = text_embedding_dim

        # 定义查询(query)的3D卷积层，用于从图像特征中提取查询信息
        # 将特征维度降至原来的1/8，减少计算量并增加模型深度
        self.query_conv = nn.Conv3d(
            feature_dim, feature_dim // 8, kernel_size=1)

        # 定义针对文本嵌入的线性变换，用于将文本嵌入投影到键(key)空间
        # 这里输出维度也是特征维度的1/8
        self.key_proj = nn.Linear(text_embedding_dim, feature_dim // 8)

        # 定义值(value)的3D卷积层，用于从图像特征中提取值信息
        # 这里保持原始特征维度不变
        self.value_conv = nn.Conv3d(feature_dim, feature_dim, kernel_size=1)

        # 定义softmax层，用于在计算注意力时进行归一化处理
        # 注意力权重需要在特定维度上进行归一化，这里选择第1个维度
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x, text_embedding):
        # 获取输入的维度信息：batch大小，通道数C，深度D，高度H，宽度W
        batch_size, C, D, H, W = x.size()

        # 将图像特征通过一个卷积层来得到查询矩阵
        proj_query = self.query_conv(x).view(
            batch_size, -1, D * H * W).permute(0, 2, 1)  # (B, D*H*W, C')

        # 将文本嵌入通过一个线性层得到键矩阵
        proj_key = self.key_proj(text_embedding)  # (B, C')

        # 计算查询矩阵和键矩阵的乘积，得到注意力能量分布
        energy = torch.bmm(proj_query, proj_key.unsqueeze(2))  # (B, D*H*W, 1)

        # 应用softmax函数获取注意力权重
        attention = self.softmax(energy)  # (B, D*H*W, 1)

        # 将图像特征通过另一个卷积层来得到值矩阵
        proj_value = self.value_conv(x).view(
            batch_size, -1, D * H * W)  # (B, C, D*H*W)

        # 调整注意力权重的形状以便和值矩阵相乘
        attention = attention.permute(0, 2, 1).expand_as(proj_value)

        # 使用元素乘法（哈达玛积）来融合注意力权重和值矩阵
        out = proj_value * attention  # 这里的乘法是逐元素相乘
        out = out.view(batch_size, C, D, H, W)  # 调整输出形状以匹配原始输入x的形状

        # 将融合了注意力权重的特征和原始输入x相加，(去除了x)
        return out


class CrossAttentionLayer3D(nn.Module):
    def __init__(self, feature_dim, text_embedding_dim):
        super(CrossAttentionLayer3D, self).__init__()
        # 特征维度，例如图像的通道数
        self.feature_dim = feature_dim
        # 文本嵌入的维度
        self.text_embedding_dim = text_embedding_dim

        # 定义查询(query)的3D卷积层，用于从图像特征中提取查询信息
        # 将特征维度降至原来的1/8，减少计算量并增加模型深度
        self.query_conv = nn.Conv3d(
            feature_dim, feature_dim // 8, kernel_size=1)

        # 定义针对文本嵌入的线性变换，用于将文本嵌入投影到键(key)空间
        # 这里输出维度也是特征维度的1/8
        self.key_proj = nn.Linear(text_embedding_dim, feature_dim // 8)

        # 定义值(value)的3D卷积层，用于从图像特征中提取值信息
        # 这里保持原始特征维度不变
        self.value_conv = nn.Conv3d(feature_dim, feature_dim, kernel_size=1)

        # 定义softmax层，用于在计算注意力时进行归一化处理
        # 注意力权重需要在特定维度上进行归一化，这里选择第1个维度
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, text_embedding):
        # 获取输入的维度信息：batch大小，通道数C，深度D，高度H，宽度W
        batch_size, C, D, H, W = x.size()

        # 将图像特征通过一个卷积层来得到查询矩阵
        proj_query = self.query_conv(x).view(
            batch_size, -1, D * H * W).permute(0, 2, 1)  # (B, D*H*W, C')

        # 将文本嵌入通过一个线性层得到键矩阵
        proj_key = self.key_proj(text_embedding)  # (B, C')

        # 计算查询矩阵和键矩阵的乘积，得到注意力能量分布
        energy = torch.bmm(proj_query, proj_key.unsqueeze(2))  # (B, D*H*W, 1)

        # 应用softmax函数获取注意力权重
        attention = self.softmax(energy)  # (B, D*H*W, 1)

        # 将图像特征通过另一个卷积层来得到值矩阵
        proj_value = self.value_conv(x).view(
            batch_size, -1, D * H * W)  # (B, C, D*H*W)

        # 调整注意力权重的形状以便和值矩阵相乘
        attention = attention.permute(0, 2, 1).expand_as(proj_value)

        # 使用元素乘法来融合注意力权重和值矩阵
        out = proj_value * attention  # 这里的乘法是逐元素相乘
        out = out.view(batch_size, C, D, H, W)  # 调整输出形状以匹配原始输入x的形状

        # 将融合了注意力权重的特征和原始输入x相加
        return out + x


class ResNetCrossAttention2(nn.Module):
    def __init__(self, n_blocks, ngf, text_embedding_dim, padding_type, norm_layer, use_dropout, use_bias, n_downsampling):
        super(ResNetCrossAttention2, self).__init__()
        self.layers = nn.ModuleList()
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            self.layers.append(ResnetBlock(ngf * mult, padding_type=padding_type,
                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))
            self.layers.append(CrossAttentionLayer3Dver2(
                ngf * mult, text_embedding_dim))

    def forward(self, x, text_embedding):
        for layer in self.layers:
            if isinstance(layer, CrossAttentionLayer3Dver2):
                x = layer(x, text_embedding)
            else:
                x = layer(x)
        return x



class ResNetCrossAttention(nn.Module):
    def __init__(self, n_blocks, ngf, text_embedding_dim, padding_type, norm_layer, use_dropout, use_bias, n_downsampling):
        super(ResNetCrossAttention, self).__init__()
        self.layers = nn.ModuleList()
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            self.layers.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))
            self.layers.append(CrossAttentionLayer3D(ngf * mult, text_embedding_dim))

    def forward(self, x, text_embedding):
        for layer in self.layers:
            if isinstance(layer, CrossAttentionLayer3D):
                x = layer(x, text_embedding)
            else:
                x = layer(x)
        return x

 

class ResnetGenerator_with_text_embeddings2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator_with_text_embeddings2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

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

        # ResNet Blocks with cross attention 修改点
        self.resnet_cross_attention_blocks = ResNetCrossAttention2(
            n_blocks, ngf, 512, padding_type, norm_layer, use_dropout, use_bias, n_downsampling)

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

    def forward(self, input, prompts=None):
        # Pass input through encoder
        encoded = self.encoder(input)
        # Pass encoded features through ResNet blocks

        # embeddings = self.res_blocks(encoded)
        if prompts is not None:
            embeddings = self.resnet_cross_attention_blocks(encoded, prompts)
        else:
            embeddings = self.res_blocks(encoded)

        # Here you can manipulate embeddings if necessary
        # embeddings = manipulate_embeddings(embeddings)
        # Pass embeddings through decoder to get the output
        return self.decoder(embeddings)


class ResnetGenerator_with_text_embeddings(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator_with_text_embeddings, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

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
                nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        self.encoder = nn.Sequential(*encoder_layers)

        # ResNet Blocks
        resnet_blocks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resnet_blocks.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))

        self.res_blocks = nn.Sequential(*resnet_blocks)
        
        # ResNet Blocks with cross attention 修改点
        self.resnet_cross_attention_blocks =  ResNetCrossAttention(n_blocks, ngf, 512, padding_type, norm_layer, use_dropout, use_bias, n_downsampling)


        # Decoder
        decoder_layers = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder_layers += [
                nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        decoder_layers += [
            nn.ReplicationPad3d(3),
            nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, input, prompts = None):
        # Pass input through encoder
        encoded = self.encoder(input)
        # Pass encoded features through ResNet blocks
        
        
        # embeddings = self.res_blocks(encoded)
        if prompts is not None:
            embeddings = self.resnet_cross_attention_blocks(encoded, prompts)
        else:
            embeddings = self.res_blocks(encoded)
            
        # Here you can manipulate embeddings if necessary
        # embeddings = manipulate_embeddings(embeddings)
        # Pass embeddings through decoder to get the output
        return self.decoder(embeddings)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


def Dynet():

    sizes, spacings = [128, 128, 64], (1.5, 1.5, 1.5)

    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (
            ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    net = monai.networks.nets.DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        res_block=True,
    )

    net.add_module("activation", torch.nn.Tanh())

    return net


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1,
                      stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class PlainCNN(nn.Module):
    """Plain CNN model.

    This model consists of 12 basic building blocks, Conv3D
    + LeakyReLU (negative_slope=0.1), in series. The global
    residual learning is also applied.
    """

    def __init__(self) -> None:
        """Initialises PlainCNN."""
        super().__init__()

        self._block_in = Conv3DBlock(1, 64)
        self._block_middle_iter = nn.ModuleList(
            Conv3DBlock(64, 64) for i in range(10))
        self._block_out = Conv3DBlock(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns output by forward calculation."""
        residual = input

        for block in [self._block_in, *self._block_middle_iter, self._block_out]:
            residual = block(residual)

        return self.tanh(residual)


class Conv3DBlock(nn.Module):
    """Conv3D + LeakyReLU (negative_slope=0.1)"""

    def __init__(self, channel_in: int, channel_out: int) -> None:
        """Initialises Conv3DBlock."""
        super().__init__()

        self._layer = nn.Sequential(
            nn.Conv3d(
                channel_in, channel_out, kernel_size=3, stride=1, padding='same',
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns output by forward calculation."""
        return self._layer(input)


class ImageDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(ImageDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=3,
                      stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, ndf * 2, kernel_size=3, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

        self.avgpool = nn.AdaptiveAvgPool3d([1, 1, 1])
        self.fc = nn.Linear(ndf*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.net(input)
        input = self.avgpool(input).squeeze(-1).squeeze(-1).squeeze(-1)
        input = self.sigmoid(self.fc(input))
        return input
