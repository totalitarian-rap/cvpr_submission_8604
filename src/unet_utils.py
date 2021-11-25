import torch
import torch.nn.functional as F
from torch import nn
from . import models_utils

class LastConv(nn.Module):
    def __init__(self, inp_chans, out_chans, kernel, padding):
        super().__init__()
        self.conv = nn.Conv2d(inp_chans, out_chans, kernel_size=kernel, padding=padding)

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, num_channels, hidden_dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels+hidden_dim, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class ConvBlockInOutChans(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Double3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, padding=0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel, padding=padding), 
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class InceptionStyleUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.path1 = nn.Sequential(
            ConvBlockInOutChans(in_channels, in_channels, kernel=1, padding=0),
            ConvBlockInOutChans(in_channels, out_channels, kernel=3, padding=1)
        )
        self.path2 = nn.Sequential(
            ConvBlockInOutChans(in_channels, in_channels, kernel=1, padding=0),
            ConvBlockInOutChans(in_channels, out_channels, kernel=5, padding=2)
        )
        self.path3 = ConvBlockInOutChans(in_channels, out_channels, kernel=1, padding=0)
        self.conv = DoubleConv(out_channels*4, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1) 
        return self.conv(
            torch.cat([
                self.path1(x1), self.path2(x1), self.path3(x1), x2], dim=1)
            )

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels+out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(torch.cat([x2, x1], dim=1))

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channels = args.model_parameters['input_channels']
        stem_filters = args.model_parameters['stem_filters']
        model = args.model_parameters['model'].lower()
        squeeze = args.model_parameters['squeeze_bottleneck']
        if model == 'prob_unet' or squeeze:
            self.squeeze_bottleneck = True 
        else:
            self.squeeze_bottleneck = False
        self.stem = DoubleConv(in_channels, stem_filters)
        self.encoder = models_utils.create_encoder(args)
        if self.squeeze_bottleneck:
            self._add_partial_decoder(args)
        self.partial_output = list()

    def forward(self, x):
        self.partial_output = list()
        loop_len = len(self.encoder) - 2 if self.squeeze_bottleneck else len(self.encoder) - 1
        x = self.stem(x)
        self.partial_output.append(x)
        for i, layer in enumerate(self.encoder):   
            x = layer(x)
            if i < loop_len:
                self.partial_output.append(x)
        if self.squeeze_bottleneck:
            x = self.linear(x)
            x = x.unsqueeze(-1).unsqueeze(-1)
            return self.partial_decoder(x)
        return x

    def _add_partial_decoder(self, args):
        num_classes = args.model_parameters['num_classes']
        output_layers = nn.ModuleList(self.encoder)[-1][1].in_features
        model_name = args.model_parameters['unet_encoder']
        layers_list = models_utils.get_layer_list(model_name)
        self.linear =  nn.Linear(num_classes, output_layers)
        self.partial_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlockInOutChans(output_layers, output_layers//2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlockInOutChans(output_layers//2, output_layers//4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlockInOutChans(output_layers//4, output_layers//4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlockInOutChans(output_layers//4, layers_list[-1]),
        )
    
class Decoder(nn.Module):
    def __init__(self, args):
        model_type = args.model_parameters['model']
        if model_type in ['unet', 'twoheaded_unet', 'vec_reg']:
            output_channels = args.model_parameters['output_channels']
        elif model_type == 'prob_unet':
            output_channels = args.model_parameters['decoder_output_channels']
        stem_filters = args.model_parameters['stem_filters']
        inception_style = args.model_parameters['inception_style_decoder']
        dilated_residues = args.model_parameters['dilated_residues']
        model_name = args.model_parameters['unet_encoder']
        layer_list = models_utils.get_layer_list(model_name)
        super().__init__() 
        if inception_style:
            up_module = InceptionStyleUp
        else:
            up_module = Up
        self.upconvs = [
            up_module(layer_list[i+1], layer_list[i])
            for i in range(len(layer_list)-1) 
        ]
        self.upconvs = self.upconvs[::-1]
        self.upconvs.append(up_module(layer_list[0], stem_filters))
        self.upconvs = nn.Sequential(*self.upconvs)
        self.out_conv = nn.Sequential(
            ConvBlockInOutChans(stem_filters, stem_filters, kernel=1, padding=0),
            LastConv(stem_filters, output_channels, kernel=1, padding=0)
        )

    def forward(self, x, prev_x):
        for i, up_layer in enumerate(self.upconvs):
            x = up_layer(x , prev_x[-i-1])
        return self.out_conv(x)


class Classifier(nn.Module):
    def __init__(self, pooling=0):
        """
        0 -- average and max poolings
        1 -- average pooling
        2 -- max pooling
        """
        assert pooling in list(range(3))
        super().__init__()
        self.pooling = pooling
        if pooling == 0:
            self.average_pool =  nn.AvgPool2d(16)
            self.max_pool = nn.MaxPool2d(16)
            self.mult = 2
        elif pooling == 1:
            self.pool = nn.AvgPool2d(16)
            self.mult = 1
        else:
            self.pool = nn.MaxPool2d(16)
            self.mult = 1
        self.linear = nn.Sequential(
            nn.Linear(370*self.mult, 512),
            nn.SiLU(),
            nn.Linear(512,1)
        )
    
    def forward(self, x):
        if self.pooling == 0:
            x = torch.hstack([self.average_pool(x), self.max_pool(x)]).squeeze()
        else:
            x = self.pool(x).squeeze()
        return self.linear(x)
    
