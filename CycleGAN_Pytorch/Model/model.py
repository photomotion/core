import torch
import torch.nn as nn

from Model.CustomLayers import *

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm', nblock=6):
        super(CycleGAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nker = nker
        self.norm = norm
        self.nblock = nblock

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = Conv2d(in_channels=in_channels, out_channels=self.nker, kernel_size=7,
                           stride=1, padding=3, norm=self.norm, activation_fn='relu')
        self.enc2 = Conv2d(in_channels=self.nker, out_channels=2 * self.nker, kernel_size=3,
                           stride=2, padding=1, norm=self.norm, activation_fn='relu')
        self.enc3 = Conv2d(in_channels=2 * self.nker, out_channels=4 * self.nker, kernel_size=3,
                           stride=2, padding=1, norm=self.norm, activation_fn='relu')

        if self.nblock:
            residual_block = []
            for i in range(self.nblock):
                residual_block += [ResidualBlock(in_channels=4 * self.nker, out_channels=4 * nker, kernel_size=3,
                                                 stride=1, padding=1, norm=self.norm)]
            self.residual_block = nn.Sequential(*residual_block)

        self.dec3 = DeConv2d(in_channels=4 * self.nker, out_channels=2 * self.nker, kernel_size=3,
                             stride=2, padding=1, norm=self.norm, activation_fn='relu')
        self.dec2 = DeConv2d(in_channels=2 * self.nker, out_channels=elf.nker, kernel_size=3,
                             stride=2, padding=1, norm=self.norm, activation_fn='relu')
        self.dec1 = DeConv2d(in_channels=self.nker, out_channels=self.out_channels, kernel_size=7,
                             stride=1, padding=3, norm=None, activation_fn=None)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.residual_block(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(out=x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=nker, kernel_size=4, stride=2,
                   padding=1, norm=None, activation_fn='leakyrelu', bias=False),
            Conv2d(in_channels=nker, out_channels=2 * nker, kernel_size=4, stride=2,
                   padding=1, norm=norm, activation_fn='leakyrelu', bias=False),
            Conv2d(in_channels=2 * nker, out_channels=4 * nker, kernel_size=4, stride=2,
                   padding=1, norm=norm, activation_fn='leakyrelu', bias=False),
            Conv2d(in_channels=4 * nker, out_channels=8 * nker, kernel_size=4, stride=2,
                   padding=1, norm=norm, activation_fn='leakyrelu', bias=False),
            Conv2d(in_channels=8 * nker, out_channels=out_channels, kernel_size=4, stride=2,
                   padding=1, norm=None, activation_fn=None, bias=False)
        )
    def forward(self, img):
        return self.model(img)
