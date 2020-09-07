import torch
import torch.nn as nn

import os
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', activation_fn='relu'):
        super().__init__()

        layers = []

        layers += [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=0, bias=bias)
        ]

        if not norm is None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not activation_fn is None:
            if activation_fn == 'relu':
                layers += [nn.ReLU()]
            elif activation_fn == 'leakyrelu':
                layers += [nn.LeakyReLU(0.2)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True, norm='bnorm', activation_fn='relu'):
        super().__init__()

        layers = []

        layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, output_padding=output_padding, bias=bias)]

        if not norm is None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=output_padding), nn.ReLU()]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=output_padding), nn.ReLU()]

        if not activation_fn is None:
            if activation_fn == 'relu':
                layers += [nn.ReLU()]
            elif activation_fn == 'leakyrelu':
                layers += [nn.LeakyReLU(0.2)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm'):
        super(ResidualBlock, self).__init__()

        layers = []

        layers += [Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=bias, norm=norm, activation_fn='relu')]

        layers += [Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=bias, norm=norm, activation_fn=None)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)