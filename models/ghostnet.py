import torch
import torch.nn as nn
import math


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)

class GhostModule(nn.Module):
    def __init__(self, in_planes, out_planes, mode, kernel_size=3, ratio=2, dw_size=3, stride=1, relu=True, bias=False):
        super(GhostModule, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mode = mode
        init_channels = math.ceil(out_planes / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_planes, init_channels, kernel_size, stride, (kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(init_channels) if mode == 'normal' else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, (dw_size-1) // 2, groups=init_channels, bias=bias),
            nn.BatchNorm2d(new_channels) if mode == 'normal' else nn.Sequential()
        )
        # self.bn = nn.BatchNorm2d(init_channels+new_channels) if mode == 'normal' else nn.Sequential()
        # self.bn = nn.BatchNorm2d(init_channels+new_channels)
        if mode == 'parallel_adapters':
        #     self.adapter1 = conv1x1(in_planes, init_channels, stride)
            self.adapter2 = conv1x1(init_channels, new_channels, groups=init_channels)
            # self.adapter2 = conv3x3(init_channels, new_channels, groups=init_channels)


    def forward(self, x):
        x1 = self.primary_conv(x)
        # if self.mode == 'parallel_adapters':
        #     x1 += self.adapter1(x)
        x2 = self.cheap_operation(x1)
        # print(x1.shape, x2.shape)
        if self.mode == 'parallel_adapters':
            x2 += self.adapter2(x1)
        # out = self.bn(torch.cat([x1, x2], dim=1))
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_planes, :, :]




