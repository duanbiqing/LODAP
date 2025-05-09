
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from ResNet import model_urls
from ghostnet import GhostModule

__all__ = ['ResNet', 'resnet18_cbam0', 'resnet34_cbam', ]


def conv3x3(in_planes, out_planes, mode, stride=1):
    "3x3 convolution with padding"
    return GhostModule(in_planes, out_planes, mode, kernel_size=3, stride=stride, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class conv_block(nn.Module):

    def __init__(self, in_planes, planes, mode, stride=1):
        super(conv_block, self).__init__()
        self.conv = conv3x3(in_planes, planes, mode, stride)

    def forward(self, x):
        out = self.conv(x)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, mode, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_block(inplanes, planes, mode, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_block(planes, planes, mode)
        self.mode = mode

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, mode='normal'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.mode = mode
        self.conv1 = GhostModule(3, 64, mode, kernel_size=3, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.feature_dim = 512

        if self.mode == 'parallel_adapters':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.zero_()
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.weight.data.zero_()
                    m.bias.data.zero_()
        else:
            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d):
            #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #     elif isinstance(m, nn.BatchNorm2d):
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.mode, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.mode))

        return nn.Sequential(*layers)

    def switch(self, mode='normal'):
        for name, module in self.named_modules():
            if hasattr(module, 'mode'):
                module.mode = mode

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        dim = x.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)
        x = pool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet18_cbam0(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model
