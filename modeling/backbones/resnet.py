 # -*- coding: utf-8 -*-
# @Time : 2020/8/5 9:55 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from torch import nn
import numpy as np
from modeling.backbones.build import *
from torch.hub import load_state_dict_from_url
from modeling.backbones.resnet_util import *
"""
resnet v1: Deep Residual Learning for Image Recognition(https://arxiv.org/abs/1512.03385)
"""

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'resnet50_dfonv']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ResNet(nn.Module):

    def __init__(self, blocks, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.out_planes = []
        for idx, block in enumerate(blocks):
            if block.__name__ == 'BasicBlock':
                self.out_planes.append(int(self.inplanes*np.exp2(idx)))
            else:
                self.out_planes.append(int(self.inplanes*np.exp2(idx)*4))
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[0], 64, layers[0])
        self.layer2 = self._make_layer(blocks[1], 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(blocks[2], 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(blocks[3], 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * blocks[3].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the modeling by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.conv1.requires_grad_(False)
            self.bn1.requires_grad_(False)
        if freeze_at >= 2:
            self.layer1.requires_grad_(False)
        if freeze_at >= 3:
            self.layer2.requires_grad_(False)
        if freeze_at >= 4:
            self.layer3.requires_grad_(False)
        if freeze_at >= 5:
            self.layer4.requires_grad_(False)

        return self

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        res1 = self.maxpool(x)

        res2 = self.layer1(res1)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)

        return {'c2': res2,
                'c3': res3,
                'c4': res4,
                'c5': res5}


def _resnet(arch, block, layers, pretrained, model_dir, freeze_at, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], model_dir=model_dir)
        model.load_state_dict(state_dict)
    return model.freeze(freeze_at)

@BACKBONE_REGISTRY.register()
def resnet18(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    r"""ResNet-18 modeling from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a modeling pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', [BasicBlock, BasicBlock, BasicBlock, BasicBlock], [2, 2, 2, 2], pretrained, model_dir,
                   freeze_at, **kwargs)

@BACKBONE_REGISTRY.register()
def resnet34(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    r"""ResNet-34 modeling from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a modeling pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', [BasicBlock, BasicBlock, BasicBlock, BasicBlock], [3, 4, 6, 3], pretrained, model_dir,
                   freeze_at, **kwargs)

@BACKBONE_REGISTRY.register()
def resnet50(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    r"""ResNet-50 modeling from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a modeling pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', [Bottleneck, Bottleneck, Bottleneck, Bottleneck], [3, 4, 6, 3], pretrained, model_dir,
                   freeze_at, **kwargs)

@BACKBONE_REGISTRY.register()
def resnet50_dfonv(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    return _resnet('resnet50', [Bottleneck, Bottleneck, DeformBottleneckBlock, DeformBottleneckBlock],
                   [3, 4, 6, 3], pretrained, model_dir, freeze_at, **kwargs)

@BACKBONE_REGISTRY.register()
def resnet101(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    r"""ResNet-101 modeling from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a modeling pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', [Bottleneck, Bottleneck, Bottleneck, Bottleneck], [3, 4, 23, 3], pretrained, model_dir,
                   freeze_at, **kwargs)

@BACKBONE_REGISTRY.register()
def resnet152(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    r"""ResNet-152 modeling from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a modeling pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', [Bottleneck, Bottleneck, Bottleneck, Bottleneck], [3, 8, 36, 3], pretrained, model_dir,
                   freeze_at, **kwargs)

@BACKBONE_REGISTRY.register()
def resnext50_32x4d(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    r"""ResNeXt-50 32x4d modeling from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a modeling pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', [Bottleneck, Bottleneck, Bottleneck, Bottleneck], [3, 4, 6, 3],
                   pretrained, model_dir, freeze_at, **kwargs)

@BACKBONE_REGISTRY.register()
def resnext101_32x8d(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    r"""ResNeXt-101 32x8d modeling from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a modeling pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', [Bottleneck, Bottleneck, Bottleneck, Bottleneck], [3, 4, 23, 3],
                   pretrained, model_dir, freeze_at, **kwargs)

@BACKBONE_REGISTRY.register()
def wide_resnet50_2(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    r"""Wide ResNet-50-2 modeling from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The modeling is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a modeling pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', [Bottleneck, Bottleneck, Bottleneck, Bottleneck], [3, 4, 6, 3],
                   pretrained, model_dir, freeze_at, **kwargs)

@BACKBONE_REGISTRY.register()
def wide_resnet101_2(pretrained=False, model_dir=None, freeze_at=0, **kwargs):
    r"""Wide ResNet-101-2 modeling from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The modeling is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a modeling pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', [Bottleneck, Bottleneck, Bottleneck, Bottleneck], [3, 4, 23, 3],
                   pretrained, model_dir, freeze_at, **kwargs)



