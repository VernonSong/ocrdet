# -*- coding: utf-8 -*-
# @Time : 2020/8/12 5:48 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch.nn as nn
import torch.nn.functional as F
from modeling.layers.norm import NormLayer
from utils.weight_init import *

__all__ = [
    'BaseHead'
]


class DefaultPredictor1X(nn.Module):
    def __init__(self, in_planes, out_planes, norm, predict_kernel=1):
        super(DefaultPredictor1X, self).__init__()
        use_bias = True if len(norm.mode) == 0 else False
        subnet = []
        subnet.append(nn.Conv2d(in_planes, in_planes // 4, 3, padding=1, bias=use_bias))
        if len(norm.mode) > 0:
            subnet.append(norm(in_planes // 4))
        subnet.append(nn.ReLU(inplace=True))
        self.subnet = nn.Sequential(*subnet)
        # predictor根据任务不同采用同不同的初始化方式
        self.predictor = nn.Conv2d(in_planes // 4, out_planes, predict_kernel, padding=predict_kernel//2)

        for m in self.subnet:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.subnet(x)
        x = self.predictor(x)
        return x


class DefaultPredictor2X(nn.Module):
    def __init__(self, in_planes, out_planes, norm, predict_kernel=1):
        super(DefaultPredictor2X, self).__init__()
        use_bias = True if len(norm.mode) == 0 else False
        subnet = []
        subnet.append(nn.Conv2d(in_planes, in_planes // 4, 3, padding=1, bias=use_bias))
        if len(norm.mode) > 0:
            subnet.append(norm(in_planes // 4))
        subnet.append(nn.ReLU(inplace=True))
        subnet.append(nn.Upsample(scale_factor=2))
        subnet.append(nn.Conv2d(in_planes // 4, in_planes // 4, 3, padding=1))
        if len(norm.mode) > 0:
            subnet.append(norm(in_planes // 4))
        subnet.append(nn.ReLU(inplace=True))
        self.subnet = nn.Sequential(*subnet)
        self.predictor = nn.Conv2d(in_planes // 4, out_planes, predict_kernel, padding=predict_kernel // 2)
        for m in self.subnet:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.subnet(x)
        x = self.predictor(x)
        return x


class DefaultPredictor4X(nn.Module):
    def __init__(self, in_planes, out_planes, norm, predict_kernel=1):
        super(DefaultPredictor4X, self).__init__()
        use_bias = True if len(norm.mode) == 0 else False
        subnet = []
        subnet.append(nn.Conv2d(in_planes, in_planes // 4, 3, padding=1, bias=use_bias))
        if len(norm.mode) > 0:
            subnet.append(norm(in_planes // 4))
        subnet.append(nn.ReLU(inplace=True))
        subnet.append(nn.ConvTranspose2d(in_planes // 4, in_planes // 4, 2, 2, bias=use_bias))
        if len(norm.mode) > 0:
            subnet.append(norm(in_planes // 4))
        subnet.append(nn.ReLU(inplace=True))
        self.subnet = nn.Sequential(*subnet)
        self.predictor = nn.ConvTranspose2d(in_planes // 4, out_planes, 2, 2, bias=use_bias)
        for m in self.subnet:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,
                                  0)

    def forward(self, x):
        x = self.subnet(x)
        x = self.predictor(x)
        return x


class BaseHead(nn.Module):
    def __init__(self, cfg, is_training):
        super(BaseHead, self).__init__()
        self.id = str(id)
        self.is_training = is_training
        layers_planes = cfg.MODEL.META_ARCH.OUT_PLANES
        self.layers_planes = {
            'p2': layers_planes,
            'p3': layers_planes,
            'p4': layers_planes,
            'p5': layers_planes,
            'p6': layers_planes,
            'p7': layers_planes,
            'fuse': layers_planes * 4
        }
        self.default_predictor = {
            '1': DefaultPredictor1X,
            '2': DefaultPredictor2X,
            '4': DefaultPredictor4X,
        }
        self.norm_layer = NormLayer(cfg.MODEL.HEAD.NORM)



