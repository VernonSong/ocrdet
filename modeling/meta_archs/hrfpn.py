# -*- coding: utf-8 -*-
# @Time : 2020/10/28 12:24 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch
import torch.nn.functional as F
from torch import nn
from modeling.backbones import build_backbone
from modeling.layers.norm import NormLayer
from utils.weight_init import *
from modeling.meta_archs.build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class HRFPN(nn.Module):
    def __init__(self, cfg):
        super(HRFPN, self).__init__()
        out_planes = cfg.MODEL.META_ARCH.OUT_PLANES
        norm = NormLayer(cfg.MODEL.META_ARCH.NORM)
        self._fuse_type = 'sum'
        output_convs = []
        self.backbone = build_backbone(cfg)
        self.strides = [4, 8, 16, 32]
        self.use_p6p7 = cfg.MODEL.META_ARCH.USE_P6P7
        self.use_fuse = cfg.MODEL.META_ARCH.USE_FUSE
        in_planes = self.backbone.out_planes
        if self.use_p6p7:
            self.extra_block = LastLevelP6P7(in_channels=in_planes[-1], out_channels=out_planes)
        use_bias = True if len(norm.mode) == 0 else False
        for idx, in_planes_per_stage in enumerate(in_planes[::-1]):
            output_conv = nn.Conv2d(in_planes_per_stage, out_planes, kernel_size=1, bias=use_bias)
            if len(norm.mode) > 0:
                output_convs.append(nn.Sequential(output_conv, norm(out_planes), nn.ReLU(inplace=True)))
            else:
                output_convs.append(nn.Sequential(output_conv, nn.ReLU(inplace=True)))
        self.output_convs = nn.ModuleList(output_convs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                c2_xavier_fill(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        x = [features['c5'], features['c4'], features['c3'], features['c2']]
        results = []
        for features, output_conv in zip(x, self.output_convs):
            results.insert(0, output_conv(features))
        outputs = {
            'p2': results[0],
            'p3': results[1],
            'p4': results[2],
            'p5': results[3]
        }

        if self.use_fuse:
            p5_up = F.interpolate(results[3], scale_factor=8)
            p4_up = F.interpolate(results[2], scale_factor=4)
            p3_up = F.interpolate(results[1], scale_factor=2)
            p2_up = results[0]
            outputs.update({'fuse': torch.cat([p2_up, p3_up, p4_up, p5_up], 1)})

        if self.use_p6p7:
            extra_outputs = self.extra_block(x[0])
            outputs.update(extra_outputs)

        return outputs


class LastLevelP6P7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for m in [self.p6, self.p7]:
            c2_xavier_fill(m)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return {
            'p6': p6,
            'p7': p7,
        }



