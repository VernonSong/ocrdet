# -*- coding: utf-8 -*-
# @Time : 2020/8/5 9:56 上午
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
class FPN(nn.Module):
    def __init__(self, cfg):
        super(FPN, self).__init__()
        out_planes = cfg.MODEL.META_ARCH.OUT_PLANES
        inter_planes = cfg.MODEL.META_ARCH.INTER_PLANES
        norm = NormLayer(cfg.MODEL.META_ARCH.NORM)
        self._fuse_type = 'sum'
        lateral_convs = []
        output_convs = []
        self.backbone = build_backbone(cfg)
        self.strides = [4, 8, 16, 32]
        self.use_p6p7 = cfg.MODEL.META_ARCH.USE_P6P7
        self.use_fuse = cfg.MODEL.META_ARCH.USE_FUSE
        if self.use_p6p7:
            """使用p5而不是c5 fcos"""
            self.extra_block = LastLevelP6P7(in_channels=inter_planes, out_channels=out_planes)
        in_planes = self.backbone.out_planes
        use_bias = True if len(norm.mode) == 0 else False
        for idx, in_planes_per_stage in enumerate(in_planes[::-1]):
            lateral_conv = nn.Conv2d(in_planes_per_stage, inter_planes, kernel_size=1, bias=use_bias)
            output_conv = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if len(norm.mode) > 0:
                lateral_convs.append(nn.Sequential(lateral_conv, norm(inter_planes)))
                output_convs.append(nn.Sequential(output_conv, norm(out_planes)))
            else:
                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.output_convs = nn.ModuleList(output_convs)

        if self.use_fuse:
            fuse_planes = out_planes * 4
            self.fuse_out_convs = nn.Conv2d(fuse_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         c2_xavier_fill(m)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        for m in self.lateral_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_xavier_fill(m)
        # for m in self.output_convs.modules():
        #     if isinstance(m, nn.Conv2d):
        #         c2_msra_fill(m)

    def forward(self, x):
        features = self.backbone(x)
        x = [features['c5'], features['c4'], features['c3'], features['c2']]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(x[1:], self.lateral_convs[1:], self.output_convs[1:]):

            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))
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
            # outputs.update({'fuse': self.fuse_out_convs(torch.cat([p2_up, p3_up, p4_up, p5_up], 1))})

        if self.use_p6p7:
            extra_outputs = self.extra_block(results[3])
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
