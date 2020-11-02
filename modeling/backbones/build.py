# -*- coding: utf-8 -*-
# @Time : 2020/8/5 5:06 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from utils.register import Registry
from modeling.layers.norm import NormLayer
from torch import nn

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
"""


def build_backbone(cfg) -> nn.Module:
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.BACKBONE.NAME
    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    pretrained_dir = cfg.MODEL.BACKBONE.PRETRAINED_DIR
    norm_layer = NormLayer(cfg.MODEL.BACKBONE.NORM)
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    return BACKBONE_REGISTRY.get(name)(pretrained, model_dir=pretrained_dir, norm_layer=norm_layer, freeze_at=freeze_at)
