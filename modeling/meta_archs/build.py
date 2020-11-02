# -*- coding: utf-8 -*-
# @Time : 2020/8/12 5:18 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from utils.register import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
"""


def build_meta_arch(cfg) -> 'nn.Module':
    name = cfg.MODEL.META_ARCH.NAME
    return META_ARCH_REGISTRY.get(name)(cfg)
