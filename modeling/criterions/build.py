# -*- coding: utf-8 -*-
# @Time : 2020/8/15 10:30 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from utils.register import Registry

CRITERION_REGISTRY = Registry("HEAD")
CRITERION_REGISTRY.__doc__ = """
"""


def build_criterion(name, cfg) -> 'nn.Module':
    return CRITERION_REGISTRY.get(name)(cfg)
