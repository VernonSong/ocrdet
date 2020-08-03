# -*- coding: utf-8 -*-
# @Time : 2020/7/28 3:43 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from utils.register import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
"""


def build_dataset(cfg):

    name = cfg.DATA.NAME
    return DATASET_REGISTRY.get(name)(cfg)