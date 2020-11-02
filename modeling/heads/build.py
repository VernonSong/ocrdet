# -*- coding: utf-8 -*-
# @Time : 2020/8/12 5:48 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from utils.register import Registry
from modeling.heads.base_head import BaseHead

HEAD_REGISTRY = Registry("HEAD")
HEAD_REGISTRY.__doc__ = """
"""


def build_head(name, cfg, is_training) -> BaseHead:
    return HEAD_REGISTRY.get(name)(cfg, is_training)
