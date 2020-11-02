# -*- coding: utf-8 -*-
# @Time : 2020/7/30 4:09 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from utils.register import Registry
from loader.encoders.base_encoder import BaseEncoder

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
"""


def build_encoder(name, cfg) -> BaseEncoder:
    return ENCODER_REGISTRY.get(name)(cfg)
