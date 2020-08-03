# -*- coding: utf-8 -*-
# @Time : 2020/7/30 4:09 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from utils.register import Registry


LABEL_ENCODER_REGISTRY = Registry("LABEL_ENCODER")
LABEL_ENCODER_REGISTRY.__doc__ = """
"""


def build_label_encoder(name, stride):
    return LABEL_ENCODER_REGISTRY.get(name)(stride)
