# -*- coding: utf-8 -*-
# @Time : 2020/7/30 4:12 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from loader.encoders.build import build_encoder
from loader.encoders.training_mask import TrainingMaskEncoder
from loader.encoders.db import DBEncoder
from loader.encoders.semantic import SemanticEncoder


__all__ = [
    "build_encoder",
]
