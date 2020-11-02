# -*- coding: utf-8 -*-
# @Time : 2020/7/28 3:56 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from loader.datasets.build import build_dataset
from loader.datasets.icdar2015 import ICDAR2015
from loader.datasets.icdar2017 import ICDAR2017
from loader.datasets.mtwi import MTWI
from loader.datasets.labelme import LabelMe
from loader.datasets.original import Original


__all__ = [
    "build_dataset",
]