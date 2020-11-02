# -*- coding: utf-8 -*-
# @Time : 2020/10/30 10:03 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from abc import ABCMeta, abstractmethod


class BasePostprocess(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def _rescale(self, p: 'Polygons', param: dict) -> 'Polygons':
        scale = [1 / param['scale'][1], 1/param['scale'][0]]
        crop = [-param['pad'][0], -param['pad'][1]]
        p.scale_and_pad(scale, crop, recover=True)
        return p

    @abstractmethod
    def __call__(self, outputs: dict):
        pass
