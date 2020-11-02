# -*- coding: utf-8 -*-
# @Time : 2020/8/4 9:02 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from abc import ABCMeta, abstractmethod


class BaseEncoder(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, cfg):
        pass

    @abstractmethod
    def init_encoder(self):
        pass

    @abstractmethod
    def calculate_encoder(self, polygon: dict):
        pass

    @abstractmethod
    def create_target(self) -> dict:
        pass





