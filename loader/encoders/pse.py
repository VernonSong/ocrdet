# -*- coding: utf-8 -*-
# @Time : 2020/8/19 11:25 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import cv2
import pyclipper
from shapely.geometry import Polygon
from loader.encoders.build import ENCODER_REGISTRY
from loader.encoders.base_encoder import BaseEncoder


@ENCODER_REGISTRY.register()
class PSEEncoder(BaseEncoder):
    """ Shape Robust Text Detection with Progressive Scale Expansion Network
        (https://arxiv.org/pdf/1903.12473.pdf)"""
    def __init__(self, cfg):
        input_shape = cfg.AUGMENT.INPUT_SHAPE
        self.stride = cfg.MODEL.PSE.STRIDE
        self.n = cfg.MODEL.PSE.N
        self.m = cfg.MODEL.PSE.M
        self.target_shape = [self.n, int(input_shape[0]/self.stride), int(input_shape[1]/self.stride)]
        self.pse_map = None

    def init_encoder(self):
        self.pse_map = np.zeros(self.target_shape, dtype=np.int32)

    def calculate_encoder(self, polygon: dict):
        if not polygon['fields']['training_tag']:
            return
        contour = polygon['contour']/self.stride
        p = Polygon(contour)
        if p.area <= 0:
            return
        try:
            for i in range(1, self.n + 1):
                rate = 1 - (1 - self.m) * (self.n - i) / (self.n - 1)
                distance = p.area * (
                        1 - np.power(rate, 2)) / p.length
                offset = pyclipper.PyclipperOffset()
                offset.AddPath(contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked_polygon = np.array(offset.Execute(-distance)[0])
                cv2.fillPoly(self.pse_map[i], [shrinked_polygon.astype(np.int32)], 1.0)
        except:
            pass

    def create_target(self) -> dict:
        return {
            'pse_map': self.pse_map
        }
