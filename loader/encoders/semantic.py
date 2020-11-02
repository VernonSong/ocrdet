# -*- coding: utf-8 -*-
# @Time : 2020/8/18 6:10 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import cv2
from loader.encoders.build import ENCODER_REGISTRY
from loader.encoders.base_encoder import BaseEncoder


@ENCODER_REGISTRY.register()
class SemanticEncoder(BaseEncoder):
    """ 用于多任务学习的语义分割"""
    def __init__(self, cfg):
        input_shape = cfg.AUGMENT.INPUT_SHAPE
        self.stride = cfg.MODEL.SEMANTIC.STRIDE
        self.target_shape = [int(input_shape[0]/self.stride), int(input_shape[1]/self.stride)]
        self.semantic_map = None

    def init_encoder(self):
        self.semantic_map = np.zeros(self.target_shape, dtype=np.int32)

    def calculate_encoder(self, polygon: dict):
        contour = [np.around(polygon['contour']/self.stride).astype(np.int32)]
        cv2.fillPoly(self.semantic_map, contour, 1)

    def create_target(self) -> dict:
        return {
            'semantic_map': np.expand_dims(self.semantic_map, 0)
        }