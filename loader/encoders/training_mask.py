# -*- coding: utf-8 -*-
# @Time : 2020/8/5 9:46 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import cv2
from loader.encoders.build import ENCODER_REGISTRY
from loader.encoders.base_encoder import BaseEncoder


@ENCODER_REGISTRY.register()
class TrainingMaskEncoder(BaseEncoder):
    def __init__(self, cfg):
        input_shape = cfg.AUGMENT.INPUT_SHAPE
        self.target_shape = [int(input_shape[0]), int(input_shape[1])]
        self.training_mask = None

    def init_encoder(self):
        self.training_mask = np.ones(self.target_shape, dtype=np.int32)

    def calculate_encoder(self, polygon: dict):
        if polygon['fields']['training_tag']:
            return
        contour = [np.around(polygon['contour']).astype(np.int32)]
        cv2.fillPoly(self.training_mask, contour, 0)

    def create_target(self) -> dict:
        return {
            'training_mask': np.expand_dims(self.training_mask, 0)
        }