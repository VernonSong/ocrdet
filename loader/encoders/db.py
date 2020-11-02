# -*- coding: utf-8 -*-
# @Time : 2020/7/30 4:25 下午
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
class DBEncoder(BaseEncoder):
    """ Real-time Scene Text Detection with Differentiable Binarization
        (https://arxiv.org/pdf/1911.08947.pdf)
        部分实现参考PaddleOCR
        (https://github.com/PaddlePaddle/PaddleOCR)"""
    def __init__(self, cfg):
        input_shape = cfg.AUGMENT.INPUT_SHAPE
        self.stride = cfg.MODEL.DB.STRIDE
        self.rate = cfg.MODEL.DB.RATE
        self.target_shape = [int(input_shape[0]/self.stride), int(input_shape[1]/self.stride)]
        self.probability_map = None
        self.threshold_map = None
        self.threshold_mask = None

    def init_encoder(self):
        self.probability_map = np.zeros(self.target_shape, dtype=np.int32)
        self.threshold_map = np.zeros(self.target_shape)
        self.threshold_mask = np.zeros(self.target_shape)

    def calculate_encoder(self, polygon: dict):
        if not polygon['fields']['training_tag']:
            return
        contour = polygon['contour']/self.stride
        p = Polygon(contour)
        if p.area <= 0:
            return
        distance = p.area * (1 - np.power(self.rate, 2)) / p.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        try:
            shrinked_polygon = np.array(offset.Execute(-distance)[0])
            dilated_polygon = np.array(offset.Execute(distance)[0])
            cv2.fillPoly(self.probability_map, [shrinked_polygon.astype(np.int32)], 1.0)
            cv2.fillPoly(self.threshold_mask, [dilated_polygon.astype(np.int32)], 1.0)

            xmin = dilated_polygon[:, 0].min()
            ymin = dilated_polygon[:, 1].min()
            xmax = dilated_polygon[:, 0].max()
            ymax = dilated_polygon[:, 1].max()
            width = xmax - xmin + 1
            height = ymax - ymin + 1
            contour[:, 0] = contour[:, 0] - xmin
            contour[:, 1] = contour[:, 1] - ymin

            xs, ys = np.meshgrid(np.arange(width), np.arange(height))
            distance_map = np.zeros((len(contour), height, width), dtype=np.float32)
            for i in range(len(contour)):
                absolute_distance = self._distance(xs, ys, contour[i-1], contour[i])
                distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
            distance_map = distance_map.min(axis=0)
            xmin_valid = min(max(0, xmin), self.target_shape[1] - 1)
            xmax_valid = min(max(0, xmax), self.target_shape[1] - 1)
            ymin_valid = min(max(0, ymin), self.target_shape[0] - 1)
            ymax_valid = min(max(0, ymax), self.target_shape[0] - 1)
            self.threshold_map[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
                1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height, xmin_valid - xmin:xmax_valid - xmax + width],
                self.threshold_map[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
        except:
            pass

    def _distance(self, xs: np.ndarray, ys: np.ndarray, point_1: np.ndarray, point_2: np.ndarray):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])
        cosin = (square_distance - square_distance_1 - square_distance_2) / (
                2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin /
                         square_distance)
        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result

    def create_target(self) -> dict:
        self.threshold_map = self.threshold_map * (0.7 - 0.3) + 0.3
        return {
            'db_probability_map': np.expand_dims(self.probability_map, 0),
            'db_threshold_map': np.expand_dims(self.threshold_map, 0),
            'db_threshold_mask': np.expand_dims(self.threshold_mask, 0)
        }

