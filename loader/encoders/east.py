# -*- coding: utf-8 -*-
# @Time : 2020/10/23 6:00 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import numba as nb
import cv2
from shapely.geometry import Polygon
from loader.encoders.build import ENCODER_REGISTRY
from loader.encoders.base_encoder import BaseEncoder


# @nb.jit
def calcu_rbox_map(scores_map, boxes, rotation_angles):
  distance_map = np.zeros([scores_map.shape[0], scores_map.shape[1], 4])
  rotation_map = np.zeros([scores_map.shape[0], scores_map.shape[1], 1])
  rbox_map_index = np.where(scores_map > 0)
  rbox_map_index = np.stack(rbox_map_index, -1)
  for index in range(rbox_map_index.shape[0]):
    (row, col) = rbox_map_index[index]
    if scores_map[row][col] > 0:
      [x0, y0], [x1, y1], [x2, y2], [x3, y3] = boxes[scores_map[row][col] - 1]
      rotation_angle = rotation_angles[scores_map[row][col] - 1]
      p0 = np.array([x0, y0])
      p1 = np.array([x1, y1])
      p2 = np.array([x2, y2])
      p3 = np.array([x3, y3])
      center = np.array([col*4, row*4])
      d_top = np.linalg.norm(np.cross(p1 - p0, p0 - center)/np.linalg.norm(p1 - p0))
      d_right = np.linalg.norm(
          np.cross(p2 - p1, p1 - center)/np.linalg.norm(p2 - p1))
      d_bottom = np.linalg.norm(
          np.cross(p3 - p2, p2 - center)/np.linalg.norm(p3 - p2))
      d_left = np.linalg.norm(
          np.cross(p0 - p3, p3 - center)/np.linalg.norm(p0 - p3))
      distance_map[row][col] = [d_top, d_right, d_bottom, d_left]
      rotation_map[row][col] = [rotation_angle]
  return distance_map, rotation_map


@ENCODER_REGISTRY.register()
class EASTEncoder(BaseEncoder):
    """ Real-time Scene Text Detection with Differentiable Binarization
        (https://arxiv.org/pdf/1911.08947.pdf)
        部分实现参考PaddleOCR
        (https://github.com/PaddlePaddle/PaddleOCR)"""
    def __init__(self, cfg):
        input_shape = cfg.AUGMENT.INPUT_SHAPE
        self.stride = cfg.MODEL.DB.STRIDE
        self.rate = cfg.MODEL.DB.RATE
        self.target_shape = [int(input_shape[0]/self.stride), int(input_shape[1]/self.stride)]
        self.score_map = None
        self.distance_map = None
        self.rotation_map = None
        self.boxes = None
        self.rotation_angles = None
        self.idx = None

    def init_encoder(self):
        self.score_map = np.zeros(self.target_shape, dtype=np.int32)
        self.distance_map = np.zeros([4] + self.target_shape)
        self.rotation_map = np.zeros(self.target_shape)
        self.boxes = []
        self.rotation_angles = []
        self.idx = 0

    def calculate_encoder(self, polygon: dict):
        self.idx += 1
        if not polygon['fields']['training_tag']:
            self.boxes.append(np.zeros([8]))
            self.rotation_angles.append(0)
            return
        contour = polygon['contour']/self.stride
        p = Polygon(contour)
        if p.area <= 0:
            return
        try:
            # 不支持3边型
            shrinked_polygon = self._shrink_polygon(contour)
            self.score_map = cv2.fillPoly(self.score_map, np.asarray([shrinked_polygon], dtype=np.int32), self.idx)
            min_rect = cv2.minAreaRect(np.asarray(polygon['contour'], dtype=np.float32))
            rbox = cv2.boxPoints(min_rect)
            rbox = self._single_sort_vertex(rbox)
            (x_center, y_center), (dh, dw), rotation_angle = self._get_box(np.asarray(rbox))
            self.boxes.append(rbox)
            self.rotation_angles.append(rotation_angle)
        except:
            self.boxes.append(np.zeros([8]))
            self.rotation_angles.append(0)

    def _shrink_polygon(self, poly):
        '''
        摘自EAST(https://github.com/argman/EAST)
        fit a poly inside the origin poly, maybe bugs here...
        used for generate the score map
        :param poly: the text poly
        :param r: r in the paper
        :return: the shrinked poly
        '''
        # shrink ratio
        r = [None, None, None, None]
        for i in range(4):
            # print(poly)
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        R = 0.3
        # find the longer pair
        if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
            # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
            ## p0, p3
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
        else:
            ## p0, p3
            # print poly
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
        return poly

    def _single_sort_vertex(self, polygon: np.ndarray):
        """
        排序规则，在文本框旋转角度小于45度的前提下，第一个点为文本框左上角点，顺时针排序
        :param poly: numpy,shape:[4,2], [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        :return:
        """
        center = np.mean(polygon, axis=0)
        # 先排序成顺时针
        atan = []
        for x, y in zip(polygon[:, 0] - center[0], polygon[:, 1] - center[1]):
            atan.append(np.arctan2(x, y))
        atan = sorted(enumerate(atan), key=lambda x: -x[1])
        # 找文本框左上角点
        sorted_polygon = [np.array([polygon[j][0], polygon[j][1]]) for j, _ in atan]
        edge = (sorted_polygon[1] + sorted_polygon[0]) / 2 - center
        degree = np.degrees(np.arctan2(edge[0], edge[1]))
        if 45 < degree < 135:
            sorted_polygon = [sorted_polygon[3]] + sorted_polygon[0:3]
        elif -135 < degree < -45:
            sorted_polygon = sorted_polygon[1:] + [sorted_polygon[0]]
        elif -45 < degree < 45:
            sorted_polygon = sorted_polygon[2:4] + sorted_polygon[0:2]

        return np.array(sorted_polygon)

    def _get_box(self, box):
        x_center = np.mean(box[:, 0])
        y_center = np.mean(box[:, 1])
        side1_center = (np.mean(box[0:2, 0]), np.mean(box[0:2, 1]))
        side2_center = (np.mean(box[1:3, 0]), np.mean(box[1:3, 1]))
        d1 = np.sqrt((side1_center[0] - x_center) ** 2 + (side1_center[1] - y_center) ** 2)
        d2 = np.sqrt((side2_center[0] - x_center) ** 2 + (side2_center[1] - y_center) ** 2)
        dh = d1
        dw = d2
        x = box[1] - box[0]
        y = [1, 0]
        costheta = x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
        rotation_angle = np.arccos(costheta)
        if box[1][1] - box[0][1] < 0:
            rotation_angle *= -1
        return (x_center, y_center), (dh, dw), rotation_angle

    def create_target(self) -> dict:
        self.distance_map, self.rotation_map = calcu_rbox_map(self.score_map, self.boxes, self.rotation_angles)
        self.score_map[self.score_map > 1] = 1
        return {
            'east_score_map': np.expand_dims(self.score_map, 0),
            'east_distance_map': np.transpose(self.distance_map, [2, 0, 1]),
            'east_rotation_map': np.transpose(self.rotation_map, [2, 0, 1])
        }


