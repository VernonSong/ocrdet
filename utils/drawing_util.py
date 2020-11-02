# -*- coding: utf-8 -*-
# @Time : 2020/8/2 11:36 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import cv2
import numpy as np
from structures import Polygons

COLORS = (((255, 0, 0), (205, 0, 0), (155, 0, 0), (105, 0, 0)),
          ((0, 255, 0), (0, 205, 0), (0, 155, 0), (0, 105, 0)),
          ((0, 0, 255), (0, 0, 205), (0, 0, 155), (0, 0, 105)),
          ((255, 255, 0), (205, 205, 0), (155, 155, 0), (105, 105, 0)),
          ((255, 0, 255), (205, 0, 205), (155, 0, 155), (105, 0, 105)),
          ((0, 255, 255), (0, 205, 205), (0, 155, 155), (0, 105, 105)),
          ((255, 0, 0), (205, 0, 0), (155, 0, 0), (105, 0, 0)),
          ((0, 255, 0), (0, 205, 0), (0, 155, 0), (0, 105, 0)),
          ((0, 0, 255), (0, 0, 205), (0, 0, 155), (0, 0, 105)),
          ((255, 255, 0), (205, 205, 0), (155, 155, 0), (105, 105, 0)),
          ((255, 0, 255), (205, 0, 205), (155, 0, 155), (105, 0, 105)),
          ((0, 255, 255), (0, 205, 205), (0, 155, 155), (0, 105, 105)),
          ((255, 255, 255), (205, 205, 205), (155, 155, 155), (105, 105, 105)))


def draw_polygons(image: np.ndarray,
                  polygons: Polygons,
                  thickness=2):
    canvas = image.copy().astype(np.uint8)
    num = len(polygons)
    if num == 0:
        return canvas

    for idx in range(num):
        item_dict = polygons.get(idx)
        points = item_dict['contour'].astype(np.int32)
        if polygons.has_field('training_tag'):
            if not item_dict['fields']['training_tag']:
                lines_color = COLORS[-1]
        if not polygons.has_field('training_tag') or item_dict['fields']['training_tag']:
            lines_color = COLORS[0]
        for i in range(len(points)):
            cv2.line(canvas, (points[i][0], points[i][1]), (points[i-1][0], points[i-1][1]), lines_color[0], thickness)

    return canvas
