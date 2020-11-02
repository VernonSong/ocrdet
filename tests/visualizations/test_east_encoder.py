# -*- coding: utf-8 -*-
# @Time : 2020/10/23 10:17 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import cv2
import matplotlib.pylab as plt
from structures import Polygons
from loader.encoders.east import EASTEncoder
from config import config as cfg

def rbox2poly(rbox, point, rotation_angle):
    d_top, d_right, d_bottom, d_left = rbox
    x_center, y_center = point
    def get_xy(dw, dh):
        x = int(dw * np.cos(rotation_angle) - dh * np.sin(rotation_angle) + x_center)
        y = int(dw * np.sin(rotation_angle) + dh * np.cos(rotation_angle) + y_center)
        return x, y

    x0, y0 = get_xy(-d_left, -d_top)
    x1, y1 = get_xy(d_right, -d_top)
    x2, y2 = get_xy(d_right, d_bottom)
    x3, y3 = get_xy(-d_left, d_bottom)
    return np.array([x0, y0, x1, y1, x2, y2, x3, y3])

def test_east_encoder():
    east_encoder = EASTEncoder(cfg)
    contours = [np.array([[200, 200], [240, 200], [280, 240], [280, 280], [200, 280]]),
              np.array([[100, 100], [140, 140], [140, 100]])]
    polygons = Polygons(contours)
    polygons.add_field('training_tag', [True, True])
    num_polygons = len(polygons)
    east_encoder.init_encoder()
    for idx in range(num_polygons):
        east_encoder.calculate_encoder(polygons.get(idx))
    target = east_encoder.create_target()

    plt.figure("img")
    plt.figure(figsize=(12, 12))
    plt.imshow(target['east_score_map'][0])
    plt.show()

    canvas = np.zeros([512, 512])
    scores_map = target['east_score_map'].transpose([1, 2, 0])
    distance_map = target['east_distance_map'].transpose([1, 2, 0])
    rotation_map = target['east_rotation_map'].transpose([1, 2, 0])
    box_map_index = np.where(scores_map[:,:,0] > 0)
    box_map_index = np.stack(box_map_index, -1)

    for index in range(box_map_index.shape[0]):
        (row, col) = box_map_index[index]
        rotation_angle = rotation_map[row][col]
        d_top, d_right, d_bottom, d_left = distance_map[row][col]
        print(d_top, d_right, d_bottom, d_left)
        x0, y0, x1, y1, x2, y2, x3, y3 = rbox2poly(
            [d_top, d_right, d_bottom, d_left], (col*4, row*4), rotation_angle)
        print(x0, y0, x1, y1, x2, y2, x3, y3)
        cv2.line(canvas, (x0, y0), (x1, y1), 1, 1)
        cv2.line(canvas, (x1, y1), (x2, y2), 1, 1)
        cv2.line(canvas, (x2, y2), (x3, y3), 1, 1)
        cv2.line(canvas, (x3, y3), (x0, y0), 1, 1)
    plt.figure("img")
    plt.figure(figsize=(12, 12))
    plt.imshow(canvas)
    plt.show()

if __name__ == "__main__":
    test_east_encoder()
