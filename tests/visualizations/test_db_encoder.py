# -*- coding: utf-8 -*-
# @Time : 2020/8/4 11:12 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import matplotlib.pylab as plt
from structures import Polygons
from loader.encoders.db import DBEncoder
from config import config as cfg


def test_db_encoder():
    db_encoder = DBEncoder(cfg)
    contours = [np.array([[0, 0], [40, 0], [80, 40], [80, 80], [0, 80]]),
              np.array([[100, 100], [140, 140], [140, 100]])]
    polygons = Polygons(contours)
    polygons.add_field('training_tag', [True, True])
    num_polygons = len(polygons)
    db_encoder.init_encoder()
    for idx in range(num_polygons):
        db_encoder.calculate_encoder(polygons.get(idx))
    target = db_encoder.create_target()

    plt.figure("img")
    plt.figure(figsize=(12, 12))
    plt.imshow(target['db_probability_map'][0])
    plt.show()

    plt.figure("img")
    plt.figure(figsize=(12, 12))
    plt.imshow(target['db_threshold_map'][0])
    plt.show()

    plt.figure("img")
    plt.figure(figsize=(12, 12))
    plt.imshow(target['db_threshold_mask'][0])
    plt.show()


if __name__ == "__main__":
    test_db_encoder()





