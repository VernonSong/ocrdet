# -*- coding: utf-8 -*-
# @Time : 2020/8/3 2:42 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import time
import matplotlib.pylab as plt
from utils.drawing_util import draw_polygons
from loader.datasets import build_dataset
from loader.augment import Augment
from config import config as cfg


def test_augment():
    dataset = build_dataset('ICDAR2015', '../../data/icdar2015')
    dataset = Augment(dataset, cfg)
    print(len(dataset))
    average_time = []
    for i in range(10):
        tic = time.time()
        data = dataset[10]
        toc = time.time()
        canvas = draw_polygons(data['image'], data['polygons'])
        average_time.append(toc - tic)
        plt.figure("img")
        plt.figure(figsize=(12, 12))
        plt.imshow(np.array(canvas, dtype=np.uint8))
        plt.show()
    print(np.mean(np.array(average_time)))


if __name__ == "__main__":
    test_augment()

