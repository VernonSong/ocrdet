# -*- coding: utf-8 -*-
# @Time : 2020/8/18 3:46 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import time
import matplotlib.pylab as plt
from utils.drawing_util import draw_polygons
from loader.datasets import build_dataset


def test_labelme():
    dataset = build_dataset('LabelMe', '../../data/labelme')
    print(len(dataset))
    average_time = []
    for i in range(10):
        tic = time.time()
        data = dataset[i]
        toc = time.time()
        canvas = draw_polygons(data['image'], data['polygons'])
        average_time.append(toc - tic)
        plt.figure("img")
        plt.figure(figsize=(12, 12))
        plt.imshow(np.array(canvas, dtype=np.uint8))
        plt.show()
    print(np.mean(np.array(average_time)))


if __name__ == "__main__":
    test_labelme()