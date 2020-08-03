# -*- coding: utf-8 -*-
# @Time : 2020/7/31 2:44 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import cv2
import imageio


def imread(path: str) -> np.ndarray:
    """
    读取图片，支持gif
    :param path: 图片路径
    :return: 图片
    """
    image = cv2.imread(path)

    if image is None:
        image = imageio.mimread(path)
        image = np.array(image)[0][:, :, :3]
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
