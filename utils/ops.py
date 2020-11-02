# -*- coding: utf-8 -*-
# @Time : 2020/7/31 1:51 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import cv2

def resize_and_pad(
        image: np.ndarray,
        target_shape: list,
        polygons=None,
        pad_mode='none'):
    if not polygons is None:
        p = polygons.copy()
    else:
        p = None
    h, w = image.shape[:2]
    left = 0
    top = 0
    # 如果不pad，则破坏aspect ratio进行resize
    if pad_mode == 'none':
        scale = (target_shape[0] / h, target_shape[1] / w)
        image = cv2.resize(image, (target_shape[1], target_shape[0]))
    # 如果pad，则保持aspect ratio
    elif pad_mode == 'left':
        imagebase = np.zeros(shape=(target_shape[0], target_shape[1], 3), dtype=np.uint8)
        scale = h / target_shape[0]
        new_w = w / scale
        new_w = int(new_w)
        if new_w > target_shape[1]:
            new_w = target_shape[1]
        image = cv2.resize(image, (int(new_w), int(target_shape[0])))
        imagebase[:,:new_w,:] = image
        image = imagebase
    else:
        scale = np.minimum(target_shape[0] / h, target_shape[1] / w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        scale = (scale, scale)
        h, w = image.shape[:2]
        if pad_mode == 'center':
            top = int(np.abs(target_shape[0] - h) / 2)
            left = int(np.abs(target_shape[1] - w) / 2)
        elif pad_mode == 'random':
            top = 0
            if target_shape[0] - h > 0:
                top = np.random.randint(0, int(np.abs(target_shape[0] - h)))
            left = 0
            if target_shape[1] - w > 0:
                left = np.random.randint(0, int(np.abs(target_shape[1] - w)))

        image = cv2.copyMakeBorder(
            image, top, np.abs(target_shape[0] - h) - top, left, np.abs(target_shape[1] - w) - left,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    if p is not None:
        p.scale_and_pad(scale, (left, top))
        return image, p, scale, (left, top)
    else:
        return image, scale, (left, top)


