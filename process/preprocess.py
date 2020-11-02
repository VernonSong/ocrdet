# -*- coding: utf-8 -*-
# @Time : 2020/7/3 9:57 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from abc import ABCMeta, abstractmethod
import numpy as np
from utils.ops import resize_and_pad


class BasePreprocess(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _resize(self, image: np.ndarray):
        pass

    @abstractmethod
    def __call__(self, outputs: dict):
        pass


class FixLongSidePreprocess(BasePreprocess):
    """
    resize with fixed short side
    """
    def __init__(self, long_side=640):
        self.long_side = long_side

    def _resize(self, image: np.ndarray):
        h, w = image.shape[:2]
        scale = self.long_side / np.maximum(h, w)
        resize_w = w
        resize_h = h
        resize_h = int(resize_h * scale)
        resize_w = int(resize_w * scale)
        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
        resize_h = max(32, resize_h)
        resize_w = max(32, resize_w)
        image, scale, pad = resize_and_pad(image, (resize_h, resize_w), pad_mode='none')
        return image, scale, pad

    def __call__(self, images: np.ndarray):
        if isinstance(images, np.ndarray):
            images = [images]
        resized_images = []
        preprocess_params = []
        for i in range(len(images)):
            image, scale, pad = self._resize(images[i])
        resized_images.append(image)
        preprocess_params.append({
            'scale': scale,
            'pad': pad,
            'ori_shape': images[i].shape[:2]
        })
        return resized_images, preprocess_params


class DefaultPreprocess(BasePreprocess):
    """
    resize with fixed short side
    """
    def __init__(self, side_range=(640, 1194)):
        self.side_range = side_range

    def _resize(self, image: np.ndarray):
        h, w = image.shape[:2]
        # todo 简化
        scale = self.side_range[0] / np.minimum(h, w)
        if np.maximum(h, w) * scale > self.side_range[1]:
            scale = self.side_range[1] / np.maximum(h, w)
        resize_w = w
        resize_h = h
        resize_h = int(resize_h * scale)
        resize_w = int(resize_w * scale)
        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
        resize_h = max(32, resize_h)
        resize_w = max(32, resize_w)
        image, scale, pad = resize_and_pad(image, (resize_h, resize_w), pad_mode='none')
        return image, scale, pad

    def __call__(self, images: np.ndarray):
        if isinstance(images, np.ndarray):
            images = [images]
        resized_images = []
        preprocess_params = []
        for i in range(len(images)):
            image, scale, pad = self._resize(images[i])
        resized_images.append(image)
        preprocess_params.append({
            'scale': scale,
            'pad': pad,
            'ori_shape': images[i].shape[:2]
        })
        return resized_images, preprocess_params


class FixedShapePreprocess(BasePreprocess):
    """
    resize with fixed shape
    """
    def __init__(self, target_shape=(864, 640)):
        self.target_shape = target_shape

    def resize(self, image: np.ndarray):
        image, scale, pad = resize_and_pad(image, self.target_shape, pad_mode='center')
        return image, scale, pad

    def __call__(self, images: np.ndarray):
        if isinstance(images, np.ndarray):
            images = [images]
        resized_images = []
        preprocess_params = []
        for i in range(len(images)):
            image, scale, pad = self.resize(images[i])
            resized_images.append(image)
            preprocess_params.append({
                'scale': scale,
                'pad': pad,
                'ori_shape': images[i].shape[:2]
            })
        return resized_images, preprocess_params