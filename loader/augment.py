# -*- coding: utf-8 -*-
# @Time : 2020/8/2 4:06 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import os
from typing import Tuple
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
import imgaug.augmenters as iaa
from utils.io_util import imread

# np.random.bit_generator = np.random._bit_generator


class Augment(Dataset):
    """数据增强模块"""
    def __init__(self, dataset: Dataset, cfg):
        self._dataset = dataset
        self.input_shape = cfg.AUGMENT.INPUT_SHAPE
        self.zoom_in = cfg.AUGMENT.ZOOM_IN
        self.min_scale = cfg.AUGMENT.MIN_SCALE
        self.max_scale = cfg.AUGMENT.MAX_SCALE
        self.max_try_times = cfg.AUGMENT.MAX_TRY_TIMES
        self.flip = cfg.AUGMENT.FLIP
        self.aspect_ratio = cfg.AUGMENT.ASPECT_RATIO
        self.translate_percent = cfg.AUGMENT.TRANSLATE_PRESENT
        self.rotate = cfg.AUGMENT.ROTATE
        self.shear = cfg.AUGMENT.SHEAR
        self.perspective_transform = cfg.AUGMENT.PERSPECTIVE_TRANSFORM
        self.brightness = cfg.AUGMENT.BRIGHTNESS
        self.hue = cfg.AUGMENT.HUE
        self.saturation = cfg.AUGMENT.SATURATION
        self.augment_background = cfg.AUGMENT.BACKGROUND
        if self.augment_background:
            self.backgrounds = [os.path.join('../../data/background', item) for item in os.listdir(
                '../../data/background')]

        self.seq = iaa.Sequential(
            [
                iaa.Fliplr(self.flip),
                iaa.Affine(
                    scale={"x": self.aspect_ratio, "y": self.aspect_ratio},
                    translate_percent={"x": self.translate_percent,
                                       "y": self.translate_percent},
                    rotate=self.rotate,
                    shear=self.shear,
                    order=[0, 1],
                    cval=(0, 255)
                ),
                iaa.PerspectiveTransform(
                    scale=self.perspective_transform
                ),
                iaa.MultiplyBrightness(self.brightness),
                iaa.MultiplySaturation(self.saturation),
                iaa.MultiplyHue(self.hue)])

    def __len__(self):
        return len(self._dataset)

    def _random_crop(self, scaled_w: int, scaled_h: int) -> Tuple[int, int]:
        offset_x = [scaled_w - self.input_shape[1], 0] if \
            scaled_w - self.input_shape[1] < 0 else [0, scaled_w - self.input_shape[1] + 1]
        offset_y = [scaled_h - self.input_shape[0], 0] if \
            scaled_h - self.input_shape[0] < 0 else [0, scaled_h - self.input_shape[0] + 1]
        offset_x = np.random.randint(offset_x[0], offset_x[1])
        offset_y = np.random.randint(offset_y[0], offset_y[1])
        return offset_x, offset_y

    def _resize_image(self, image: np.ndarray, scaled_w: int, scaled_h: int,
                      offset_x: int, offset_y: int) -> np.ndarray:
        image = cv2.resize(image, dsize=(scaled_w, scaled_h))
        # 随机选取一张无文字图作为背景
        if self.augment_background:
            s = np.random.randint(0, len(self.backgrounds))
            backgrounds = imread(self.backgrounds[s])
            image_ = cv2.resize(backgrounds, (int(self.input_shape[1] * 1.5), int(self.input_shape[0] * 1.5)))
        else:
            image_ = np.zeros([self.input_shape[0] * 2, self.input_shape[1] * 2, 3]).astype(np.uint8)
        pad_h = np.maximum(0, -offset_y)
        pad_w = np.maximum(0, -offset_x)
        crop_h = np.maximum(0, offset_y)
        crop_w = np.maximum(0, offset_x)
        h = np.minimum(self.input_shape[0], image.shape[0])
        w = np.minimum(self.input_shape[1], image.shape[1])
        image_[pad_h + int(self.input_shape[0] * 0.25):pad_h + h + int(self.input_shape[0] * 0.25),
        pad_w + int(self.input_shape[1] * 0.25):pad_w + w + int(self.input_shape[1] * 0.25)] = image[crop_h:crop_h + h,
                                                                                               crop_w:crop_w + w]
        return image_

    def _random_scale_and_crop(self, image: np.ndarray, polygons: 'Polygons') -> Tuple[np.ndarray, 'Polygons']:
        background = np.random.choice([True, False], p=[0.1, 0.9])
        zoom_in = np.random.choice([True, False], p=np.array([self.zoom_in, 1. - self.zoom_in]))
        if zoom_in:
            min_scale = 1
            max_scale = self.max_scale
        else:
            min_scale = self.min_scale
            max_scale = 1
        for i in range(self.max_try_times):
            p_ = polygons.copy()
            h, w = image.shape[:2]
            pre_scale = np.maximum(self.input_shape[0] / image.shape[0], self.input_shape[1] / image.shape[1])
            scale = np.random.random_integers(min_scale * 10, max_scale * 10) / 10
            scale = pre_scale * scale
            scaled_w = int(np.around(w * scale))
            scaled_h = int(np.around(h * scale))
            offset_x, offset_y = self._random_crop(scaled_w, scaled_h)
            p_.scale_and_pad(scale, [-offset_x, -offset_y])
            p_.fliter_out(np.array([[0, 0], [self.input_shape[1], self.input_shape[0]]]))
            training_tags = p_.get_field('training_tag')
            if len(training_tags) == 0 or np.max(training_tags) or background:
                # pre pad，防止之后旋转等增强导致instance到图片外
                image = self._resize_image(image, scaled_w, scaled_h, offset_x, offset_y)
                p_.scale_and_pad(1, [self.input_shape[1] * 0.25, self.input_shape[0] * 0.25])
                return image, p_

        scale = np.minimum(self.input_shape[0] / image.shape[0], self.input_shape[1] / image.shape[1])
        h, w = image.shape[:2]
        scaled_w = int(np.around(w * scale))
        scaled_h = int(np.around(h * scale))
        offset_x, offset_y = self._random_crop(scaled_w, scaled_h)
        polygons.scale_and_pad(scale, [-offset_x, -offset_y])
        polygons.fliter_out(np.array([[0, 0], [self.input_shape[1], self.input_shape[0]]]))
        image = self._resize_image(image, scaled_w, scaled_h, offset_x, offset_y)
        polygons.scale_and_pad(1, [self.input_shape[1] * 0.25, self.input_shape[0] * 0.25])
        return image, polygons

    def post_crop(self, image, polygons):
        polygons_window = polygons.get_window()
        image_window = [int(self.input_shape[1] * 0.25), int(self.input_shape[0] * 0.25),
                        int(self.input_shape[1] * 1.25), int(self.input_shape[0] * 1.25)]
        xmin = np.maximum(polygons_window[0], 0) if polygons_window[0] < image_window[0] else image_window[0]
        ymin = np.maximum(polygons_window[1], 0) if polygons_window[1] < image_window[1] else image_window[1]
        xmax = np.minimum(polygons_window[2], image.shape[1]) if polygons_window[2] > image_window[2] else image_window[
            2]
        ymax = np.minimum(polygons_window[3], image.shape[0]) if polygons_window[3] > image_window[3] else image_window[
            3]
        image = cv2.resize(image[int(ymin): int(ymax), int(xmin): int(xmax)],
                           (self.input_shape[1], self.input_shape[0]))
        polygons.scale_and_pad((self.input_shape[1] / (xmax - xmin), self.input_shape[0] / (ymax - ymin)),
                               [-xmin, -ymin], recover=True)
        return image, polygons

    def __getitem__(self, idx):
        item = self._dataset[idx]
        image = item['image']
        polygons = item['polygons']
        num_points = polygons.get_field('num_points')
        image, polygons = self._random_scale_and_crop(image, polygons)

        if len(polygons) == 0:
            image = self.seq(
                images=np.expand_dims(image, 0))[0]
            image, polygons = self.post_crop(image, polygons)
        else:
            total_points = np.concatenate(polygons.get_contours())
            image, total_points = self.seq(
                image=image, keypoints=[total_points])
            total_points = total_points[0]
            contours = np.split(total_points, np.cumsum(num_points))[:-1]
            polygons.set_contours(contours)

            image, polygons = self.post_crop(image, polygons)
            window = np.array([[0, 0], [image.shape[1], image.shape[0]]])
            polygons.fliter_out(window)

        polygons.fliter_small(10)

        return {
            'image': image,
            'polygons': polygons
        }



