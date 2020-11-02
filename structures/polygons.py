# -*- coding: utf-8 -*-
# @Time : 2020/7/30 4:42 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import copy
from typing import List, Tuple, Union
import numpy as np
import cv2


class Polygons(object):
    def __init__(self, contours: List[np.ndarray]):
        self._contours = []
        for contour in contours:
            shape = contour.shape
            if len(shape) != 2 or shape[0] < 3 or shape[1] != 2:
                raise ValueError('Polygons have wrong shape')
            self._contours.append(contour.astype(np.float))
        num_points = [len(contour) for contour in contours]
        self._fields = {}
        self.add_field('num_points', num_points)

    def __len__(self) -> int:
        return len(self._contours)

    def has_field(self, key: str) -> bool:
        return key in self._fields

    def _set_field(self, key: str, values: Union[list, np.ndarray]):
        # 检查字段数目与polygon数目是否相等
        if len(values) != len(self):
            raise ValueError('Invalid dimensions for field data')
        if len(values) > 1:
            # 检查字段是否为同一类型
            value_type = type(values[0]).__name__
            for value in values[1:]:
                if type(value).__name__ != value_type:
                    raise TypeError('Different type in field data')
        self._fields[key] = values

    def set_field(self, key: str, values: Union[list, np.ndarray]):
        # 检查是否存在该字段
        if not self.has_field(key):
            raise ValueError('Field ' + key + 'is not exists')
        self._set_field(key, values)

    def add_field(self, key: str, values: Union[list, np.ndarray]):
        # 检查是否存在该字段
        if self.has_field(key):
            raise ValueError('Field ' + key + 'already exists')
        self._set_field(key, values)

    def get_field(self, key: str) -> Union[list, np.ndarray]:
        if not self.has_field(key):
            raise ValueError('Field ' + key + 'not exists')
        # deep copy
        return copy.deepcopy(self._fields[key])

    def get_contours(self):
        return copy.deepcopy(self._contours)

    def set_contours(self, contours: List[np.ndarray]):
        if len(contours) != len(self):
            raise ValueError('Invalid dimensions for contours')
        _contours = []
        for contour in contours:
            shape = contour.shape
            if len(shape) != 2 or shape[0] < 3 or shape[1] != 2:
                raise ValueError('Polygons have wrong shape')
            _contours.append(contour.astype(np.float))
        self._contours = _contours

    def copy(self):
        p = Polygons(copy.deepcopy(self._contours))
        for field in self._fields:
            p._set_field(field, self.get_field(field))
        return p

    def scale_and_pad(self,
                      scale: Union[List[Union[float, int]], Tuple[Union[float, int], ...], np.ndarray, float, int],
                      pad: Union[List[int], Tuple[int, ...], np.ndarray],
                      recover=False):
        if len(self) == 0:
            return
        points = np.concatenate(self._contours, 0).reshape(-1, 2)
        if recover:
            points[:, 0] += pad[0]
            points[:, 1] += pad[1]
        if isinstance(scale, int) or isinstance(scale, float):
            points *= scale
        else:
            points[:, 0] *= scale[0]
            points[:, 1] *= scale[1]

        if not recover:
            points[:, 0] += pad[0]
            points[:, 1] += pad[1]
        num_points = self._fields['num_points']
        contours = np.split(points, np.cumsum(num_points))[:-1]
        self._contours = contours

    def get_window(self):
        points = []
        # 计算training polyon的window
        for i, t in enumerate(self._fields['training_tag']):
            if t:
                points.append(self._contours[i])
        if len(points) == 0:
            return [10000, 10000, -1, -1]
        points = np.concatenate(points, 0).reshape(-1, 2)
        xmin = points[:, 0].min()
        ymin = points[:, 1].min()
        xmax = points[:, 0].max() + 1
        ymax = points[:, 1].max() + 1
        return [xmin, ymin, xmax, ymax]

    def fliter_out(self, window: np.ndarray):
        num = len(self)
        if not self.has_field('training_tag'):
            self.add_field('training_tag', np.ones([num]).astype(np.bool))
        if num == 0:
            return
        if not self.has_field('valid'):
            self.add_field('valid', np.ones([num]).astype(np.bool))
        training_tags = self._fields['training_tag']
        num_points = self._fields['num_points']
        points = np.concatenate(self._contours, 0).reshape(-1, 2)

        xmin_w = window[0, 0]
        ymin_w = window[0, 1]
        xmax_w = window[1, 0]
        ymax_w = window[1, 1]

        valid_xmin = (points[:, 0] - xmin_w) >= 0
        valid_ymin = (points[:, 1] - ymin_w) >= 0
        valid_xmax = (points[:, 0] - xmax_w) < 0
        valid_ymax = (points[:, 1] - ymax_w) < 0

        valid = np.stack([valid_xmin, valid_ymin, valid_xmax, valid_ymax], -1)
        valid = np.split(valid, np.cumsum(num_points))[:-1]
        for idx, valid_per_polygon in enumerate(valid):
            valid_per_polygon = np.min(valid_per_polygon)
            if not valid_per_polygon:
                training_tags[idx] = False
        self._set_field('training_tag', training_tags)

    def fliter_small(self, min_area=5):
        num = len(self)
        if not self.has_field('training_tag'):
            self.add_field('training_tag', np.ones([num]).astype(np.bool))
        if num == 0:
            return
        training_tags = self._fields['training_tag']
        for idx in range(num):
            contour = self._contours[idx].astype(np.int32)
            if cv2.contourArea(contour) < min_area:
                training_tags[idx] = False

    def get(self, idx: int) -> dict:
        if idx >= len(self):
            raise ValueError('Index exceeds the range of polygons')
        item_dict = {}
        item_dict['contour'] = self._contours[idx].copy()
        item_dict['fields'] = {}
        for key in self._fields:
            item_dict['fields'][key] = copy.deepcopy(self._fields[key][idx])
        return item_dict

    def get_keys(self):
        keys = []
        for key in self._fields:
            keys.append(key)
        return keys

    @staticmethod
    def cat(polygons_list):
        if len(polygons_list) == 0:
            raise ValueError('length of polygons list is 0')
        if len(polygons_list) == 1:
            return polygons_list[0]
        keys = polygons_list[0].get_keys()
        _dict = {}
        _contours = polygons_list[0].get_data()
        for key in keys:
            _dict[key] = polygons_list[0].get_field(key)

        for polygons in polygons_list[1:]:
            if keys != polygons.get_keys():
                raise ValueError('polygons have diffients keys')

            for key in keys:
                value = polygons.get_field(key)
                if isinstance(value, list):
                    _dict[key] += value
                elif isinstance(value, np.ndarray):
                    _dict[key] = np.concatenate([_dict[key], value])
            _contours += polygons.get_data()

        p = Polygons(_contours.copy())
        for key in _dict:
            if key != 'num_points':
                p.add_field(key, _dict[key].copy())
        return p

    def delete(self, idxs: Union[list, np.ndarray]):
        idxs = np.array(idxs)
        if len(idxs.shape) != 1:
            raise ValueError('wrong idxs shape')
        idxs = np.unique(idxs)[::-1]
        print(idxs)
        for idx in idxs:
            del(self._contours[idx])
            for key in self._fields:
                if isinstance(self._fields[key], np.ndarray):
                    self._fields[key] = np.delete(self._fields[key], idx)
                else:
                    del(self._fields[key][idx])










