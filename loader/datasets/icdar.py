# -*- coding: utf-8 -*-
# @Time : 2020/7/28 3:44 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import os
import random
from data.datasets.base_dataset import BaseDataset
from data.datasets.build import DATASET_REGISTRY
from structures.polygons import Polygons
from utils.io_util import imread
import re


@DATASET_REGISTRY.register()
class ICDAR(BaseDataset):
    """
    icdar格式数据集读取，在icdar2015中
    """
    def __init__(self,
                 cfg):
        super(ICDAR, self).__init__(cfg)

    def _read_files(self):
        images_dir = os.path.join(self.dataset_dir, 'images')
        labels_dir = os.path.join(self.dataset_dir, 'labels')
        images_list = os.listdir(images_dir)
        labels_list = os.listdir(labels_dir)
        files_list = []
        for image in images_list:
            id = os.path.splitext(image)[0]
            label = 'gt_' + id + '.txt'
            if not (label in labels_list):
                files_list.append({
                    'id': id,
                    'image': os.path.join(images_dir, image),
                    'label': None,
                })
            else:
                files_list.append({
                    'id': id,
                    'image': os.path.join(images_dir, image),
                    'label': os.path.join(labels_dir, label),
                })

        random.shuffle(files_list)
        return files_list

    def _load_label(self, label_path: str):
        polygon_list = []
        training_list = []
        text_list = []
        with open(label_path, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                line = re.sub('[\r\n\t ]', '', line)
                line = line.split(',')
                x0, y0, x1, y1, x2, y2, x3, y3 = list(map(float, line[:8]))
                text = line[-1]
                polygon = np.asarray([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                polygon_list.append(polygon)
                text_list.append(text)
                # todo 去掉Bangla
                if text == '*' or text == '###':
                    training_list.append(False)
                else:
                    training_list.append(True)

        return {'polygon': np.array(polygon_list).reshape(-1, 4, 2),
                'text': np.array(text_list),
                'training_tag': np.array(training_list)}

    def __getitem__(self, idx):
        file_dict = self.files_list[idx]
        image = imread(file_dict['image'])
        labels = self._load_label(file_dict['label'])
        polygons = Polygons(labels['polygon'])
        polygons.add_field('text', labels['text'])
        polygons.add_field('training_tag', labels['training_tag'])

        return {
            'id': file_dict['id'],
            'image': image,
            'polygons': polygons
        }

