# -*- coding: utf-8 -*-
# @Time : 2020/8/18 3:33 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import os
from loader.datasets.base_dataset import BaseDataset
from loader.datasets.build import DATASET_REGISTRY
from structures.polygons import Polygons
from utils.io_util import imread
import re
import json


@DATASET_REGISTRY.register()
class LabelMe(BaseDataset):
    """
    LabelMe标注格式数据集
    """
    def __init__(self,
                 cfg):
        super(LabelMe, self).__init__(cfg)

    def _read_files(self):
        images_dir = os.path.join(self.dataset_dir, 'images')
        labels_dir = os.path.join(self.dataset_dir, 'labels')
        images_list = os.listdir(images_dir)
        labels_list = os.listdir(labels_dir)
        files_list = []
        for image in images_list:
            id = os.path.splitext(image)[0]

            label = id + '.json'
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

        return files_list

    def _load_label(self, label_path: str):
        contours = []
        training_tags = []
        with open(label_path, encoding='UTF-8-sig') as f:
            js = json.loads(f.read())
            for polygon in js['shapes']:
                category_id = polygon['label']
                contour = np.array(polygon['points']).reshape(-1, 2)
                training_tags.append(True if int(category_id) == 1 else False)
                contours.append(contour)
        return {'contour': contours,
                'training_tag': np.array(training_tags)}

    def __getitem__(self, idx):
        file_dict = self.files_list[idx]
        image = imread(file_dict['image'])
        labels = self._load_label(file_dict['label'])
        polygons = Polygons(labels['contour'])
        polygons.add_field('training_tag', labels['training_tag'])

        return {
            'id': file_dict['id'],
            'image': image,
            'polygons': polygons
        }

