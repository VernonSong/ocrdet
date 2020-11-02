# -*- coding: utf-8 -*-
# @Time : 2020/7/28 3:44 下午
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


@DATASET_REGISTRY.register()
class ICDAR2015(BaseDataset):
    """
    icdar2015格式数据集
    """
    def __init__(self,
                 dataset_dir):
        super(ICDAR2015, self).__init__(dataset_dir)

    def _read_files(self) -> list:
        images_dir = os.path.join(self.dataset_dir, 'ch4_training_images')
        labels_dir = os.path.join(self.dataset_dir, 'ch4_training_localization_transcription_gt')
        images_list = os.listdir(images_dir)
        labels_list = os.listdir(labels_dir)
        files_list = []
        for image in images_list:
            id = os.path.splitext(image)[0]
            label = 'gt_' + id + '.txt'
            if not (label in labels_list):
                label = None
            files_list.append({
                'id': id,
                'image': os.path.join(images_dir, image),
                'label': os.path.join(labels_dir, label),
            })
        return files_list

    def _load_label(self, label_path: str):
        contours = []
        training_tags = []
        texts = []
        with open(label_path, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                line = re.sub('[\r\n\t ]', '', line)
                line = line.split(',')
                x0, y0, x1, y1, x2, y2, x3, y3 = list(map(float, line[:8]))
                text = line[-1]
                contour = np.asarray([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                contours.append(contour)
                texts.append(text)
                # todo 去掉部分语种
                if text == '*' or text == '###':
                    training_tags.append(False)
                else:
                    training_tags.append(True)

        return {'contour': contours,
                'text': texts,
                'training_tag': training_tags}

    def __getitem__(self, idx):
        file_dict = self.files_list[idx]
        image = imread(file_dict['image'])
        labels = self._load_label(file_dict['label'])
        polygons = Polygons(labels['contour'])
        polygons.add_field('text', labels['text'])
        polygons.add_field('training_tag', labels['training_tag'])

        return {
            'id': file_dict['id'],
            'image': image,
            'polygons': polygons
        }

