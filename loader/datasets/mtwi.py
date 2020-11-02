# -*- coding: utf-8 -*-
# @Time : 2020/8/3 5:09 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import os
from loader.datasets.build import DATASET_REGISTRY
from loader.datasets.icdar2015 import ICDAR2015


@DATASET_REGISTRY.register()
class MTWI(ICDAR2015):
    def _read_files(self) -> list:
        images_dir = os.path.join(self.dataset_dir, 'image_train')
        labels_dir = os.path.join(self.dataset_dir, 'txt_train')
        images_list = os.listdir(images_dir)
        labels_list = os.listdir(labels_dir)
        files_list = []
        for image in images_list:
            id = os.path.splitext(image)[0]
            label = id + '.txt'
            if not (label in labels_list):
                label = None
            files_list.append({
                'id': id,
                'image': os.path.join(images_dir, image),
                'label': os.path.join(labels_dir, label),
            })
        return files_list
