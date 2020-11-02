# -*- coding: utf-8 -*-
# @Time : 2020/10/19 2:31 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import os
from loader.datasets.icdar2015 import ICDAR2015
from loader.datasets.build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ICDAR2017(ICDAR2015):
    """
    icdar2017格式数据集
    """
    def __init__(self,
                 dataset_dir):
        super(ICDAR2017, self).__init__(dataset_dir)

    def _read_files(self) -> list:
        images_dir = os.path.join(self.dataset_dir, 'ch8_training_images')
        labels_dir = os.path.join(self.dataset_dir, 'ch8_training_localization_transcription_gt_v2')
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
