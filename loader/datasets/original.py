# -*- coding: utf-8 -*-
# @Time : 2020/10/30 10:11 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import os
from loader.datasets.base_dataset import BaseDataset
from loader.datasets.build import DATASET_REGISTRY
from utils.io_util import imread


@DATASET_REGISTRY.register()
class Original(BaseDataset):
    """
    无标签数据
    """
    def __init__(self,
                 dataset_dir):
        super(Original, self).__init__(dataset_dir)

    def _read_files(self) -> list:
        files_list = []
        for item in os.listdir(self.dataset_dir):
            if not item.split('.')[-1] in ['jpg', 'png', 'jpeg', 'JPG']:
                continue
            id = os.path.splitext(item)[0]
            files_list.append({
                'id': id,
                'image': os.path.join(self.dataset_dir, item)
            })
        return files_list

    def _load_label(self):
        pass

    def __getitem__(self, idx):
        file_dict = self.files_list[idx]
        image = imread(file_dict['image'])

        return {
            'id': file_dict['id'],
            'image': image
        }
