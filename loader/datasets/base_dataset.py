# -*- coding: utf-8 -*-
# @Time : 2020/7/28 3:42 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from abc import ABCMeta, abstractmethod
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self,
                 dataset_dir):
        super(BaseDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.files_list = self._read_files()

    @abstractmethod
    def _read_files(self) -> list:
        pass

    @abstractmethod
    def _load_label(self, label_path: str) -> dict:
        pass

    def __len__(self) -> int:
        return len(self.files_list)