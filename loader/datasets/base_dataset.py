# -*- coding: utf-8 -*-
# @Time : 2020/7/28 3:42 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from torch.utils import data


class BaseDataset(data.Dataset):
    def __init__(self,
                 cfg):
        super(BaseDataset, self).__init__()

        self.dataset_dir = cfg.DATA.DIR
        self.files_list = self._read_files()

    def _read_files(self) -> list:
        pass

    def _load_label(self, label_path: str) -> dict:
        pass

    def __len__(self) -> int:
        return len(self.files_list)