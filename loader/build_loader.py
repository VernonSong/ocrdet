# -*- coding: utf-8 -*-
# @Time : 2020/8/2 3:40 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import itertools
import torch
import numpy as np
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import Sampler
from loader.datasets import build_dataset
from loader.augment import Augment
from loader.target_encoder import TargetEncoder

__all__ = [
    'build_train_data_loader'
]


class TrainingSampler(Sampler):
    """
    参考detectron2(https://github.com/facebookresearch/detectron2)
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle

    def __len__(self):
        return self._size

    def __iter__(self):

        yield from itertools.islice(self._infinite_indices(), 0, None)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(np.random.randint(2 ** 31))
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


def build_train_data_loader(cfg):
    names = cfg.DATA.NAMES
    dirs = cfg.DATA.DIRS
    dataset = []
    for name, dataset_dir in zip(names, dirs):
        dataset.append(build_dataset(name, dataset_dir))
    # 读取数据集
    dataset = ConcatDataset(dataset)
    # 数据增强
    dataset = Augment(dataset, cfg)
    # encoder
    dataset = TargetEncoder(dataset, cfg)
    sampler = TrainingSampler(len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKS,
        pin_memory=cfg.SOLVER.PIN_MEMORY,
        sampler=sampler)
    return data_loader
