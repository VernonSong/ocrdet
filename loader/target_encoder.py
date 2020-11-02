# -*- coding: utf-8 -*-
# @Time : 2020/7/31 3:04 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from torch.utils.data.dataset import Dataset
from loader.encoders.build import build_encoder


class TargetEncoder(Dataset):
    """加载encoder"""
    def __init__(self,
                 dataset: Dataset,
                 cfg):
        self._dataset = dataset
        encoders = cfg.MODEL.CODERS
        self.encoders = [build_encoder('TrainingMaskEncoder', cfg)]
        for encoder in encoders:
            self.encoders.append(build_encoder(encoder + 'Encoder', cfg))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item: int):
        data = self._dataset[item]
        image = data['image']
        polygons = data['polygons']
        if (image.shape[0] % 32) != 0 or (image.shape[1] % 32) != 0:
            raise ValueError('image shape must be an integer multiple of 32')
        for encoder in self.encoders:
            encoder.init_encoder()
        num_polygons = len(polygons)
        for idx in range(num_polygons):
            polygon = polygons.get(idx)
            for encoder in self.encoders:
                encoder.calculate_encoder(polygon)

        targets = {
            'image': image.transpose((2, 0, 1)),
        }

        for encoder in self.encoders:
            targets.update(encoder.create_target())
        return targets
