# -*- coding: utf-8 -*-
# @Time : 2020/10/23 4:48 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch


def calcu_iou(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor):
    a = a.float().reshape(mask.shape) * mask
    b = b.reshape(a.shape) * mask
    intersection = torch.sum(a * b)
    union = torch.sum((a + b) + 1e-10)
    return 2. * intersection / union