# -*- coding: utf-8 -*-
# @Time : 2020/8/8 10:25 上午
# @Author : SongWeinan
# @Software: PyCharm
# 醉后不知天在水，满船清梦压星河。
# ======================================================================================================================
import unittest
import torch
from modeling.backbones.resnet import *


class ResNetTestCase(unittest.TestCase):
    def test_resnet50(self):
        net = resnet50(pretrained=True, model_dir='../../pretrained', freeze_at=2)
        x = torch.ones([1, 3, 512, 512])
        x = net(x)
        self.assertEqual(x['c2'].shape, torch.Size([1, 256, 128, 128]))
        self.assertEqual(x['c3'].shape, torch.Size([1, 512, 64, 64]))
        self.assertEqual(x['c4'].shape, torch.Size([1, 1024, 32, 32]))
        self.assertEqual(x['c5'].shape, torch.Size([1, 2048, 16, 16]))

    def test_resnet50_dconv(self):
        net = resnet50_dfonv(pretrained=False, model_dir='None')
        x = torch.ones([1, 3, 512, 512])
        x = net(x)
        self.assertEqual(x['c2'].shape, torch.Size([1, 256, 128, 128]))
        self.assertEqual(x['c3'].shape, torch.Size([1, 512, 64, 64]))
        self.assertEqual(x['c4'].shape, torch.Size([1, 1024, 32, 32]))
        self.assertEqual(x['c5'].shape, torch.Size([1, 2048, 16, 16]))


if __name__ == '__main__':
    unittest.main()
