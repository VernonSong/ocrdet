# -*- coding: utf-8 -*-
# @Time : 2020/8/12 5:14 下午
# @Author : SongWeinan
# @Software: PyCharm
# 醉后不知天在水，满船清梦压星河。
# ======================================================================================================================
import unittest
import torch
from modeling.meta_archs.fpn import FPN
from config import config as cfg


class FPNTestCase(unittest.TestCase):
    cfg.MODEL.BACKBONE.NAME = 'resnet50'
    def test_fpn(self):
        net = FPN(cfg)
        x = torch.ones([1, 3, 512, 512])
        x = net(x)
        self.assertEqual(x['p2'].shape, torch.Size([1, 256, 128, 128]))
        self.assertEqual(x['p3'].shape, torch.Size([1, 256, 64, 64]))
        self.assertEqual(x['p4'].shape, torch.Size([1, 256, 32, 32]))
        self.assertEqual(x['p5'].shape, torch.Size([1, 256, 16, 16]))

    def test_fpn_p6p7(self):
        cfg.MODEL.META_ARCH.USE_P6P7 = True
        net = FPN(cfg)
        x = torch.ones([1, 3, 512, 512])
        x = net(x)
        self.assertEqual(x['p2'].shape, torch.Size([1, 256, 128, 128]))
        self.assertEqual(x['p3'].shape, torch.Size([1, 256, 64, 64]))
        self.assertEqual(x['p4'].shape, torch.Size([1, 256, 32, 32]))
        self.assertEqual(x['p5'].shape, torch.Size([1, 256, 16, 16]))
        self.assertEqual(x['p6'].shape, torch.Size([1, 256, 8, 8]))
        self.assertEqual(x['p7'].shape, torch.Size([1, 256, 4, 4]))

    def test_fpn_fuse(self):
        cfg.MODEL.META_ARCH.USE_FUSE = True
        net = FPN(cfg)
        x = torch.ones([1, 3, 512, 512])
        x = net(x)
        self.assertEqual(x['p2'].shape, torch.Size([1, 256, 128, 128]))
        self.assertEqual(x['p3'].shape, torch.Size([1, 256, 64, 64]))
        self.assertEqual(x['p4'].shape, torch.Size([1, 256, 32, 32]))
        self.assertEqual(x['p5'].shape, torch.Size([1, 256, 16, 16]))
        self.assertEqual(x['fuse'].shape, torch.Size([1, 1024, 128, 128]))


if __name__ == '__main__':
    unittest.main()
