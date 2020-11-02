# -*- coding: utf-8 -*-
# @Time : 2020/10/25 3:51 下午
# @Author : SongWeinan
# @Software: PyCharm
# 醉后不知天在水，满船清梦压星河。
# ======================================================================================================================
import unittest
import torch
from modeling.backbones.hrnet import hrnet_w48, hrnet_w32, hrnet_w18_small



class HRnetTestCase(unittest.TestCase):

    def test_hrnet_w48(self):
        net = hrnet_w48(pretrained=True, norm_layer=torch.nn.BatchNorm2d)
        x = torch.ones([1, 3, 512, 512])
        x = net(x)
        self.assertEqual(x['p2'].shape, torch.Size([1, 48, 128, 128]))
        self.assertEqual(x['p3'].shape, torch.Size([1, 96, 64, 64]))
        self.assertEqual(x['p4'].shape, torch.Size([1, 256, 32, 32]))
        self.assertEqual(x['p5'].shape, torch.Size([1, 256, 16, 16]))

    # def test_fpn_p6p7(self):
    #     cfg.MODEL.META_ARCH.USE_P6P7 = True
    #     net = FPN(cfg)
    #     x = torch.ones([1, 3, 512, 512])
    #     x = net(x)
    #     self.assertEqual(x['p2'].shape, torch.Size([1, 256, 128, 128]))
    #     self.assertEqual(x['p3'].shape, torch.Size([1, 256, 64, 64]))
    #     self.assertEqual(x['p4'].shape, torch.Size([1, 256, 32, 32]))
    #     self.assertEqual(x['p5'].shape, torch.Size([1, 256, 16, 16]))
    #     self.assertEqual(x['p6'].shape, torch.Size([1, 256, 8, 8]))
    #     self.assertEqual(x['p7'].shape, torch.Size([1, 256, 4, 4]))


if __name__ == '__main__':
    unittest.main()
