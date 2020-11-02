# -*- coding: utf-8 -*-
# @Time : 2020/8/25 12:09 下午
# @Author : SongWeinan
# @Software: PyCharm
# 醉后不知天在水，满船清梦压星河。
# ======================================================================================================================
import unittest
import numpy as np
import torch
from modeling import Model
from modeling.criterions.east import EASTCriterion
from config import config as cfg


class MyTestCase(unittest.TestCase):
    def test_east_stride4_forward(self):
        cfg.merge_from_file('../../experiments/base_east.yml')
        cfg.MODEL.EAST.STRIDE = 4
        model = Model(cfg, is_training=False)
        x = torch.ones([1, 3, 512, 512])
        x = model(x)
        self.assertEqual(x[0].shape, torch.Size([1, 1, 128, 128]))
        self.assertEqual(x[1].shape, torch.Size([1, 4, 128, 128]))
        self.assertEqual(x[2].shape, torch.Size([1, 128, 128]))

    def test_east_stride1_forward(self):
        cfg.merge_from_file('../../experiments/base_east.yml')
        cfg.MODEL.EAST.STRIDE = 1
        model = Model(cfg, is_training=False)
        x = torch.ones([1, 3, 512, 512])
        x = model(x)
        self.assertEqual(x[0].shape, torch.Size([1, 1, 512, 512]))
        self.assertEqual(x[1].shape, torch.Size([1, 4, 512, 512]))
        self.assertEqual(x[2].shape, torch.Size([1, 512, 512]))

    def test_east_backward(self):
        cfg.merge_from_file('../../experiments/base_east.yml')
        cfg.MODEL.DB.STRIDE = 1
        gt_score_map = torch.randint(0, 2, (4, 1, 512, 512))
        gt_distance_map = torch.Tensor(np.random.random_sample((4, 4, 512, 512)))
        gt_rotation_map = torch.Tensor(np.random.random_sample((4, 1, 512, 512)))
        predicted = {
            'east_score_map': gt_score_map.float(),
            'east_distance_map': gt_distance_map,
            'east_rotation_map': gt_rotation_map
        }
        target = {
            'east_score_map': gt_score_map.float(),
            'east_distance_map': gt_distance_map,
            'east_rotation_map': gt_rotation_map,
            'training_mask': torch.ones_like(gt_score_map),
        }

        east_criterion = EASTCriterion(cfg)
        east_loss, losses, metries = east_criterion(predicted, target)
        self.assertEqual(east_loss, 0)
        self.assertEqual(metries['east_score_iou'], 1)
        self.assertEqual(metries['east_distance_iou'], 1)


if __name__ == '__main__':
    unittest.main()
