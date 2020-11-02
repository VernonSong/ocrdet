# -*- coding: utf-8 -*-
# @Time : 2020/8/15 2:51 下午
# @Author : SongWeinan
# @Software: PyCharm
# 醉后不知天在水，满船清梦压星河。
# ======================================================================================================================
import unittest
import torch
from modeling import Model
from modeling.criterions.db import DBCriterion
from config import config as cfg


class DBTestCase(unittest.TestCase):
    def test_db_stride4_forward(self):
        cfg.merge_from_file('../../experiments/base_db.yml')
        cfg.MODEL.DB.STRIDE = 4
        model = Model(cfg, is_training=False)
        x = torch.ones([1, 3, 512, 512])
        x = model(x)
        self.assertEqual(x[0].shape, torch.Size([1, 1, 128, 128]))

    def test_db_stride1_forward(self):
        cfg.merge_from_file('../../experiments/base_db.yml')
        cfg.MODEL.DB.STRIDE = 1
        model = Model(cfg, is_training=False)
        x = torch.ones([1, 3, 512, 512])
        x = model(x)
        self.assertEqual(x[0].shape, torch.Size([1, 1, 512, 512]))

    def test_db_backward(self):
        cfg.merge_from_file('../../experiments/base_db.yml')
        cfg.MODEL.DB.STRIDE = 1
        gt_binary_map = torch.randint(0, 2, (4, 1, 512, 512))
        gt_threshold_map = torch.rand_like(gt_binary_map, dtype=torch.float)
        predicted = {
            'db_binary_map': gt_binary_map.float(),
            'db_probability_map': gt_binary_map.float(),
            'db_threshold_map': gt_threshold_map
        }
        target = {
            'db_probability_map': gt_binary_map.float(),
            'db_threshold_map': gt_threshold_map,
            'training_mask': torch.ones_like(gt_binary_map),
            'db_threshold_mask': torch.ones_like(gt_binary_map),
        }

        db_criterion = DBCriterion(cfg)
        db_loss, losses, metries = db_criterion(predicted, target)
        self.assertEqual(db_loss, 0)
        self.assertEqual(metries['db_iou'], 1)


if __name__ == '__main__':
    unittest.main()
