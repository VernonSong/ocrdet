# -*- coding: utf-8 -*-
# @Time : 2020/10/23 5:25 下午
# @Author : SongWeinan
# @Software: PyCharm
# 醉后不知天在水，满船清梦压星河。
# ======================================================================================================================
import unittest
import torch
from modeling import Model
from modeling.criterions.pse import PSECriterion
from config import config as cfg


class PSETestCase(unittest.TestCase):
    def test_pse_stride4_forward(self):
        cfg.merge_from_file('../../experiments/base_pse.yml')
        cfg.MODEL.PSE.STRIDE = 4
        model = Model(cfg, is_training=False)
        x = torch.ones([1, 3, 512, 512])
        x = model(x)
        self.assertEqual(x[0].shape, torch.Size([1, 6, 128, 128]))

    def test_pse_stride1_forward(self):
        cfg.merge_from_file('../../experiments/base_pse.yml')
        cfg.MODEL.PSE.STRIDE = 1
        model = Model(cfg, is_training=False)
        x = torch.ones([1, 3, 512, 512])
        x = model(x)
        self.assertEqual(x[0].shape, torch.Size([1, 6, 512, 512]))

    def test_pse_backward(self):
        cfg.merge_from_file('../../experiments/base_pse.yml')
        cfg.MODEL.PSE.STRIDE = 1
        gt_pse_map = torch.randint(0, 2, (4, 6, 512, 512))
        predicted = {
            'pse_map': gt_pse_map.float(),
        }
        target = {
            'pse_map': gt_pse_map,
            'training_mask': torch.ones((4, 1, 512, 512)),
        }

        pse_criterion = PSECriterion(cfg)
        pse_loss, losses, metries = pse_criterion(predicted, target)
        self.assertEqual(pse_loss, 0)
        self.assertEqual(metries['pse_iou'], 1)


if __name__ == '__main__':
    unittest.main()
