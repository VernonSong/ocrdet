# -*- coding: utf-8 -*-
# @Time : 2020/11/2 6:03 下午
# @Author : SongWeinan
# @Software: PyCharm
# 醉后不知天在水，满船清梦压星河。
# ======================================================================================================================
import unittest
from modeling.layers.losses import *


class MyTestCase(unittest.TestCase):
    def test_balance_dicecoefficient_loss(self):
        predicted = torch.Tensor([[0.2, 1], [0.8, 0], [0.5, 0], [0, 0.3]])
        target = torch.Tensor([[0, 1], [0, 0], [1, 0], [0, 0]])
        training_mask = torch.Tensor([[1, 0], [1, 1], [1, 1], [1, 1]])
        loss_func = BalanceDiceCoefficientLoss()
        loss, iou = loss_func(predicted, target, training_mask)
        torch.testing.assert_allclose(loss, 0.6429)


if __name__ == '__main__':
    unittest.main()
