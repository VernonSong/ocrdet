# -*- coding: utf-8 -*-
# @Time : 2020/8/18 6:48 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modeling.criterions.build import CRITERION_REGISTRY
from modeling.layers.losses import *


@CRITERION_REGISTRY.register()
class SemanticCriterion(nn.Module):
    def __init__(self, cfg):
        super(SemanticCriterion, self).__init__()
        self.dce_loss = BalanceDiceCoefficientLoss()
        self.semantic_weight = cfg.MODEL.SEMANTIC.SEMANTIC_WEIGHT

    def forward(self, predicted, target):
        pred_semantic_map = predicted['semantic_map']
        device = pred_semantic_map.device
        gt_semantic_map = target['semantic_map'].to(device)
        training_mask = target['training_mask'].to(device).float()

        print(training_mask.shape)
        training_mask = F.interpolate(training_mask, pred_semantic_map.shape[-2:], mode='nearest').bool()
        semantic_loss_, _ = self.dce_loss(pred_semantic_map, gt_semantic_map, training_mask)
        semantic_loss = semantic_loss_ * self.semantic_weight
        # db_loss = (binary_loss * self.binary_weight) + (
        #         probability_loss * self.probability_weight)
        losses = {
            'semantic_loss_': semantic_loss_,
            'semantic_loss': semantic_loss
        }
        metries = {
            'pred_semantic_map': pred_semantic_map,
            'gt_semantic_map': gt_semantic_map
        }
        return semantic_loss, losses, metries
