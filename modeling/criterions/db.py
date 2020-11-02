# -*- coding: utf-8 -*-
# @Time : 2020/8/15 10:27 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch.nn.functional as F
from modeling.criterions.build import CRITERION_REGISTRY
from modeling.layers.losses import *
from modeling.layers.metries import *


@CRITERION_REGISTRY.register()
class DBCriterion(nn.Module):
    def __init__(self, cfg):
        super(DBCriterion, self).__init__()
        self.dcf_loss = BalanceDiceCoefficientLoss()
        self.bce_loss = BCEloss()
        self.l1loss = MaskL1Loss()
        self.binary_weight = cfg.MODEL.DB.BINARY_WEIGHT
        self.probability_weight = cfg.MODEL.DB.PROBABILITY_WEIGHT
        self.shreshold_weight = cfg.MODEL.DB.SHRESHOLD_WEIGHT

    def forward(self, predicted, target):
        pred_probability_map = predicted['db_probability_map']
        pred_threshold_map = predicted['db_threshold_map']
        pred_binary_map = predicted['db_binary_map']
        device = pred_probability_map.device
        gt_probability_map = target['db_probability_map'].to(device)
        gt_threshold_map = target['db_threshold_map'].to(device)
        threshold_mask = target['db_threshold_mask'].to(device)
        training_mask = target['training_mask'].to(device).float()
        training_mask = F.interpolate(training_mask, pred_binary_map.shape[-2:], mode='nearest').bool()
        binary_loss, _ = self.dcf_loss(pred_binary_map, gt_probability_map, training_mask)
        probability_loss = self.bce_loss(pred_probability_map, gt_probability_map, training_mask)
        shreshold_loss = self.l1loss(pred_threshold_map, gt_threshold_map, threshold_mask)
        db_loss = (binary_loss * self.binary_weight) + (
                probability_loss * self.probability_weight) + (shreshold_loss * self.shreshold_weight)

        db_iou = calcu_iou(pred_binary_map, gt_probability_map, training_mask)
        losses = {
            'db_loss': db_loss,
            'db_binary_loss': binary_loss,
            'db_probability_loss': probability_loss,
            'db_shreshold_loss': shreshold_loss
        }
        metries = {
            'db_pred_binary_map': pred_binary_map,
            'db_pred_probability_map': pred_probability_map,
            'db_gt_probability_map': gt_probability_map,
            'db_pred_threshold_map': pred_threshold_map,
            'db_gt_threshold_map': gt_threshold_map,
            'db_iou': db_iou
        }
        return db_loss, losses, metries

