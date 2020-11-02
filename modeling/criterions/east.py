# -*- coding: utf-8 -*-
# @Time : 2020/10/25 12:37 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch.nn.functional as F
from modeling.criterions.build import CRITERION_REGISTRY
from modeling.layers.losses import *


@CRITERION_REGISTRY.register()
class EASTCriterion(nn.Module):
    def __init__(self, cfg):
        super(EASTCriterion, self).__init__()
        self.dce_loss = DiceCoefficientLoss()
        self.distance_loss = DistanceIouLoss()
        self.rotation_loss = RotationLoss()
        self.score_weight = cfg.MODEL.EAST.SCORE_WEIGHT
        self.distance_weight = cfg.MODEL.EAST.DISTANCE_WEIGHT
        self.rotation_weight = cfg.MODEL.EAST.ROTATION_WEIGHT

    def forward(self, predicted, target):
        pred_score_map = predicted['east_score_map']
        pred_distance_map = predicted['east_distance_map']
        pred_ratation_map = predicted['east_rotation_map']
        device = pred_score_map.device
        gt_score_map = target['east_score_map'].to(device)
        gt_distance_map = target['east_distance_map'].to(device)
        gt_rotation_map = target['east_rotation_map'].to(device)
        training_mask = target['training_mask'].to(device).float()
        training_mask = F.interpolate(training_mask, pred_score_map.shape[-2:], mode='nearest').bool()
        score_loss, score_iou = self.dce_loss(pred_score_map, gt_score_map, training_mask)
        distance_loss, distance_iou = self.distance_loss(pred_distance_map, gt_distance_map, gt_score_map)
        rotation_loss = self.rotation_loss(pred_ratation_map, gt_rotation_map, gt_score_map)
        east_loss = (score_loss * self.score_weight) + (
                distance_loss * self.distance_weight) + (rotation_loss * self.rotation_weight)

        losses = {
            'east_loss': east_loss,
            'east_score_loss': score_loss,
            'east_distance_loss': distance_loss,
            'east_rotation_loss': rotation_loss
        }
        metries = {
            'east_pred_score_map': pred_score_map,
            'east_pred_distance_map': pred_distance_map,
            'east_pred_ratation_map': pred_ratation_map,
            'east_gt_score_map': gt_score_map,
            'east_gt_distance_map': gt_distance_map,
            'east_gt_rotation_map': gt_rotation_map,
            'east_score_iou': score_iou,
            'east_distance_iou': distance_iou
        }
        return east_loss, losses, metries

