# -*- coding: utf-8 -*-
# @Time : 2020/10/23 5:29 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch.nn.functional as F
from modeling.criterions.build import CRITERION_REGISTRY
from modeling.layers.losses import *
from modeling.layers.metries import *


@CRITERION_REGISTRY.register()
class PSECriterion(nn.Module):
    def __init__(self, cfg):
        super(PSECriterion, self).__init__()
        self.dce_loss = DiceCoefficientLoss()
        self.complete_weight = cfg.MODEL.PSE.COMPLETE_WEIGHT
        self.shrunk_weight = cfg.MODEL.PSE.SHRUNK_WEIGHT

    def forward(self, predicted, target):
        pred_pse_map = predicted['pse_map']
        device = pred_pse_map.device
        gt_pse_map = target['pse_map'].to(device).float()
        training_mask = target['training_mask'].to(device).float()
        training_mask = F.interpolate(training_mask, pred_pse_map.shape[-2:], mode='nearest').bool()
        pred_complete_text_map = pred_pse_map[:, -1]
        gt_complete_text_map = gt_pse_map[:, -1]
        pred_shrunk_text_map = pred_pse_map[:, :-1]
        gt_shrunk_text_map = gt_pse_map[:, :-1]
        B, C, H, W = gt_pse_map.shape
        in_text_mask = gt_complete_text_map.view(B, 1, H, W).repeat(1, C - 1, 1, 1)
        complete_loss, complete_iou = self.dce_loss(pred_complete_text_map, gt_complete_text_map, training_mask)
        shrunk_loss, shrunk_iou = self.dce_loss(pred_shrunk_text_map, gt_shrunk_text_map, in_text_mask)
        pse_loss = (complete_loss * self.complete_weight) + (shrunk_loss * self.shrunk_weight)
        pse_iou = (complete_iou + shrunk_iou) / 2.
        losses = {
            'pse_loss': pse_loss,
            'pse_complete_loss': complete_loss,
            'pse_shrunk_loss': shrunk_loss
        }

        metries = {
            'pse_pred_pse_map': torch.mean(pred_pse_map, 1, keepdim=True),
            'pse_gt_pse_map': torch.mean(gt_pse_map, 1, keepdim=True),
            'pse_iou': pse_iou
        }
        return pse_loss, losses, metries