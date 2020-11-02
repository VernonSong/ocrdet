# -*- coding: utf-8 -*-
# @Time : 2020/8/15 10:27 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch
import torch.nn as nn


class BalanceDiceCoefficientLoss(nn.Module):
    def __init__(self, negative_ratio=3.):
        super(BalanceDiceCoefficientLoss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, predicted, target, training_mask):
        predicted = predicted.view(-1)
        target = target.view(-1)
        training_mask = training_mask.view(-1)
        positive = target * training_mask
        negative = (1. - target) * training_mask
        positive_num = torch.sum(positive)
        negative_num = torch.min(
            torch.sum(negative), positive_num * self.negative_ratio)
        loss = torch.abs(predicted - target)
        if positive_num == 0:
            loss = torch.mean(loss)
            return loss

        #with torch.no_grad():

        negative_loss = negative * loss
        negative = negative.bool()
        negative_loss, negative_idx = torch.topk(negative_loss[negative], negative_num.int())

        negative_predicted = torch.index_select(predicted[negative], 0, negative_idx)
        negative_target = torch.index_select(target[negative], 0, negative_idx)
        positive_predicted = predicted[positive.bool()]
        positive_target = target[positive.bool()]
        predicted = torch.cat([positive_predicted, negative_predicted])
        target = torch.cat([positive_target, negative_target])

        intersection = torch.sum(predicted * target)
        union = torch.sum((target + predicted) + 1e-10)
        iou = 2. * intersection / union
        loss = 1. - iou
        return loss, iou


class RotationLoss(nn.Module):
    def __init__(self):
        super(RotationLoss, self).__init__()

    def forward(self, predicted, target, positive_index=None):
        target = target.float()
        predicted = predicted.reshape(target.shape)
        loss = 1 - torch.cos(predicted - target)
        loss = loss.reshape(-1)
        if positive_index is not None:
            loss = loss[positive_index.long()]
        loss = torch.mean(loss)
        return loss


class DistanceIouLoss(nn.Module):
    def __init__(self):
        super(DistanceIouLoss, self).__init__()

    def forward(self, predicted, target, positive_index=None, negative_ratio=3.):
        d1_gt, d2_gt, d3_gt, d4_gt = torch.split(target.double(), split_size_or_sections=1, dim=1)
        d1_pred, d2_pred, d3_pred, d4_pred = torch.split(predicted.double(), split_size_or_sections=1, dim=1)
        area_gt = ((d1_gt + d3_gt) * (d2_gt + d4_gt)).float()
        area_pred = ((d1_pred + d3_pred) * (d2_pred + d4_pred)).float()
        w_intersect = (torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)).float()
        h_intersect = (torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)).float()
        area_intersect = w_intersect * h_intersect
        area_union = (area_gt + area_pred - area_intersect)
        iou = (area_intersect + 1.) / (area_union + 1.)
        iou = iou.reshape(-1)
        if positive_index is not None:
            iou = iou[positive_index.long()]
        loss = -torch.log(iou)
        loss = torch.mean(loss)
        return loss, torch.mean(iou)


class BCEloss(nn.Module):
    def __init__(self,
                 negative_ratio=3.,
                 ):
        super(BCEloss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred, target, mask):
        positive = target * mask
        negative = (1. - target)* mask
        positive_num = torch.sum(positive)
        negative_num = torch.min(
            torch.sum(negative), positive_num * self.negative_ratio)
        loss = nn.functional.binary_cross_entropy(pred, target.float(), reduction='none')
        if positive_num == 0:
            loss = torch.mean(loss)
            return loss
        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = negative_loss.reshape([-1])
        negative_loss, _ = torch.topk(negative_loss, negative_num.int())

        balance_loss = (torch.sum(positive_loss) + torch.sum(negative_loss)) / (
                               positive_num + negative_num + 1e-6)
        return balance_loss


class BalanceL1Loss(nn.Module):
    def __init__(self, negative_ratio=3.):
        super(BalanceL1Loss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred: torch.Tensor, gt, mask):
        binaray = (gt > 0.).float()
        positive = binaray * mask
        negative = (1 -binaray) * mask
        positive_num = positive.sum().long()
        negative_num = torch.min(negative.sum(), positive_num * self.negative_ratio)
        loss = torch.abs(pred - gt)
        if positive_num == 0:
            loss = torch.mean(loss)
            return loss
        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = negative_loss.reshape([-1])
        negative_loss, _ = torch.topk(negative_loss, negative_num.long())
        balance_loss = (torch.sum(positive_loss) + torch.sum(negative_loss)) / (
                positive_num + negative_num + 1e-6)
        return balance_loss


class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum
        dif = torch.abs(pred - gt)
        dif = (dif * mask).sum()
        loss = dif / mask_sum
        return loss

