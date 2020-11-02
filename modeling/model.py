# -*- coding: utf-8 -*-
# @Time : 2020/8/14 5:57 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch
import torch.nn as nn
from modeling.heads import build_head
from modeling.criterions import build_criterion
from modeling.meta_archs import build_meta_arch


class Model(nn.Module):
    def __init__(self, cfg, is_training):
        super(Model, self).__init__()
        self.meta_arch = build_meta_arch(cfg)
        coders = cfg.MODEL.CODERS
        self.device = cfg.SOLVER.DEVICE
        heads = []
        criterions = []
        self.is_training = is_training
        for coder in coders:
            heads.append(build_head(coder + 'Head', cfg, self.is_training))
            if self.is_training:
                criterions.append(build_criterion(coder + 'Criterion', cfg))

        pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).to(
            self.device).view(1, 3, 1, 1).repeat(1, 1, 1, 1)
        pixel_std = torch.Tensor([0.229, 0.224, 0.225]).to(
            self.device).view(1, 3, 1, 1).repeat(1, 1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.heads = nn.ModuleList(heads)
        self.criterions = nn.ModuleList(criterions)

    def forward(self, batch_inputs):
        if self.is_training:
            x = batch_inputs['image'].to(self.device).float()
        else:
            x = batch_inputs.to(self.device).float()
        x = self.normalizer(x/255.)
        x = self.meta_arch(x)
        if self.is_training:
            predicted = {}
            for head in self.heads:
                predicted.update(head(x))
            losses = {}
            metries = {}
            total_loss = 0
            for criterion in self.criterions:
                outputs = criterion(predicted, batch_inputs)
                total_loss += outputs[0]
                losses.update(outputs[1])
                metries.update(outputs[2])
            return total_loss, losses, metries
        else:
            predicted = []
            for head in self.heads:
                predicted += head(x)
            return predicted






