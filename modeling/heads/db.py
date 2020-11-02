# -*- coding: utf-8 -*-
# @Time : 2020/8/5 9:56 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch
import torch.nn.functional as F
import numpy as np
from modeling.heads.base_head import *
from modeling.heads.build import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class DBHead(BaseHead):
    def __init__(self, cfg, is_training):
        super(DBHead, self).__init__(cfg, is_training)
        stride = cfg.MODEL.DB.STRIDE
        self.input_layer = cfg.MODEL.DB.INPUT_LAYER
        self.scores_thresh = cfg.MODEL.DB.SCORES_THRESH
        upsampling_scale = int(np.ceil(4 / stride))
        in_planes = self.layers_planes[self.input_layer]
        self.probability_predictor = self.default_predictor[str(upsampling_scale)](in_planes, 1, self.norm_layer)
        self.thresh_predictor = self.default_predictor[str(upsampling_scale)](in_planes, 1, self.norm_layer)

    def forward(self, x: dict):
        x = x[self.input_layer]
        probability_logits = self.probability_predictor(x)
        threshold_logits = self.thresh_predictor(x)
        probability_map = F.sigmoid(probability_logits)
        threshold_map = F.sigmoid(threshold_logits)
        binary_map = torch.reciprocal(
            1 + torch.exp(-50 * (probability_map - threshold_map)))
        if self.is_training:
            return {'db_probability_logits': probability_logits,
                    'db_thresh_logits': threshold_logits,
                    'db_probability_map': probability_map,
                    'db_threshold_map': threshold_map,
                    'db_binary_map': binary_map}
        else:
            return [binary_map]

