# -*- coding: utf-8 -*-
# @Time : 2020/10/24 10:48 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import torch.nn.functional as F
import numpy as np
from modeling.heads.base_head import *
from modeling.heads.build import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class EASTHead(BaseHead):
    def __init__(self, cfg, is_training):
        super(EASTHead, self).__init__(cfg, is_training)
        stride = cfg.MODEL.EAST.STRIDE
        self.input_layer = cfg.MODEL.EAST.INPUT_LAYER
        self.scores_thresh = cfg.MODEL.EAST.SCORES_THRESH
        self.text_scale = cfg.MODEL.EAST.TEXT_SCALE

        upsampling_scale = int(np.ceil(4 / stride))
        in_planes = self.layers_planes[self.input_layer]
        self.score_predictor = self.default_predictor[str(upsampling_scale)](in_planes, 1, self.norm_layer)
        self.geo_predictor = self.default_predictor[str(upsampling_scale)](in_planes, 5, self.norm_layer)

    def forward(self, x: dict):
        x = x[self.input_layer]
        score_logits = self.score_predictor(x)
        geo_logits = self.geo_predictor(x)
        score_map = F.sigmoid(score_logits)
        distance_map = F.sigmoid(geo_logits[:, :4]) * self.text_scale
        ratation_map = (F.sigmoid(geo_logits[:, 4]) - 0.5) * np.pi/2
        if self.is_training:
            return {'east_score_logits': score_logits,
                    'east_score_map': score_map,
                    'east_distance_map': distance_map,
                    'east_rotation_map': ratation_map}
        else:
            return [score_map, distance_map, ratation_map]

