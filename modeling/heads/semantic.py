# -*- coding: utf-8 -*-
# @Time : 2020/8/18 6:42 下午
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
class SemanticHead(BaseHead):
    def __init__(self, cfg, is_training):
        super(SemanticHead, self).__init__(cfg, is_training)
        stride = cfg.MODEL.SEMANTIC.STRIDE
        self.input_layer = cfg.MODEL.SEMANTIC.INPUT_LAYER
        self.scores_thresh = cfg.MODEL.SEMANTIC.SCORES_THRESH
        upsampling_scale = int(np.ceil(4 / stride))
        in_planes = self.layers_planes[self.input_layer]
        self.semantic_predictor = self.default_predictor[str(upsampling_scale)](in_planes, 1, self.norm_layer)

    def forward(self, x: dict):
        x = x[self.input_layer]
        semantic_logits = self.semantic_predictor(x)
        semantic_map = F.sigmoid(semantic_logits)

        if self.is_training:
            return {'semantic_logits': semantic_logits,
                    'semantic_map': semantic_map}
        else:
            return [semantic_map]
