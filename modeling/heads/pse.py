# -*- coding: utf-8 -*-
# @Time : 2020/10/21 4:03 下午
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
class PSEHead(BaseHead):
    def __init__(self, cfg, is_training):
        super(PSEHead, self).__init__(cfg, is_training)
        stride = cfg.MODEL.PSE.STRIDE
        self.input_layer = cfg.MODEL.PSE.INPUT_LAYER
        self.scores_thresh = cfg.MODEL.PSE.SCORES_THRESH
        self.n = cfg.MODEL.PSE.N
        upsampling_scale = int(np.ceil(4 / stride))
        in_planes = self.layers_planes[self.input_layer]
        self.pse_predictor = self.default_predictor[str(upsampling_scale)](in_planes, self.n, self.norm_layer)

    def forward(self, x: dict):
        x = x[self.input_layer]
        pse_logits = self.pse_predictor(x)
        pse_map = F.sigmoid(pse_logits)
        if self.is_training:
            return {'pse_logits': pse_logits,
                    'pse_map': pse_map}
        else:
            return [pse_map]