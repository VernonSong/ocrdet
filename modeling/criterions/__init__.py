# -*- coding: utf-8 -*-
# @Time : 2020/8/15 10:30 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from modeling.criterions.build import build_criterion
from modeling.criterions.db import DBCriterion
from modeling.criterions.semantic import SemanticCriterion


__all__ = [
    'build_criterion'
]