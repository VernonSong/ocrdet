# -*- coding: utf-8 -*-
# @Time : 2020/8/12 5:49 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from modeling.heads.build import build_head
from modeling.heads.east import EASTHead
from modeling.heads.db import DBHead
from modeling.heads.pse import PSEHead
from modeling.heads.semantic import SemanticHead


__all__ = [
    'build_head'
]