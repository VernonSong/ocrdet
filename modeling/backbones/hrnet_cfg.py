# -*- coding: utf-8 -*-
# @Time : 2020/8/31 10:53 上午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from yacs.config import CfgNode as CN

_C = CN()

# high_resoluton_net related params for segmentation
_C.HIGH_RESOLUTION_NET = CN()
_C.HIGH_RESOLUTION_NET.PRETRAINED = ''
_C.HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
_C.HIGH_RESOLUTION_NET.STEM_INPLANES = 64
_C.HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
_C.HIGH_RESOLUTION_NET.WITH_HEAD = True

_C.HIGH_RESOLUTION_NET.STAGE1 = CN()
_C.HIGH_RESOLUTION_NET.STAGE1.NUM_MODULES = 1
_C.HIGH_RESOLUTION_NET.STAGE1.NUM_BRANCHES = 1
_C.HIGH_RESOLUTION_NET.STAGE1.NUM_BLOCKS = [4]
_C.HIGH_RESOLUTION_NET.STAGE1.NUM_CHANNELS = [32]
_C.HIGH_RESOLUTION_NET.STAGE1.BLOCK = 'BASIC'
_C.HIGH_RESOLUTION_NET.STAGE1.FUSE_METHOD = 'SUM'

_C.HIGH_RESOLUTION_NET.STAGE2 = CN()
_C.HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
_C.HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
_C.HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
_C.HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [32, 64]
_C.HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
_C.HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

_C.HIGH_RESOLUTION_NET.STAGE3 = CN()
_C.HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
_C.HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
_C.HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [32, 64, 128]
_C.HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
_C.HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

_C.HIGH_RESOLUTION_NET.STAGE4 = CN()
_C.HIGH_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
_C.HIGH_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
_C.HIGH_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.HIGH_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
_C.HIGH_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
_C.HIGH_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'
_C.HIGH_RESOLUTION_NET.NUM_CLASSES = 1000