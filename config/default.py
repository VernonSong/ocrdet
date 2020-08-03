# -*- coding: utf-8 -*-
# @Time : 2020/7/31 2:56 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
# 数据集名字
_C.DATA.NAMES = []
# 数据集文件夹
_C.DATA.DIRS = []

_C.AUGMENT = CN()
# 模型输入图片大小
_C.AUGMENT.INPUT_SHAPE = (512, 512)
# 裁剪最大尝试次数
_C.AUGMENT.MAX_TRY_TIMES = 30
# 翻转比例
_C.AUGMENT.FLIP = 0.
# zoom in样本占比
_C.AUGMENT.ZOOM_IN = 0.5
# crop比例
_C.AUGMENT.MIN_SCALE = 0.5
# pad比例
_C.AUGMENT.MAX_SCALE = 4.
# resize范围
_C.AUGMENT.ASPECT_RATIO = (1., 1.)
# 平移范围
_C.AUGMENT.TRANSLATE_PRESENT = (-0.1, 0.1)
# 旋转范围
_C.AUGMENT.ROTATE = (-5, 5)
# 仿射变换范围
_C.AUGMENT.SHEAR = (-5, 5)
# 透视变换范围
_C.AUGMENT.PERSPECTIVE_TRANSFORM = (0.0, 0.1)
# 明暗变化范围
_C.AUGMENT.BRIGHTNESS = (0.5, 1.5)
# 色相变化范围
_C.AUGMENT.HUE = (0.5, 1.5)
# 饱和度变化范围
_C.AUGMENT.SATURATION = (0.5, 1.5)

