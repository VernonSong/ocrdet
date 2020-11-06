# -*- coding: utf-8 -*-
# @Time : 2020/8/17 6:46 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import subprocess

# 阿里云m40
subprocess.run("nohup /home/anaconda3/bin/python train.py >/dev/null 2>&1 &", shell=True)