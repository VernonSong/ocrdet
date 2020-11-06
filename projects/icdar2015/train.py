# -*- coding: utf-8 -*-
# @Time : 2020/8/15 9:50 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import argparse
import os
from engine import Trainer
from config import config as cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, default='experiments/db_resnet50_s1.yml', help="config file")
    parser.add_argument("--start", type=int, default=0, help="start iter")
    parser.add_argument("--gpu", type=int, default=0, help="start iter")

    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg.merge_from_file(args.cfg)
    trainer = Trainer(cfg)
    trainer.train(args.start)


if __name__ == "__main__":
    main()

