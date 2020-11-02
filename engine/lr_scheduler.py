# -*- coding: utf-8 -*-
# @Time : 2020/8/15 4:32 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import math
from bisect import bisect_right
from typing import List
import torch
from utils.register import Registry

LR_SCHEDULER_REGISTRY = Registry("LR_SCHEDULER")
LR_SCHEDULER_REGISTRY.__doc__ = """
"""

# todo warm restart

@LR_SCHEDULER_REGISTRY.register()
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cfg,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):

        self.warmup_iters = cfg.SOLVER.LR.WARMUP_ITERS
        # warm起始学习率与终止学习率的比
        self.warmup_factor = cfg.SOLVER.LR.WARMUP_FACTOR
        # 阶梯下降步数站总步数比
        milestones = cfg.SOLVER.LR.MILESTONES
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.max_iter  = cfg.SOLVER.MAX_ITERS
        self.milestones = [int(milestone*self.max_iter) for milestone in milestones]
        # 下降比
        self.gamma = cfg.SOLVER.LR.GAMMA

        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


@LR_SCHEDULER_REGISTRY.register()
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cfg: "CfgNod",
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):

        self.max_iters = cfg.SOLVER.MAX_ITERS
        self.warmup_factor = cfg.SOLVER.LR.WARMUP_FACTOR
        self.warmup_iters = cfg.SOLVER.LR.WARMUP_ITERS
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def build_LRscheduler(optimizer: torch.optim.Optimizer, cfg: "CfgNode"):
    name = cfg.SOLVER.LR.NAME
    return LR_SCHEDULER_REGISTRY.get(name)(optimizer, cfg)




