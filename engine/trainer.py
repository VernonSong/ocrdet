# -*- coding: utf-8 -*-
# @Time : 2020/7/31 2:47 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import os
import time
import torch
from typing import List, Tuple, Union
from loader import build_train_data_loader
from engine.lr_scheduler import build_LRscheduler
from modeling import Model
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA


class Trainer(object):
    def __init__(self,
                 cfg):
        self.storage = {}
        self.device = cfg.SOLVER.DEVICE
        self.max_iter = cfg.SOLVER.MAX_ITERS
        self.log_dir = cfg.SOLVER.TENSORBOARD_WRITER.LOG_DIR
        self.base_lr = cfg.SOLVER.LR.BASE_LR
        optimizer_name = cfg.SOLVER.OPTIMIZER
        self.weight_decay = cfg.SOLVER.WEIGHT_DECAY
        self.weights = cfg.SOLVER.WEIGHTS
        self.image_period = cfg.SOLVER.TENSORBOARD_WRITER.IMAGE_PERIOD
        self.scalar_period = cfg.SOLVER.TENSORBOARD_WRITER.SCALAR_PERIOD
        self.save_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.save_model_dir = cfg.SOLVER.SAVE_DIR
        self.model_name = cfg.SOLVER.CHECKPOINT_NAME

        data_loader = build_train_data_loader(cfg)
        self._data_loader_iter = iter(data_loader)
        self.model = Model(cfg, True).train().to(self.device)
        self.optimizer = self.build_optimizer(optimizer_name, self.model)
        self.lr_scheduler = build_LRscheduler(self.optimizer, cfg)
        self.iter = 0
        self.writer = None
        self.tic = 0
        self.toc = 0

    def build_optimizer(self, name: str, model: torch.nn.Module) -> torch.optim.Optimizer:
        """No bias decay:
        Bag of Tricks for Image Classification with Convolutional Neural Networks
        (https://arxiv.org/pdf/1812.01187.pdf)"""
        weight_p, bias_p = [], []
        for p_name, p in model.named_parameters():
            if 'bias' in p_name:
                bias_p += [p]
            else:
                weight_p += [p]
        parameters = [
            {'params': weight_p, 'weight_decay': self.weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ]

        if name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=self.base_lr)
        if name == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=self.base_lr)
        if name == 'SWA':
            """Stochastic Weight Averaging: 
            Averaging Weights Leads to Wider Optima and Better Generalization
            (https://arxiv.org/pdf/1803.05407.pdf)"""
            base_opt = torch.optim.SGD(parameters, lr=self.base_lr)
            return SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=self.base_lr)

    def before_train(self):
        if self.weights != '':
            checkpoint = torch.load(self.weights)
            self.model.load_state_dict(checkpoint)
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        self.writer = SummaryWriter(self.log_dir)
        self.model.train()

    def after_train(self):
        model_name = self.model_name + '_' + str(self.iter) + '.pth'
        torch.save(self.model.state_dict(), os.path.join(self.save_model_dir, model_name))

    def before_step(self):
        self.tic = time.time()

    def after_step(self):
        # 统计时间
        self.toc = time.time()
        iter_time = self.toc - self.tic
        self.storage['iter_time'] = iter_time
        # 写tensorboard
        for key in self.storage:
            if isinstance(self.storage[key], dict):
                sub_dict = self.storage[key]
                for sub_key in sub_dict:
                    value = sub_dict[sub_key]
                    self._write_tensorboard(key+'/'+sub_key, value)
            else:
                value = self.storage[key]
                self._write_tensorboard(key, value)

        # 保存模型
        if self.iter % self.save_period == 0:
            model_name = self.model_name + '_' + str(self.iter) + '.pth'
            torch.save(self.model.state_dict(), os.path.join(self.save_model_dir, model_name))

    def _write_tensorboard(self, key: str, value: Union[torch.Tensor, int, float]):
        if isinstance(value, torch.Tensor) and len(value.shape) == 4:
            if self.iter % self.image_period == 0:
                self.writer.add_images(key, value, self.iter)
        elif self.iter % self.scalar_period == 0:
            self.writer.add_scalar(key, value, self.iter)

    def train(self, start_iter=0):
        try:
            self.before_train()
            for self.iter in range(start_iter, self.max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()
        finally:
            self.after_train()

    def run_step(self):
        data = next(self._data_loader_iter)
        total_loss, losses, metries = self.model(data)

        self.storage['total_loss'] = total_loss
        self.storage['losses'] = losses
        self.storage['image'] = data['image']
        self.storage['training_mask'] = data['training_mask']
        self.storage['metries'] = metries
        grads = {}

        self.storage['grads'] = grads

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.storage['lr'] = self.lr_scheduler.get_lr()[0]
        self.lr_scheduler.step()

        for name, parm in self.model.named_parameters():
            if parm.grad is not None:
                grads[name] = torch.mean(torch.abs(parm.grad))








