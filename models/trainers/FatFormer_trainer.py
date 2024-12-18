import functools
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.trainers.base import BaseTrainer, init_weights
from models.modules.FatFormer import build_model

class FatFormerTrainer(BaseTrainer):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(FatFormerTrainer, self).__init__(opt)
        self.opt = opt

        self.model = build_model(self.opt)

        # set trainable weights

        for name, param in self.model.named_parameters():
            if 'clip_model.visual' in name:
                if 'forgery_aware_adapter' in name:
                    param.requires_grad = True
                elif 'ln_post' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif 'clip_model' in name:
                if 'clip_model.transformer' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = True

        # DDP mode
        self.model = DDP(self.model.to(opt.device), device_ids=[opt.device], find_unused_parameters=True)

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999),
                                               weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)

        elif opt.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.CrossEntropyLoss()

        self.model.to(opt.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.opt.device)
        self.label = input[1].to(self.opt.device)

    def forward(self):
        self.output = self.model(self.input)
        # self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):
        # return self.loss_fn(self.output.squeeze(1), self.label)
        return self.loss_fn(self.output, self.label)
    def optimize_parameters(self):
        self.forward()
        # self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.loss = self.loss_fn(self.output, self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



