import functools
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.trainers.base import BaseTrainer, init_weights
from models.modules.SimE import load_model

import loralib as lora

class SimETrainer(BaseTrainer):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(SimETrainer, self).__init__(opt)
        self.opt = opt

        self.model, _ = load_model(opt.arch, device='cpu')

        # set trainable weights
        lora.mark_only_lora_as_trainable(self.model)
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

        # DDP mode
        self.model = DDP(self.model.to(opt.device), device_ids=[opt.device])

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999),
                                               weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)

        elif opt.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.to(opt.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.opt.device)
        self.label = input[1].to(self.opt.device).float()

    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



