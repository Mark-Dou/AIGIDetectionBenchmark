import os
import torch
import torch.nn as nn
import numpy as np
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import TrainerCallback, TrainingArguments, PreTrainedModel
from transformers.trainer_callback import TrainerState, TrainerControl

from .CLBaseTrainer import CLBaseTrainer

import warnings

warnings.filterwarnings('ignore')


class GradientCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__()
        self.KFAC_FISHER_INFO = {}
        self.model: nn.Module = kwargs.get("model")
        self.device: str = kwargs.get("device")
        self.args: argparse.Namespace = kwargs.get("args")
        self.data_size = 10000

        self.initialize_kfac_fisher()
        self.gamma = 1.

        self.kfac_ewc_loss = 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        assert len(self.KFAC_FISHER_INFO) > 0, "fisher and previous_weights should not be empty"

    def register_kfac_hooks(self, model):

        layer_outputs = {}

        def create_hook_fn(name):
            def hook_fn(module, inputs, outputs):

                outputs = outputs[0] if isinstance(outputs, tuple) else outputs

                layer_outputs[name] = {
                    "name": name,
                    "input": inputs[0].detach() if isinstance(inputs, tuple) else inputs.detach(),
                    "output": outputs.detach(),
                }

            return hook_fn

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.MultiheadAttention)):
                module.register_forward_hook(create_hook_fn(name))  # 将 name 传递到 hook_fn

        return layer_outputs

    def initialize_kfac_fisher(self):

        for name, layer in self.model.named_modules():

            if isinstance(layer, nn.Linear):
                # for linear layer
                for param_name, p in layer.named_parameters():
                    if p.requires_grad:
                        if 'bias' not in param_name:
                            g_dim, a_dim = p.size(0), p.size(1)
                            A = torch.eye(a_dim).to(self.device) / torch.sqrt(
                                torch.tensor(self.data_size, dtype=torch.float32))
                            G = torch.eye(g_dim).to(self.device) / torch.sqrt(
                                torch.tensor(self.data_size, dtype=torch.float32))
                            key = f"{name}.{param_name}"

                            self.KFAC_FISHER_INFO[key] = {"A": A, "G": G, "weight": p.detach().clone().data.zero_(),
                                                          "bias": None}

            elif isinstance(layer, nn.Conv2d):
                # for convolution layer
                for param_name, p in layer.named_parameters():
                    if p.requires_grad:
                        if 'bias' not in param_name:
                            in_channels = layer.in_channels
                            out_channels = layer.out_channels
                            kernel_size = p.size(2) * p.size(3)
                            A = torch.eye(in_channels * kernel_size).to(self.device) / torch.sqrt(
                                torch.tensor(self.data_size, dtype=torch.float32))
                            G = torch.eye(out_channels * kernel_size).to(self.device) / torch.sqrt(
                                torch.tensor(self.data_size, dtype=torch.float32))
                            key = f"{name}.{param_name}"

                            self.KFAC_FISHER_INFO[key] = {"A": A, "G": G, "weight": p.detach().clone().data.zero_(),
                                                          "bias": None}

    def compute_kfac(self, a, g, layer):
        # computing covariance for activations and gradients
        if isinstance(layer, nn.Linear):
            cov_a = self.compute_cov_a_linear(a, layer)
            cov_g = self.compute_cov_g_linear(g, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = self.compute_cov_a_conv2d(a, layer)
            cov_g = self.compute_cov_g_conv2d(g, layer)
        else:
            cov_a = cov_g = None
        return cov_a, cov_g

    def compute_cov_a_linear(self, a, ):

        batch_size = a.size(0)
        return a.t() @ (a / batch_size)

    def compute_cov_g_linear(self, g):

        return (g / g.size(0)) @ g.t()

    def compute_cov_a_conv2d(self, a, layer):
        batch_size = a.size(0)

        a = self._extract_patches(a, layer.kernel_size, layer.stride, layer.padding)

        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        a = a / spatial_size
        return a.t() @ (a / batch_size)

    def compute_cov_g_conv2d(self, g, layer):
        g = g.view(-1, layer.in_channels)
        return (g / g.size(0)) @ g.t()

    def _extract_patches(self, x, kernel_size, stride, padding):

        if padding[0] + padding[1] > 0:
            x = F.pad(x, (padding[1], padding[1], padding[0],
                          padding[0])).data  # Actually check dims
        x = x.unfold(2, kernel_size[0], stride[0])
        x = x.unfold(3, kernel_size[1], stride[1])
        x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
        x = x.view(
            x.size(0), x.size(1), x.size(2),
            x.size(3) * x.size(4) * x.size(5))
        return x

    def update_kfac_info(self, a, g, name, layer):

        for param_name, param in layer.named_parameters():
            if param.requires_grad and 'bias' not in param_name:
                key = f"{name}.{param_name}"
                if key not in self.KFAC_FISHER_INFO:
                    raise KeyError(f"Key {key} not found in KFAC_FISHER_INFO.")
                A, G = self.KFAC_FISHER_INFO[key]["A"], self.KFAC_FISHER_INFO[key]["G"]
                cov_a, cov_g = self.compute_kfac(a, g, layer)

                # update k-fac information
                A = 0.9 * A + 0.1 * cov_a
                G = 0.9 * G + 0.1 * cov_g

                self.KFAC_FISHER_INFO[key]["A"], self.KFAC_FISHER_INFO[key]["G"] = A, G

    def estimate_kfac_fisher(self, dataset):
        """After completing training on a context, estimate KFAC Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes
        """

        # Set model to evaluation mode
        self.model.eval()

        # register hook
        layer_outputs = self.register_kfac_hooks(self.model)

        # Create dataloader
        data_loader = DataLoader(dataset, batch_size=self.args.per_device_train_batch_size, num_workers=8)

        for i, inputs in enumerate(tqdm(data_loader)):
            x, y = inputs['image'].to(self.device), inputs['label'].to(self.device),
            output = self.model(x)

            label = torch.LongTensor([y]) if type(y) == int else y  # -> shape: [self.fisher_batch]
            label = label.to(self.device)

            bce_loss = F.binary_cross_entropy_with_logits(output.view(-1), label.float())

            self.model.zero_grad()
            bce_loss.backward()

            for module, activations in layer_outputs.items():
                name = activations["name"]
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    a = activations["input"]
                    g = module.weight.grad.detach() if hasattr(module.weight, "grad") else None

                    if g is not None:
                        # update kfac information
                        self.update_kfac_info(a, g, name, module)

        # Set model back to its initial mode
        self.model.train()

    def update_kfac_ewc_loss(self, kfac_ewc_loss):
        self.kfac_ewc_loss = kfac_ewc_loss

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def get_KFAC_FISHER_INFO(self) -> dict:
        return self.KFAC_FISHER_INFO

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass


class OSLATrainer(CLBaseTrainer):

    def __init__(self, skip_initial_training, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(GradientCallback(model=self.model, device=self.device, args=self.args))
        self.cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, GradientCallback):
                self.cb = cb
                break

        self.skip_initial_training = skip_initial_training

    def compute_loss(self, model, inputs, return_outputs=False, compute_kfac_ewc_loss=True):
        outputs = model(inputs['image'])
        if self.opt.detection_model in ['SAFE']:
            loss = self.loss_fct(outputs, inputs['labels'])
        else:
            loss = self.loss_fct(outputs.view(-1), inputs['labels'].float())

        if compute_kfac_ewc_loss:
            kfac_loss = self.compute_kfac_loss()
            self.cb.update_kfac_ewc_loss(kfac_loss.item())
            loss += kfac_loss

        return (loss, outputs) if return_outputs else loss

    def compute_kfac_loss(self):

        def loss_for_layer(KFAC_FISHER_INFO, label, params, layer):
            info = KFAC_FISHER_INFO[label]
            A = info["A"].detach().to(self.device)  # Activation covariance
            G = info["G"].detach().to(self.device)  # Gradient covariance
            weight0 = info["weight"]  # Reference weights
            weight = params  # Current weights

            if isinstance(layer, nn.Conv2d):
                weight = weight.view(weight.size(0), -1)  # (out_channels, in_channels * kernel_size)
                weight0 = weight0.view(weight0.size(0), -1)

                dp = weight - weight0

                out_channels = layer.out_channels
                feature_map_size = G.size(0) // out_channels
                G = G.view(out_channels, feature_map_size, out_channels, feature_map_size)
                G = G.mean(dim=(1, 3))

                loss_g = torch.sum(dp.T @ G @ dp)
                loss_a = torch.sum(dp @ A @ dp.T)
                return loss_g + loss_a
            else:
                # for linear layer
                dp = weight - weight0
                return torch.sum(dp * (G @ dp @ A))

        KFAC_FISHER_INFO = self.cb.get_KFAC_FISHER_INFO()
        loss = torch.tensor(0.0, device=self.device)

        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                for param_name, p in layer.named_parameters():
                    if p.requires_grad:
                        if 'bias' not in param_name:
                            key = f"{name}.{param_name}"
                            nl = loss_for_layer(KFAC_FISHER_INFO, key, p, layer)
                            loss += nl

        return 0.5 * loss

    def continual_learning(self):

        for idx, (name, train_set) in enumerate(self.continual_training_dataset.items()):

            self.current_task_name = name

            if idx == 0 and self.skip_initial_training:
                # task-0 has been done (offline detector)
                # need to rebuild K-FAC information for task-0
                self.cb.estimate_kfac_fisher(train_set)
                self.save_model(name)
                continue

            # continual learning for sequential tasks
            self.update_train_set(train_set)
            self.train()
            self.cb.estimate_kfac_fisher(train_set)
            self.save_model(name)