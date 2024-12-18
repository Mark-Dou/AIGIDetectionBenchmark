import os
import copy
import torch
import torch.nn as nn
import numpy as np
import argparse
import torchvision

from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import TrainerCallback, TrainingArguments, PreTrainedModel
from transformers.trainer_callback import TrainerState, TrainerControl
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomInvert
from torchvision.transforms import RandAugment

from .CLBaseTrainer import CLBaseTrainer

import warnings

warnings.filterwarnings('ignore')

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['input_ids', 'labels']
        self.init_buffer()

    def init_buffer(self) -> None:
        for attr_str in self.attributes:
            setattr(self, attr_str, [None for _ in range(self.buffer_size)])

    def add_data(self, input_ids, labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param input_ids: tensor containing the images
        :param labels: tensor containing the labels
        :return:
        """
        n = input_ids.shape[0] if hasattr(input_ids, 'shape') else len(input_ids)
        for i in range(n):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.input_ids[index] = input_ids[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)

    def get_data(self, size: int) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :return:
        """
        n = self.input_ids.shape[0] if hasattr(self.input_ids, 'shape') else len(self.input_ids)

        if size > min(self.num_seen_examples, n):
            size = min(self.num_seen_examples, n)

        choice = np.random.choice(min(self.num_seen_examples, n), size=size, replace=False)

        if len(choice) == 0:
            return None, None

        input_ids = torch.stack([self.input_ids[c] for c in choice])
        labels = torch.stack([self.labels[c] for c in choice])
        return input_ids, labels

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self) -> Tuple:
        """
        Return all the items in the memory buffer.
        :return: a tuple with all the items in the memory buffer
        """
        ret_tuple = (torch.stack([ee.cpu()
                                  for ee in self.input_ids]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

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


class LinearTrainer(CLBaseTrainer):

    def __init__(self, skip_initial_training, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(GradientCallback(model=self.model, device=self.device, args=self.args))
        self.cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, GradientCallback):
                self.cb = cb
                break

        # set to 500 by default
        buffer_size: float = kwargs.get("buffer_size", 500)
        buffer_rate: float = kwargs.get("buffer_rate", 0.1)

        if buffer_size is None and (0.0 < buffer_rate < 1.0):
            real_buffer_size = int(buffer_rate * self.ave_train_samples_per_task)
        else:
            real_buffer_size = buffer_size
            print("buffer_size: " + str(buffer_size))
            Warning("buffer_size is not None, buffer_rate will be ignored")
        self.buffer = Buffer(real_buffer_size, 'cpu')

        # Augmentation Chain
        self.transform_1 = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.6, 1.)),
            RandomHorizontalFlip(),
        ).to(self.device)

        self.transform_2 = nn.Sequential(
            torchvision.transforms.ConvertImageDtype(torch.uint8),
            RandAugment(3, 9),
            torchvision.transforms.ConvertImageDtype(torch.float32),
        ).to(self.device)

        self.transform_3 = nn.Sequential(
            torchvision.transforms.ConvertImageDtype(torch.uint8),
            RandAugment(3, 9),
            torchvision.transforms.ConvertImageDtype(torch.float32),
        ).to(self.device)

        self.skip_initial_training = skip_initial_training

    def prepare_buffer_with_initial_task_data(self):
        initial_train_set = self.continual_training_dataset[self.task_names[0]]
        initial_train_dataloader = DataLoader(initial_train_set, batch_size=self.args.per_device_train_batch_size,
                                              shuffle=False, collate_fn=self.data_collator)
        for inputs in tqdm(initial_train_dataloader):
            image, label = inputs['image'], inputs['labels']
            self.buffer.add_data(image, label)
        print("Initial task data loaded into buffer.")


    def concat_inputs(self, input_ids: torch.Tensor, labels: torch.Tensor, buffer_inputs_ids: torch.Tensor,
                      buffer_labels: torch.Tensor) -> Tuple:

        input_ids = torch.cat([input_ids, buffer_inputs_ids], dim=0)
        labels = torch.cat([labels, buffer_labels], dim=0)
        return input_ids, labels

    def compute_loss(self, model, inputs, return_outputs=False, compute_kfac_ewc_loss=True):
        buffer_inputs, buffer_labels = self.buffer.get_data(len(inputs['image']))

        # add data to buffer if the current length of buffer is lower than buffer size
        self.buffer.add_data(inputs['image'], inputs['labels'])

        if self.current_task_name != self.task_names[0] and buffer_inputs is not None and buffer_labels is not None:
            buffer_inputs, buffer_labels = buffer_inputs.to(inputs['image'].device), buffer_labels.to(
                inputs['labels'].device)
            self.buffer.add_data(inputs['image'], inputs['labels'])

            concatenated_inputs, concatenated_labels = self.concat_inputs(inputs["image"], inputs["labels"],
                                                                          buffer_inputs, buffer_labels)
            inputs_updated = {"image": concatenated_inputs}
        else:
            inputs_updated = {"image": inputs["image"]}
            concatenated_labels = inputs['labels']

        # Augmentation Chain
        combined_aug1 = self.transform_1(inputs_updated["image"])
        combined_aug2 = self.transform_2(combined_aug1)
        combined_aug = self.transform_3(combined_aug2)

        outputs_vanilla = model(image=inputs_updated["image"])
        outputs_step1 = model(image=combined_aug1)
        outputs_step2 = model(image=combined_aug2)
        outputs_step3 = model(image=combined_aug)

        if self.opt.detection_model in ['SAFE']:
            loss = self.loss_fct(outputs_vanilla, concatenated_labels) + \
                   self.loss_fct(outputs_step1, concatenated_labels) + \
                   self.loss_fct(outputs_step2, concatenated_labels) + \
                   self.loss_fct(outputs_step3, concatenated_labels)
        else:
            loss = self.loss_fct(outputs_vanilla.view(-1), concatenated_labels.float()) + \
                   self.loss_fct(outputs_step1.view(-1), concatenated_labels.float()) + \
                   self.loss_fct(outputs_step2.view(-1), concatenated_labels.float()) + \
                   self.loss_fct(outputs_step3.view(-1), concatenated_labels.float())

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

            if idx == 0:
                if self.skip_initial_training:
                    # task-0 has been done (offline detector)
                    # need to rebuild K-FAC information for task-0
                    self.cb.estimate_kfac_fisher(train_set)
                    self.prepare_buffer_with_initial_task_data()
                    self.save_model(name)
                    continue
                else:
                    # continual learning for sequential tasks
                    self.update_train_set(train_set)
                    self.train()
                    self.cb.estimate_kfac_fisher(train_set)
                    self.save_model(name)
            else:

                # Remember knowledge of previous tasks
                self.pre_model = copy.deepcopy(self.model).cuda()
                p_model = copy.deepcopy(self.pre_model.state_dict())

                self.update_train_set(train_set)
                self.train()
                self.cb.estimate_kfac_fisher(train_set)

                # Learn new knowledge of current task
                q_model = self.model.state_dict()

                # Linear Interpolation
                ans = self.model.state_dict()
                beta = 1.0 / (idx + 1)
                for k in self.model.state_dict().keys():
                    ans[k] = p_model[k] * (1 - beta) + q_model[k] * beta

                self.model.load_state_dict(ans)
                self.save_model(name)