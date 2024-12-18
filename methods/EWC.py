import torch

from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_callback import TrainerState, TrainerControl

from .CLBaseTrainer import CLBaseTrainer

class GradientCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__()
        self.fisher = {}
        self.model: nn.Module = kwargs.get("model")
        self.previous_weights = {}
        self.trainable_params = {}
        self.init_fisher_and_weights()
        self.ewc_loss = 0
        assert len(self.fisher) > 0 and len(
            self.trainable_params) > 0, "fisher and previous_weights should not be empty"

    def update_ewc_loss(self, ewc_loss):
        self.ewc_loss = ewc_loss

    def init_fisher_and_weights(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = p.detach().clone().data.zero_()
                self.trainable_params[n] = p.detach().clone().data

    def set_fisher_information(self, fisher):
        self.fisher = fisher

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
            save trainable parameters' weights
        """

        self.previous_weights = {n: p.detach().clone() for n, p in self.model.named_parameters() if
                                 n in self.trainable_params.keys()}

    def get_fisher_and_prior(self) -> Tuple[dict, dict]:
        return self.fisher, self.previous_weights

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
            update fisher matrix
        """
        for n, p in self.model.named_parameters():
            if n in self.trainable_params.keys() and p.grad is not None:
                self.fisher[n] += p.grad.detach().clone().data.pow(2) / state.global_step
            elif p.grad is None:
                Warning(f"parameter {n} has no gradient")


class EWCTrainer(CLBaseTrainer):

    def __init__(self, skip_initial_training, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(GradientCallback(model=self.model))
        self.ewc_lambda = kwargs.get("ewc_lambda", 0.5)
        self.cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, GradientCallback):
                self.cb = cb
                break

        self.fisher_ready = False
        self.skip_initial_training = skip_initial_training

    def rebuild_fisher_information(self, dataloader):
        self.model.train()
        new_fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if
                      param.requires_grad}

        for inputs in tqdm(dataloader):
            images, labels = inputs['image'].to(self.device), inputs['label'].to(self.device),
            outputs = self.model(images)
            self.model.zero_grad()
            loss = self.compute_loss(self.model, {"image": images, "labels": labels}, return_outputs=False)
            loss.backward()
            # accumulate fisher information
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_fisher[name] += param.grad.detach().pow(2)

        # compute average fisher information
        num_samples = len(dataloader.dataset)
        averaged_fisher = {name: fisher_info / num_samples for name, fisher_info in new_fisher.items()}
        self.cb.set_fisher_information(averaged_fisher)

        # update previous_weights
        self.cb.previous_weights = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

        self.fisher_ready = True

    def compute_loss(self, model, inputs, return_outputs=False, compute_ewc=True):
        outputs = model(inputs['image'])
        
        if self.opt.detection_model in ['SAFE']:
            loss = self.loss_fct(outputs, concatenated_labels)
        else:
            loss = self.loss_fct(outputs.view(-1), concatenated_labels.float())

        if compute_ewc and self.fisher_ready:
            ewc_loss = self.compute_ewc_loss(model)
            self.cb.update_ewc_loss(ewc_loss.item())
            loss += ewc_loss
        return (loss, outputs) if return_outputs else loss

    def compute_ewc_loss(self, model):
        """
            compute ewc loss
        """
        ewc_loss = 0
        fisher, previous_weights = self.cb.get_fisher_and_prior()
        assert len(fisher) > 0 and len(previous_weights) > 0, "fisher and previous_weights should not be empty"

        for n, p in model.named_parameters():
            n = n.replace('module.', '')
            if n in fisher.keys():
                ewc_loss += (fisher[n] * (p - previous_weights[n]).pow(2)).sum() * self.ewc_lambda / 2

        if ewc_loss < 1e-5:
            Warning("EWC regularization loss is too small, please check the hyper-parameters")
        return ewc_loss

    def continual_learning(self):

        for idx, (name, train_set) in enumerate(self.continual_training_dataset.items()):

            self.current_task_name = name

            if idx == 0 and self.skip_initial_training:
                # task-0 has been done (offline detector)
                # need to rebuild fisher information for task-0
                initial_train_set = self.continual_training_dataset[self.task_names[idx]]
                initial_train_dataloader = DataLoader(initial_train_set,
                                                      batch_size=self.args.per_device_train_batch_size,
                                                      shuffle=True)
                self.rebuild_fisher_information(initial_train_dataloader)
                self.save_model(name)
                continue

            # continual learning for sequential tasks
            self.update_train_set(train_set)
            self.train()
            self.save_model(name)