import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_callback import TrainerState, TrainerControl

from .CLBaseTrainer import CLBaseTrainer

import warnings
warnings.filterwarnings('ignore')

class GradientCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__()
        self.model: nn.Module = kwargs.get("model")
        self.W = {}
        self.p_old = {}
        self.register_starting_param_values()
        self.prepare_importance_estimates_dicts()
        self.si_loss = 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        assert len(self.W) > 0 and len(
            self.p_old) > 0, "fisher and previous_weights should not be empty"

    def update_si_loss(self, si_loss):
        self.si_loss = si_loss

    def register_starting_param_values(self):
        '''Register the starting parameter values into the model as a buffer.'''
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.model.register_buffer('{}_SI_prev_context'.format(n), p.detach().clone())

    def prepare_importance_estimates_dicts(self):
        '''Prepare <dicts> to store running importance estimates and param-values before update.'''
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.W[n] = p.data.clone().zero_()
                self.p_old[n] = p.data.clone()

    def get_W(self) -> dict:
        return self.W

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''Update the running parameter importance estimates in W.'''
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                n = n.replace('.', '__')
                self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                self.p_old[n] = p.detach().clone()


class SITrainer(CLBaseTrainer):

    def __init__(self, skip_initial_training, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(GradientCallback(model=self.model))
        self.epsilon = kwargs.get("epsilon", 0.1)
        self.cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, GradientCallback):
                self.cb = cb
                break

        self.skip_initial_training = skip_initial_training

    def rebuild_information(self, dataloader):
        self.model.train()
        W = {name.replace('.', '__'): torch.zeros_like(param) for name, param in self.model.named_parameters() if
                      param.requires_grad}

        p_old = {name.replace('.', '__'): param.data.clone() for name, param in self.model.named_parameters() if
                 param.requires_grad}

        for inputs in tqdm(dataloader):
            images, labels = inputs['image'].to(self.device), inputs['label'].to(self.device),
            outputs = self.model(images)
            self.model.zero_grad()
            loss = self.compute_loss(self.model, {"image": images, "labels": labels}, return_outputs=False)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    name = name.replace('.', '__')
                    W[name] += W[name].add_(-param.grad * (param.detach() - p_old[name]))
                    p_old[name] = param.detach().clone()

        self.cb.W = W
        self.cb.p_old = {n.replace('.', '__'): p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher_ready = True

    def compute_loss(self, model, inputs, return_outputs=False, compute_si=True):
        outputs = model(inputs['image'])
        if self.opt.detection_model in ['SAFE']:
            loss = self.loss_fct(outputs, inputs['labels'])
        else:
            loss = self.loss_fct(outputs.view(-1), inputs['labels'].float())
        if compute_si:
            si_loss = self.compute_surrogate_loss(model)
            self.cb.update_si_loss(si_loss.item())
            loss += si_loss
        return (loss, outputs) if return_outputs else loss

    def compute_surrogate_loss(self, model):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self.model, '{}_SI_prev_context'.format(n))
                    omega = getattr(self.model, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p - prev_values) ** 2).sum())

            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self.device)

    def update_omega(self, model):
        '''After completing training on a context, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed context
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        W = self.cb.get_W()

        # Loop over all parameters
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self.model, '{}_SI_prev_context'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n] / (p_change ** 2 + self.epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.model.register_buffer('{}_SI_prev_context'.format(n), p_current)
                self.model.register_buffer('{}_SI_omega'.format(n), omega_new)

    def save_model(self, name) -> str:
        if self.args.output_dir is not None:
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir, exist_ok=True)

            save_dir = os.path.join(self.args.output_dir, f"checkpoint_{name}.pth")

            new_stat = {}
            for k, v in self.model.state_dict().items():
                k = k.replace('__', '.')
                if '_SI_prev_context' in k:
                    k = k.replace('_SI_prev_context', '')
                if '_SI_omega' in k:
                    k = k.replace('_SI_omega', '')
                new_stat[k] = v

            torch.save(new_stat, save_dir)
            print(f"save task: {name} to {self.args.output_dir}")
            return save_dir

    def continual_learning(self):

        for idx, (name, train_set) in enumerate(self.continual_training_dataset.items()):

            self.current_task_name = name

            if idx == 0 and self.skip_initial_training:
                # task-0 has been done (offline detector)
                # need to rebuild fisher information for task-0
                initial_train_dataloader = DataLoader(train_set,
                                                      batch_size=self.args.per_device_train_batch_size,
                                                      shuffle=True)
                self.rebuild_information(initial_train_dataloader)
                self.update_omega(model=self.model)
                self.save_model(name)
                continue

            # continual learning for sequential tasks
            self.update_train_set(train_set)
            self.cb.prepare_importance_estimates_dicts()
            self.train()
            self.update_omega(model=self.model)
            self.save_model(name)
