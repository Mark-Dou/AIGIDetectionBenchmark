import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_callback import TrainerState, TrainerControl

from .CLBaseTrainer import CLBaseTrainer


def format_name(name):
    index = name.rfind("model")
    return name[index:]


class AveTaskGradientCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__()
        self.model: nn.Module = kwargs.get("model")
        self.current_task_name = kwargs.get('current_task_name')  # need update during training
        self.n_tasks = kwargs.get('n_tasks')
        self.grads = {}
        self.task_names = kwargs.get('task_names')
        self.init_grads()

    def init_grads(self):
        """
            {param_name: torch.zeros([param_size, n_tasks], dtype=torch.bfloat16)}
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad:  # reduce memory usage
                self.grads[n] = torch.ones([p.data.numel()], dtype=torch.bfloat16).to('cpu')

    def store_grads(self):
        for n, p in self.model.named_parameters():
            if format(n) in self.grads.keys():
                self.ave_grads(n, p.grad.detach().clone().view(-1).to('cpu'))
                # print(f"store {n} grad")

    def ave_grads(self, formated_name, new_grads):
        self.grads[formated_name] = (self.grads[formated_name] * (
                    self.task_names.index(self.current_task_name) + 1) + new_grads) / (
                                                self.task_names.index(self.current_task_name) + 2)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
            update grads with projected grads if the dot product of current grads and previous grads is negative
        """
        for n, p in self.model.named_parameters():
            if n in self.grads.keys() and p.requires_grad and p.grad is not None:
                p.grad = self.get_updated_grads(n, p.grad)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
            store current task grads
        """
        self.store_grads()

    def get_updated_grads(self, name, grad, eps=1e-4):
        """
            name: param name
            grad: current param grad
            idx: current task index
        """
        ori_shape = grad.shape
        grad = grad.view(-1)
        pre_grad = self.grads[name].cuda().to(torch.float32)
        grad, pre_grad = grad.unsqueeze(1), pre_grad.unsqueeze(1)
        dot_product = torch.mm(grad.t(), pre_grad)

        if (dot_product < 0) != 0:
            new_grad = grad - (torch.mm(grad.t(), pre_grad) + eps) / (torch.mm(pre_grad.t(), pre_grad) + eps) * pre_grad
            grad.copy_(new_grad)

        return grad.view(ori_shape)

    def update_current_task_name(self, name: str):
        self.current_task_name = name


class AveGEMTrainer(CLBaseTrainer):
    def __init__(self, skip_initial_training, **kwargs):
        super().__init__(**kwargs)

        self.add_callback(AveTaskGradientCallback(
            model=self.model,
            current_task_name=self.current_task_name,
            n_tasks=self.n_tasks,
            task_names=self.task_names))

        self.skip_initial_training = skip_initial_training
        self.gem_cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, AveTaskGradientCallback):
                self.gem_cb = cb
                break

    def rebuild_gradient_information(self, dataloader):
        self.model.train()

        for inputs in tqdm(dataloader):
            images, labels = inputs['image'].to(self.device), inputs['label'].to(self.device),
            outputs = self.model(images)
            self.model.zero_grad()
            loss = self.compute_loss(self.model, {"image": images, "labels": labels}, return_outputs=False)
            loss.backward()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['image'])
        if self.opt.detection_model in ['SAFE']:
            loss = self.loss_fct(outputs, inputs['labels'])
        else:
            loss = self.loss_fct(outputs.view(-1), inputs['labels'].float())

        return (loss, outputs) if return_outputs else loss

    def continual_learning(self):

        for idx, (name, train_set) in enumerate(self.continual_training_dataset.items()):

            self.current_task_name = name
            self.gem_cb.update_current_task_name(name)

            if idx == 0 and self.skip_initial_training:
                # task-0 has been done (offline detector)
                # need to rebuild gradient information for task-0
                initial_train_set = self.continual_training_dataset[self.task_names[0]]
                initial_train_dataloader = DataLoader(initial_train_set,
                                                      batch_size=self.args.per_device_train_batch_size,
                                                      shuffle=True,)
                self.rebuild_gradient_information(initial_train_dataloader)
                self.gem_cb.store_grads()
                self.save_model(name)
                continue

            # continual learning for sequential tasks
            self.update_train_set(train_set)
            self.train()
            self.save_model(name)
