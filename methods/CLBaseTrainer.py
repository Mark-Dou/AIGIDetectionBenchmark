import os
import torch
import torch.nn as nn
import argparse

from datasets import Dataset
from typing import Callable, Tuple, Union, Optional, Dict, List
from statistics import mean
from transformers import Trainer, PreTrainedModel, EvalPrediction, TrainerCallback, TrainingArguments


class CLBaseTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            opt: argparse.Namespace = None,
            train_dataset: Optional[Dict[str, Dataset]] = None,
            device: str = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        # Initializer of the Trainer class is adjusted to remove unnecessary parameters
        super().__init__(model, args, train_dataset=train_dataset, model_init=model_init,
                         compute_metrics=compute_metrics, callbacks=callbacks, optimizers=optimizers)

        # Initialize additional attributes
        self.args = args
        self.opt = opt
        self.continual_training_dataset = train_dataset

        # initial task name is set to the 0-task
        self.current_task_name = list(train_dataset.keys())[0]
        self.ave_train_samples_per_task = mean([len(dataset) for dataset in train_dataset.values()])
        self.task_names = list(train_dataset.keys())
        self.n_tasks = len(self.task_names)

        # loss function
        if self.opt.detection_model in ["SAFE"]:
            self.loss_fct = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fct = torch.nn.BCEWithLogitsLoss()

        self.device = device
        self.pre_model = model

    def continual_learning(self):
        for name, train_set in self.continual_training_dataset.items():
            self.current_task_name = name
            self.update_train_set(train_set)
            self.train()
            self.save_model(name)

    def update_train_set(self, train_set):
        """
            update train set before a trainer.train() start
        """
        self.train_dataset = train_set  # update train set

    def save_model(self, name) -> str:
        if self.args.output_dir is not None:
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir, exist_ok=True)

            save_dir = os.path.join(self.args.output_dir, f"checkpoint_{name}.pth")
            torch.save(self.model.state_dict(), save_dir)
            print(f"save task: {name} model to {self.args.output_dir}")
            return save_dir


