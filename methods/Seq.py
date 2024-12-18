import torch

from .CLBaseTrainer import CLBaseTrainer


class SeqTrainer(CLBaseTrainer):

    def __init__(self, skip_initial_training, **kwargs):
        super().__init__(**kwargs)

        self.skip_initial_training = skip_initial_training

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['image'])
        if self.opt.detection_model in ['SAFE']:
            loss = self.loss_fct(outputs, concatenated_labels)
        else:
            loss = self.loss_fct(outputs.view(-1), concatenated_labels.float())

        return (loss, outputs) if return_outputs else loss

    def continual_learning(self):

        for idx, (name, train_set) in enumerate(self.continual_training_dataset.items()):

            self.current_task_name = name
            if idx == 0 and self.skip_initial_training:
                # task-0 has been done (offline detector)
                self.save_model(name)
                continue

            # continual learning for sequential tasks
            self.current_task_name = name
            self.update_train_set(train_set)
            self.train()
            self.save_model(name)

