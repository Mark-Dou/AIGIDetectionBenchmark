import torch

from .CLBaseTrainer import CLBaseTrainer

class JointTrainer(CLBaseTrainer):

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

        # to save historical dataset
        accumulated_train_sets = None

        for idx, (name, train_set) in enumerate(self.continual_training_dataset.items()):

            self.current_task_name = name
            if idx == 0 and self.skip_initial_training:
                # task-0 has been done (offline detector)
                accumulated_train_sets = train_set
                self.save_model(name)
                continue

            self.current_task_name = name
            # concat current dataset to the historical datasets for jointly training
            accumulated_train_sets = torch.utils.data.ConcatDataset([accumulated_train_sets, train_set])
            self.train_dataset = accumulated_train_sets
            self.train()
            self.save_model(name)