import torch
import copy

from .CLBaseTrainer import CLBaseTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class iCaRLTrainer(CLBaseTrainer):

    def __init__(self, skip_initial_training, **kwargs):
        super().__init__(**kwargs)
        self.skip_initial_training = skip_initial_training

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['image'])
        if self.opt.detection_model in ['SAFE']:
            loss_cls = self.loss_fct(outputs, concatenated_labels)
        else:
            loss_cls = self.loss_fct(outputs.view(-1), concatenated_labels.float())

        loss_kd = self._KD_loss(
            outputs,
            self.pre_model(inputs['image']),
            T=2,
        )

        loss = loss_cls + loss_kd

        return (loss, outputs) if return_outputs else loss

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

    def continual_learning(self):

        for idx, (name, train_set) in enumerate(self.continual_training_dataset.items()):

            self.current_task_name = name
            if idx == 0 and self.skip_initial_training:
                # task-0 has been done (offline detector)
                self.save_model(name)
                continue

            # continual learning for sequential tasks
            self.pre_model = copy.deepcopy(self.model).cuda().eval()
            self.update_train_set(train_set)
            self.train()
            self.save_model(name)