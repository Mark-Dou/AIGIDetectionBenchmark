import torch
import numpy as np

from typing import Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

from .CLBaseTrainer import CLBaseTrainer

torch.multiprocessing.set_sharing_strategy('file_system')


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


class ERTrainer(CLBaseTrainer):
    def __init__(self, skip_initial_training, **kwargs):
        super().__init__(**kwargs)

        buffer_size: float = kwargs.get("buffer_size", None)
        buffer_rate: float = kwargs.get("buffer_rate", 0.1)

        if buffer_size is None and (0.0 < buffer_rate < 1.0):
            real_buffer_size = int(buffer_rate * self.ave_train_samples_per_task)
        else:
            real_buffer_size = buffer_size
            print("buffer_size: " + str(buffer_size))
            Warning("buffer_size is not None, buffer_rate will be ignored")
        self.buffer = Buffer(real_buffer_size, 'cpu')

        self.skip_initial_training = skip_initial_training

    def prepare_buffer_with_initial_task_data(self):
        initial_train_set = self.continual_training_dataset[self.task_names[0]]
        initial_train_dataloader = DataLoader(initial_train_set, batch_size=self.args.per_device_train_batch_size, shuffle=False, collate_fn=self.data_collator)
        for inputs in tqdm(initial_train_dataloader):
            image, label = inputs['image'], inputs['labels']
            self.buffer.add_data(image, label)
        print("Initial task data loaded into buffer.")

    def concat_inputs(self, input_ids: torch.Tensor, labels: torch.Tensor, buffer_inputs_ids: torch.Tensor,
                      buffer_labels: torch.Tensor) -> Tuple:

        input_ids = torch.cat([input_ids, buffer_inputs_ids], dim=0)
        labels = torch.cat([labels, buffer_labels], dim=0)
        return input_ids, labels

    def compute_loss(self, model, inputs, return_outputs=False):
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

            outputs = model(inputs_updated["image"])
        else:
            concatenated_labels = inputs['labels']
            outputs = model(inputs["image"])

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
                # need to construct buffer for task-0
                self.prepare_buffer_with_initial_task_data()
                self.save_model(name)
                continue

            # continual learning for sequential tasks
            self.update_train_set(train_set)
            self.train()
            self.save_model(name)

