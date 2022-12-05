from transformers.data.data_collator import InputDataClass, DataCollatorForLanguageModeling, default_data_collator
from typing import List, Union, Dict
import numpy as np
import torch
import random, math
from itertools import chain, tee


class DataCollator:
    """
    Wrapper for multiple datacollater wrt dataset/task for batches
    """

    def __init__(self, tokenizer, mlm_probability=0.15, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.data_collators = {
            0: DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_probability),
            1: default_data_collator,
            2: default_data_collator
        }

    def __call__(
            self, features: List[Union[InputDataClass, Dict]]
    ) -> Dict[str, torch.Tensor]:
        # task-type: {0:'mlm', 1:'token-classification', 2:'sequence-classification'}
        task_type = features[0]['task_type'].data
        return self.data_collators[task_type.item()](features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class DataLoaderWithTaskLangName:
    """
    Wrapper for a DataLoader to also yield a task that is represented together with a language {task-name}:{lang-name}
    """

    def __init__(self, task_name, lang_name, data_loader):
        self.task_name = task_name
        self.lang_name = lang_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(f'{self.task_name}:{self.lang_name}')
            yield batch


class Dataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict, evaluation=False, sampling_strategy=None, temperature=None):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(len(dataloader.dataset) for dataloader in self.dataloader_dict.values())
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.evaluation = evaluation

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        if self.evaluation:
            self.sampling_strategy == 'no_sampling'
        if self.sampling_strategy == 'temperature':
            sampled_batch_numbers = self.temperature_sampling(self.num_batches_dict)
        elif self.sampling_strategy == 'size_proportional':
            sampled_batch_numbers = self.size_proportional_sampling(self.num_batches_dict)
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * sampled_batch_numbers[task_name]
        if sum(self.num_batches_dict.values()) - len(task_choice_list) > 0:
            random_tasks = random.choices(task_choice_list,
                                          k=sum(self.num_batches_dict.values()) - len(task_choice_list))
            for t in random_tasks:
                sampled_batch_numbers[self.task_name_list[t]] += 1
            task_choice_list += random_tasks
        task_choice_list = np.array(task_choice_list)
        if not self.evaluation:
            np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(chain(*tee(dataloader,
                                       math.ceil(sampled_batch_numbers[task_name] / self.num_batches_dict[task_name]))))
            if self.sampling_strategy == 'temperature' and
               sampled_batch_numbers[task_name] > self.num_batches_dict[task_name]
            else iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])

    def temperature_sampling(self, num_batches_dict):
        total_size = sum(num_batches_dict.values())
        sampling_ratios = {task_name: (size / total_size) ** (1.0 / self.temperature)
                           for task_name, size in num_batches_dict.items()}
        sampling_ratios = {task_name: sampling_ratios[task_name] / sum(sampling_ratios.values())
                           for task_name in num_batches_dict.keys()}
        # upsampled_numbers = {task_name: int(sampling_ratios[task_name] / max(sampling_ratios.values()) *
        #                                    max(num_batches_dict.values())) for task_name in sampling_ratios.keys()}
        sampled_numbers = {task_name: int(sampling_ratios[task_name] * sum(num_batches_dict.values()))
                           for task_name in num_batches_dict.keys()}
        return sampled_numbers

    def size_proportional_sampling(self, num_batches_dict):
        return num_batches_dict
