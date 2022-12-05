import json
import math
import os
import time
from typing import Optional, List
import pickle
import torch
import transformers
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from transformers.trainer_utils import speed_metrics

from src.hyperx.datacollator import DataLoaderWithTaskLangName, Dataloader


class HyperxTrainer(transformers.Trainer):
    def __init__(self, sampling_strategy=None, temperature=None, evaluater=None, hyperx_args= None, **kwargs):
        self.hpx_args = hyperx_args
        self.temperature = temperature
        self.sampling_strategy = sampling_strategy
        self.evaluater = evaluater
        super(HyperxTrainer, self).__init__(**kwargs)

    def get_single_dataloader(self, task_name, lang_name, dataset, batch_size):
        """
        Create a single-task data loader that also yields task names
        """
        train_sampler = (
            RandomSampler(dataset)
            if self.args.local_rank == -1
            else DistributedSampler(dataset)
        )

        data_loader = DataLoaderWithTaskLangName(
            task_name=task_name,
            lang_name=lang_name,
            data_loader=DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return Dataloader(
            {
                f'{task_name}:{lang_name}': self.get_single_dataloader(task_name, lang_name, task_dataset,
                                                                       self.args.train_batch_size)
                for (task_name, lang_name), task_dataset in self.train_dataset.items()
            },
            sampling_strategy=self.sampling_strategy,
            temperature=self.temperature
        )

    def get_eval_dataloader(self):
        return Dataloader(
            {
                f'{task_name}:{lang_name}': self.get_single_dataloader(task_name, lang_name, task_dataset,
                                                                       self.args.eval_batch_size)
                for (task_name, lang_name), task_dataset in self.eval_dataset.items()
            }, evaluation=True
        )

    def get_test_dataloader(self, test_dataset):
        return Dataloader(
            {
                f'{task_name}:{lang_name}': self.get_single_dataloader(task_name, lang_name, task_dataset,
                                                                       self.args.eval_batch_size)
                for (task_name, lang_name), task_dataset in test_dataset.items()
            }, evaluation=True
        )

    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = self.args.output_dir
        for task_name in self.model.taskmodels_dict.keys():
            os.makedirs(f"{output_dir}", exist_ok=True)
            os.makedirs(f"{output_dir}/{task_name}_model", exist_ok=True)
            torch.save(self.model.taskmodels_dict[task_name].state_dict(),
                       f"{output_dir}/{task_name}_model/pytorch_model.bin")
            with open(f'{output_dir}/hyperx_args.bin', 'wb') as args_file:
                pickle.dump(self.hpx_args, args_file)
            with open(f'{output_dir}/trainer_args.bin', 'wb') as args_file:
                pickle.dump(self.args, args_file)

    def evaluate(self,
                 eval_dataset: Optional[Dataset] = None,
                 ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval"):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if metric_key_prefix == "eval":
            eval_dataloader_dict = self.get_eval_dataloader().dataloader_dict
        else:
            eval_dataloader_dict = self.get_test_dataloader(eval_dataset).dataloader_dict
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        all_output_metrics = {}
        for task_name, dataloader in eval_dataloader_dict.items():
            self.evaluater.set_task(task_name)
            self.label_names = ["labels"] if not task_name.split(':')[0] == 'dep' else ["labels_rels", "labels_arcs"]
            self.compute_metrics = self.evaluater.compute_metrics
            hpx_metric_key_prefix = f'{metric_key_prefix}_{task_name}'
            start_time = time.time()
            output = eval_loop(
                dataloader,
                description=f"Evaluation for {task_name}",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=hpx_metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            output.metrics.update(
                speed_metrics(
                    hpx_metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )
            self.log(output.metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
            self._memory_tracker.stop_and_update_metrics(output.metrics)
            all_output_metrics.update(output.metrics)
        return all_output_metrics