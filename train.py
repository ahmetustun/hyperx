import logging
import os
import re
import random

import torch
from prettytable import PrettyTable
import transformers
from arguments import parse_train_args
from src.hyperx.setup import HyperxSetup
from src.hyperx.hyperx_trainer import HyperxTrainer
from datasets.utils.tqdm_utils import set_progress_bar_enabled
import warnings

logger = logging.getLogger(__name__)
set_progress_bar_enabled(False)
warnings.filterwarnings('ignore', module='seqeval')


def main():
    args = parse_train_args()
    train_task_lang_pair = args.train_task_language_pairs
    eval_task_lang_pair = args.eval_task_language_pairs

    hpx_setup = HyperxSetup(args)
    hpx_model = hpx_setup.setup_model()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.load_from_ckpt:
        logger.warning(f'***** Loading model from {args.load_from_ckpt} *****')
        load_result = dict()
        for sub_dir in os.listdir(args.load_from_ckpt):
            if sub_dir.endswith('_model') and sub_dir.split('_')[0] in hpx_model.taskmodels_dict:
                load_result[sub_dir.split('_')[0]] = hpx_model.taskmodels_dict[sub_dir.split('_')[0]].load_state_dict(
                    torch.load(os.path.join(args.load_from_ckpt, sub_dir, 'pytorch_model.bin'),
                               map_location=device), strict=False)
        for result in load_result:
            logger.warning(f'{result}: {load_result[result]}')

    dataset_dict = hpx_setup.setup_datasets()

    logger.warning('***** Mapping datasets to input features *****')
    features_dict = {}
    for task_name, lang_dict in dataset_dict.items():
        for lang_name, dataset in lang_dict.items():
            features_dict[(task_name, lang_name)] = {}
            for phase, phase_dataset in dataset.items():
                features_dict[(task_name, lang_name)][phase] = phase_dataset.map(
                    hpx_setup.convert_func_dict[task_name],
                    batched=True,
                    load_from_cache_file=False,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=phase_dataset.column_names
                )
                features_dict[(task_name, lang_name)][phase].set_format(
                    type="torch",
                    columns=hpx_setup.columns_dict[task_name],
                )
                logger.warning(
                    f'{task_name} '
                    f'{lang_name} '
                    f'{phase} '
                    f'{len(features_dict[(task_name, lang_name)][phase])}'
                )

    train_dataset = {}
    train_from_test = set()
    for (task_name, lang_name), dataset in features_dict.items():
        if f'{task_name}#{lang_name}' in train_task_lang_pair:
            if 'train' in dataset:
                if args.max_train_samples and args.max_train_samples < len(dataset["train"]):
                    train_dataset[(task_name, lang_name)] = dataset["train"].select(
                        [i for i in range(args.max_train_samples)])
                else:
                    train_dataset[(task_name, lang_name)] = dataset["train"]
            else:
                logger.warning(f'***** Using test data for training __{task_name}#{lang_name}__ *****')
                train_from_test.add((task_name, lang_name))
                if args.max_train_samples and args.max_train_samples < len(dataset["test"]):
                    train_dataset[(task_name, lang_name)] = dataset["test"].select(
                        [i for i in range(args.max_train_samples)])
                else:
                    train_dataset[(task_name, lang_name)] = dataset["test"]

    eval_dataset = {}
    for (task_name, lang_name), dataset in features_dict.items():
        if (task_name, lang_name) not in train_from_test and \
                f'{task_name}#{lang_name}' in eval_task_lang_pair and \
                'validation' in dataset:
            eval_dataset[(task_name, lang_name)] = dataset['validation']

    if args.unfreeze_params_regex:
        for n, p in hpx_model.encoder.named_parameters():
            if re.search(args.unfreeze_params_regex, n):
                p.requires_grad = True

    if args.freeze_params_regex:
        for n, p in hpx_model.encoder.named_parameters():
            if re.search(args.freeze_params_regex, n):
                p.requires_grad = False

    frozen_params = []
    unfrozen_params = []
    for n, p in hpx_model.named_parameters():
        if not p.requires_grad:
            frozen_params.append(n)
        else:
            unfrozen_params.append(n)

    table = PrettyTable(['Modules', 'Total Params', 'Trainable Params'])
    table.add_row([hpx_setup.model_name.split('-')[0],
                   sum(p.numel() for p in hpx_model.encoder.parameters()),
                   sum(p.numel() for p in hpx_model.encoder.parameters() if p.requires_grad)])
    for task_name, model in hpx_model.taskmodels_dict.items():
        if task_name != 'dep':
            table.add_row([task_name,
                       sum(p.numel() for p in hpx_model.taskmodels_dict[task_name].cls.parameters()),
                       sum(p.numel() for p in hpx_model.taskmodels_dict[task_name].cls.parameters() if p.requires_grad)])
        else:
            table.add_row([task_name,
                           sum([sum(p.numel() for p in hpx_model.taskmodels_dict[task_name].biaffine_arcs.parameters()),
                               sum(p.numel() for p in hpx_model.taskmodels_dict[task_name].biaffine_rels.parameters())]),
                           sum([sum(p.numel() for p in hpx_model.taskmodels_dict[task_name].biaffine_arcs.parameters()
                                    if p.requires_grad),
                               sum(p.numel() for p in hpx_model.taskmodels_dict[task_name].biaffine_rels.parameters()
                                   if p.requires_grad)])])

    logger.warning('***** Parameter Table *****')
    logger.warning(f'Unfrozen params: {unfrozen_params}')
    logger.warning(f'Frozen params: {frozen_params}')
    logger.warning(table)

    trainer = HyperxTrainer(
        model=hpx_model,
        hyperx_args=args,
        args=transformers.TrainingArguments(
            num_train_epochs=args.num_train_epochs,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            do_train=True,
            max_steps=args.max_train_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            save_steps=args.save_steps,
            no_cuda=args.no_cuda,
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            disable_tqdm=args.disable_progress_bar,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed,
            fp16=args.fp16,
        ),
        data_collator=hpx_setup.data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        temperature=args.temperature,
        sampling_strategy=args.sampling_strategy,
        evaluater=hpx_setup.hpx_eval,
    )
    trainer.train()


if __name__ == "__main__":
    main()
