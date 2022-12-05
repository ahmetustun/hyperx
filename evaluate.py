import json
import logging
import os
import pickle
import torch
from arguments import parse_eval_args
from src.hyperx.setup import HyperxSetup
from src.hyperx.hyperx_trainer import HyperxTrainer
from datasets.utils.tqdm_utils import set_progress_bar_enabled
import warnings

logger = logging.getLogger(__name__)
set_progress_bar_enabled(False)
warnings.filterwarnings('ignore', module='seqeval')


def main():
    args = parse_eval_args()
    with open(os.path.join(args.eval_ckpt, 'hyperx_args.bin'), 'rb') as file:
        hyperx_args = pickle.load(file)
    with open(os.path.join(args.eval_ckpt, 'trainer_args.bin'), 'rb') as file:
        trainer_args = pickle.load(file)

    hpx_setup = HyperxSetup(hyperx_args)
    hpx_model = hpx_setup.setup_model()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.eval_ckpt:
        logger.warning(f'***** Loading model from {args.eval_ckpt} *****')
        load_result = dict()
        for sub_dir in os.listdir(args.eval_ckpt):
            if sub_dir.endswith('_model') and sub_dir.split('_')[0] in hpx_model.taskmodels_dict:
                load_result[sub_dir.split('_')[0]] = hpx_model.taskmodels_dict[sub_dir.split('_')[0]].load_state_dict(
                    torch.load(os.path.join(args.eval_ckpt, sub_dir, 'pytorch_model.bin'),
                               map_location=device), strict=False)
        for result in load_result:
            logger.warning(f'{result}: {load_result[result]}')

    test_tasks = set()
    test_langs = set()
    if args.test_task_language_pairs and hyperx_args.tasks and hyperx_args.languages:
        logger.warning(f'TEST: {args.test_task_language_pairs}')
        for pair in args.test_task_language_pairs:
            task, lang = pair.split('#')
            test_tasks.add(task)
            test_langs.add(lang)
            if not (task in hyperx_args.tasks and lang in hyperx_args.languages):
                raise ValueError(f'Test {pair} should be in training tasks-languages pair')
    else:
        raise ValueError('Test task--language pairs, tasks or languages are missing')
    dataset_dict = hpx_setup.setup_datasets()

    logger.warning('***** Mapping datasets to input features *****')
    features_dict = {}
    for task_name, lang_dict in dataset_dict.items():
        if task_name not in test_tasks:
            continue
        for lang_name, dataset in lang_dict.items():
            if lang_name not in test_langs:
                continue
            features_dict[(task_name, lang_name)] = {}
            for phase, phase_dataset in dataset.items():
                features_dict[(task_name, lang_name)][phase] = phase_dataset.map(
                    hpx_setup.convert_func_dict[task_name],
                    batched=True,
                    load_from_cache_file=False,
                    num_proc=hyperx_args.preprocessing_num_workers,
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

    test_dataset = {}
    for (task_name, lang_name), dataset in features_dict.items():
        test_dataset[(task_name, lang_name)] = dataset['test']

    trainer_args.per_device_eval_batch_size = args.per_device_eval_batch_size
    trainer = HyperxTrainer(
        model=hpx_model,
        args=trainer_args,
        evaluater=hpx_setup.hpx_eval,
    )
    eval_output = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    with open(os.path.join(args.eval_ckpt, 'results.json'), 'w') as file:
        json.dump(eval_output, file)


if __name__ == "__main__":
    main()
