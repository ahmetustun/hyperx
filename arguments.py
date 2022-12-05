import argparse
import pickle
import os
import logging
from transformers import SchedulerType

logger = logging.getLogger(__name__)


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="polynomial",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--condition_to_layer_id", action='store_true', default=False,
        help="Use layer_id embedding in the hypernetwork"
    )
    parser.add_argument(
        "--project_source_embeddings", action='store_true', default=False,
        help="Use an intermediate projection for source embedding before the hypernetwork"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="List of tasks to learn via the hypernetwork"
    )
    parser.add_argument(
        "--languages", nargs="+", default=None, help="List of languages to learn via the hjypernetwork"
    )
    parser.add_argument(
        "--train_task_language_pairs", nargs="+", default=None,
        help="List of training task-language pairs (Notation should be <task-name>#<lang-name>)"
    )
    parser.add_argument(
        "--eval_task_language_pairs", nargs="+", default=None,
        help="List of evaluation task-language pairs during training (Notation should be <task-name>#<lang-name>)"
    )
    parser.add_argument(
        "--language_embedding_dim", type=int, default=64
    )
    parser.add_argument(
        "--task_embedding_dim", type=int, default=64
    )
    parser.add_argument(
        "--layer_id_embedding_dim", type=int, default=64
    )
    parser.add_argument(
        "--projected_source_embedding_dim", type=int, default=32,
        help="Dimension for the final projected embedding before the hypernetwork"
    )
    parser.add_argument(
        "--source_hidden_dim", type=int, default=192,
        help="Dimension for the final projected embedding before the hypernetwork"
    )
    parser.add_argument(
        "--add_layer_norm_before_adapter", action='store_true', default=False,
        help="Use a layernorm (pre-norm) at beginning of adapter module"
    )
    parser.add_argument(
        "--add_layer_norm_after_adapter", action='store_true', default=False,
        help="Use a layernorm (post-norm) at the end of adapter module"
    )
    parser.add_argument(
        "--adapter_dim", type=int, default=256, help="Bottleneck dimension for adapter layer"
    )
    parser.add_argument(
        "--adapter_non_linearity", type=str, default='relu', help="Adapter non-linearity"
    )
    parser.add_argument(
        "--conditional_layer_norm", action='store_true', default=False,
        help="Learn layernorm inside the adapter module via the hypernetwork"
    )
    parser.add_argument(
        "--save_steps", type=int, default=5000, help="Number of steps to save the checkpoint"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=5000, help="Number of steps to evaluate during training"
    )
    parser.add_argument(
        "--no_cuda", action='store_true', default=False
    )
    parser.add_argument(
        "--evaluation_strategy", type=str, default='steps'
    )
    parser.add_argument(
        "--sampling_strategy", type=str, default='temperature'
    )
    parser.add_argument(
        "--temperature", type=int, default=5
    )
    parser.add_argument(
        "--data_folder", type=str, default='data'
    )
    parser.add_argument(
        "--disable_progress_bar", action='store_true', default=False
    )
    parser.add_argument(
        "--no_hypernet", action='store_true', default=False, help="Do not use the hypernetwork"
    )
    parser.add_argument(
        "--after_layer_output", action='store_true', default=False,
        help="Position of the adapter module in the Transformer layer"
    )
    parser.add_argument(
        "--fp16", action='store_true', default=False
    )
    parser.add_argument(
        "--load_from_ckpt", type=str, default=None
    )
    parser.add_argument(
        "--unfreeze_params_regex", type=str, default=None, help="Regex for the parameter names to unfreeze"
    )
    parser.add_argument(
        "--freeze_params_regex", type=str, default=None, help="Regex for the parameter names to keep frozen"
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None
    )

    args = parser.parse_args()

    if args.train_task_language_pairs and args.tasks and args.languages:
        logger.warning(f'TRAIN: {args.train_task_language_pairs}')
        for pair in args.train_task_language_pairs:
            if not (pair.split('#')[0] in args.tasks and pair.split('#')[1] in args.languages):
                raise ValueError(f'Train {pair} should be in tasks and languages respectively')
    else:
        raise ValueError('Train task--language pairs, tasks or languages are missing')

    if args.eval_task_language_pairs:
        logger.warning(f'EVAL: {args.eval_task_language_pairs}')
        for pair in args.eval_task_language_pairs:
            if not (pair.split('#')[0] in args.tasks and pair.split('#')[1] in args.languages):
                raise ValueError(f'Eval {pair} should be in tasks and languages respectively')

    if args.condition_to_layer_id:
        args.source_embedding_dim = args.layer_id_embedding_dim + args.task_embedding_dim + args.language_embedding_dim
    else:
        args.source_embedding_dim = args.task_embedding_dim + args.language_embedding_dim

    if not args.project_source_embeddings:
        args.projected_source_embedding_dim = args.source_embedding_dim

    return args


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_task_language_pairs", nargs="+", default=None,
        help="List of test task-language pairs during training (Notation should be <task-name>#<lang-name>)"
    )
    parser.add_argument(
        "--eval_ckpt", type=str, default=None, help="Checkpoint directory to evaluate"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    return parser.parse_args()
