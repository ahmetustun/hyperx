TASKS=$2
LANGS=$3
TRAIN_TASK_LANGS=$4
EVAL_TASK_LANGS=$5

python train.py --model_name_or_path 'bert-base-multilingual-cased' \
                --max_train_steps 100000 \
                --preprocessing_num_workers 1 \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --output_dir 'outputs' \
                --save_steps 5000 \
                --eval_steps 5000 \
                --tasks $TASKS \
                --languages $LANGS \
                --train_task_language_pairs $TRAIN_TRAIN_TASK_LANGS \
                --eval_task_language_pairs $EVAL_TASK_LANGS \
                --condition_to_layer_id \
                --project_source_embeddings \
                --projected_source_embedding_dim 192 \
                --adapter_dim 256 \
                --learning_rate 1e-4 \
                --warmup_steps 4000 \
                --freeze_params_regex '^(?!.*(hypernet|LayerNorm)).*' \
                --conditional_layer_norm \
                --add_layer_norm_after_adapter \
                --fp16