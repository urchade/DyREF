#export MODEL="spanbert-base-cased" # or bert-base-cased ...
export MODEL="/gpfswork/rech/pds/upa43yu/models/spanbert-base-cased" # or bert-base-cased ...
export OUTPUT_DIR="output"
export AGGREGATION="mean" # last, mean, max, weighted
export MASKING="full" # full, bidirectionnal, independent, causal
#export TRAIN_FILE="squad/train.jsonl" # path to training dataset
#export PREDICT_FILE="/squad/dev.jsonl" # path to dev dataset

export TRAIN_FILE="/gpfswork/rech/ohy/upa43yu/mrqa-few-shot/squad/squad-train-seed-42-num-examples-16.jsonl" # path to training dataset
export PREDICT_FILE="/gpfswork/rech/ohy/upa43yu/mrqa-few-shot/squad/squad-train-seed-42-num-examples-16.jsonl" # path to dev dataset

python train_dyref.py \
    --model_type=bert \
    --model_name_or_path=$MODEL \
    --tokenizer_name=$MODEL \
    --output_dir=$OUTPUT_DIR \
    --train_file=$TRAIN_FILE \
    --predict_file=$PREDICT_FILE \
    --do_train \
    --do_eval \
    --aggregation=$AGGREGATION \
    --masking=$MASKING \
    --cache_dir=.cache \
    --max_seq_length=384 \
    --doc_stride=128 \
    --threads=4 \
    --save_steps=50000 \
    --per_gpu_train_batch_size=12 \
    --per_gpu_eval_batch_size=16 \
    --learning_rate=3e-5 \
    --max_answer_length=10 \
    --warmup_ratio=0.1 \
    --min_steps=2500 \
    --num_train_epochs=5 \
    --seed=42 \
    --use_cache=False \
    --evaluate_every_epoch=False