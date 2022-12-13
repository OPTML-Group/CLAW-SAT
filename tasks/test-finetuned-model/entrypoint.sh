#!/bin/bash

set -ex

export PYTHONPATH="$PYTHONPATH:/seq2seq/"

python /contracode/representjs/main.py test --batch_size 32 --num_workers 8 --n_decoder_layers 4 \
  --program_mode identity --label_mode sri/py150 --model_type $MODEL_TYPE \
  --checkpoint_file /contracode/data/runs/names_ft/ckpt_best.pth \
  --test_filepath /inputs/raw/test.jsonl.gz \
  --spm_filepath /outputs/spm/sri_py150_train_unigram_spm.model