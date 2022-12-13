#!/bin/bash

set -ex

# gunzip /inputs/train.jsonl.gz

python /model/scripts/run_sentencepiece.py make_corpus \
    --input /inputs/train.jsonl \
    --output /inputs/train.txt

gzip /inputs/train.jsonl

python /model/scripts/run_sentencepiece.py spm_train \
    --input /inputs/train.txt \
    --model_prefix sri_py150_train_unigram_spm \
    --vocab_size 15000 \
    --character_coverage 0.9995 \
    --model_type unigram

if [ ! -e /outputs/spm ]; then
    mkdir -p /outputs/spm
fi
mv sri_py150_train_unigram_spm.model /outputs/spm
mv sri_py150_train_unigram_spm.vocab /outputs/spm
