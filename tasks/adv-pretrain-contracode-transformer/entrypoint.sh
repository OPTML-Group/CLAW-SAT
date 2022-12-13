#!/bin/bash

set -ex

export PYTHONPATH="$PYTHONPATH:/seq2seq/"


# pretrain the seq2seq encoder with contracode
# python /model/representjs/pretrain_distributed.py ${MODEL_NAME} \
#     --num_epochs=100 --batch_size=512 --lr=1e-4 --num_workers=4 \
#     --subword_regularization_alpha 0.1 --program_mode contrastive --save_every 5000 \
#     --train_filepath=/inputs/augmented/train.pkl.gz \
#     --min_alternatives 2 --dist_url tcp://localhost:10001 --rank 0 \
#     --encoder_type seq2seq --lstm_project_mode hidden_identity --n_encoder_layers 2 \
#     --d_model 512 --d_rep 512 \
#     --vocab_filepath /outputs/seq2seq/$DATASET/normal/lstm/checkpoints/Best_F1/input_vocab.pt \
#     --max_length 130

# python /model/representjs/pretrain_distributed.py ${MODEL_NAME} \
#   --num_epochs=200 --batch_size=512 --lr=1e-4 --num_workers=4 \
#   --subword_regularization_alpha 0.1 --program_mode contrastive --label_mode contrastive --save_every 5000 \
#   --train_filepath=/inputs/javascript_augmented.pickle.gz \
#   --spm_filepath=/inputs/csnjs_8k_9995p_unigram_url.model \
#   --min_alternatives 2 --dist_url tcp://localhost:10001 --rank 0 \
#   --encoder_type seq2seq --lstm_project_mode hidden --n_encoder_layers 2 \
#   --d_model 512 \ 
#   --d_rep 512 \ 
#   --max_length 130


# adv_pretrain
python /model/representjs/adv_pretrain_distributed.py ${MODEL_NAME} \
    --num_epochs=50 --batch_size=96 --lr=1e-5 --num_workers=6 \
    --subword_regularization_alpha 0.1 --program_mode adv_contrastive --save_every 5000 \
    --train_filepath=/inputs/augmented/train.pkl.gz \
    --min_alternatives 2 --dist_url tcp://localhost:10001 --rank 0 \
    --encoder_type adv_transformer --lstm_project_mode hidden_identity --n_encoder_layers 6 \
    --loss_mode adv \
    --d_model 512 --d_rep 512 \
    --num_replacements 1500 \
    --vocab_filepath /outputs/seq2seq/$DATASET/normal/lstm/checkpoints/Best_F1/input_vocab.pt \
    --max_length 130 \
    --adv_lr $ADV_LR \
    --u_pgd_epochs $U_PGD_EPOCHS \
    --num_sites $NUM_SITES