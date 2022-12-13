#!/bin/bash

set -ex

# export PYTHONPATH="$PYTHONPATH:/seq2seq/"

TRAIN_FILE=/inputs/preprocessed/train.tsv
VALID_FILE=/inputs/preprocessed/valid.tsv
if grep -qF 'from_file' "${TRAIN_FILE}"; then
  echo "Stripping first column hashes..."
  cat "${TRAIN_FILE}" | cut -f2- > /train.tsv
  cat "${VALID_FILE}" | cut -f2- > /valid.tsv
  TRAIN_FILE=/train.tsv
  VALID_FILE=/valid.tsv
fi

# python /contracode/representjs/main.py train --run_name names_ft \
#   --program_mode identity --label_mode sri/py150 --model_type $MODEL_TYPE \
#   --num_epochs 10 --save_every 5 --batch_size 32 --num_workers 4 --lr 1e-4 \
#   --train_filepath $TRAIN_FILE \
#   --eval_filepath $VALID_FILE \
#   --resume_path /outputs/contracode/pretrain_lstm2l_hidden_1629915688/ckpt_pretrain_ep0101_step0235000.pth \
#   --vocab_filepath /seq2seq_checkpoints --max_length 128 --d_model 128

FLAGS=""
if [ "${DECODER_ONLY}" == "true" ]; then
    FLAGS+="--fix_encoder"
fi

python /model/compute_weight_difference.py \
    --train_path ${TRAIN_FILE} \
    --dev_path ${VALID_FILE} \
    --expt_name lstm \
    --expt_dir /outputs/seq2seq/${MODEL_NAME} \
    --load_checkpoint "Latest" \
    --pretrain /outputs/contracode/${CHECKPOINT} \
    --batch_size 64 \
    --epochs $EPOCHS \
    --vocab_path /outputs/seq2seq/sri/py150/normal/lstm/checkpoints/Best_F1 \
    --finetune \
    --hidden_size 512 \
    ${FLAGS}
