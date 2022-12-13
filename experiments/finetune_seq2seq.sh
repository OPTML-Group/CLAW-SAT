GPU=1 \
MODEL_TYPE="seq2seq" \
DATASET="sri/py150" \
DECODER_ONLY="true" \
EPOCHS=10 \
CHECKPOINT="pretrain_csn_hidden_identity_512_512_random/ckpt_pretrain_ep0100_step0075000.pth" \
MODEL_NAME="finetuned_csn_h-sz-512-identity-random-fz" \
    time make finetune-contracode

# GPU=1 \
# MODEL_TYPE="seq2seq" \
# DATASET="sri/py150" \
# DECODER_ONLY="true" \
# EPOCHS=10 \
# CHECKPOINT="pretrain_csn_hidden_identity_512_512_adv/ckpt_pretrain_ep0100_step0075000.pth" \
# MODEL_NAME="finetuned_csn_h-sz-512-identity-adv-fz" \
#     time make finetune-contracode

# GPU=1 \
# MODEL_TYPE="seq2seq" \
# DATASET="sri/py150" \
# DECODER_ONLY="false" \
# EPOCHS=10 \
# CHECKPOINT="pretrain_csn_hidden_identity_512_512_random/ckpt_pretrain_ep0100_step0075000.pth" \
# MODEL_NAME="finetuned_csn_h-sz-512-identity-random" \
#     time make finetune-contracode

# GPU=1 \
# MODEL_TYPE="seq2seq" \
# DATASET="sri/py150" \
# DECODER_ONLY="false" \
# EPOCHS=10 \
# CHECKPOINT="pretrain_csn_hidden_identity_512_512_adv/ckpt_pretrain_ep0100_step0075000.pth" \
# MODEL_NAME="finetuned_csn_h-sz-512-identity-adv" \
#     time make finetune-contracode