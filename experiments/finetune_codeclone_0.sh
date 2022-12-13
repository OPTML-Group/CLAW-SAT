GPU=0 \
MODEL_TYPE="seq2seq" \
DATASET="codeclone/java" \
DECODER_ONLY="false" \
EPOCHS=10 \
CHECKPOINT="pretrain_codeclone_combined_adv_1_1/ckpt_pretrain_ep0026_step0030000.pth" \
MODEL_NAME="finetuned-codeclone-adv" \
    time make finetune-contracode-codeclone
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