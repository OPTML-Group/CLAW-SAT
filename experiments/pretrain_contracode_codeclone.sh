# VIEWS="adv-contra" \
# DATASET="c2s/java-small" \
# MODEL_NAME="pretrain_codeclone_random" \
#     time make pretrain-contracode-codeclone
VIEWS="combined_random" \
DATASET="c2s/java-small" \
MODEL_NAME="pretrain_codeclone_random_codeclone" \
    time make pretrain-contracode-codeclone
# DATASET="csn_javascript" \
#     time make pretrain-contracode

# VIEWS="random" \
# DATASET="csn/python" \
# MODEL_NAME="pretrain_csn_hidden_identity_512_512_random" \
#     time make pretrain-contracode