TRANSFORM_NAME="transforms.Combined"
DATASET_NAME_SMALL="java-small"
DATASET_NAME="c2s/java-small"
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/normal"
NUM_REPLACE=1500

#random augmentation

DATASET=$DATASET_NAME \
TRANSFORM=$TRANSFORM_NAME \
EXPT_NAME="random" \
SPLIT="train" \
    time make augment-dataset

# # adversarial augmentation

# EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}"
# ./experiments/adv_attack.sh 1 2 1 false 1 1 true 3 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false

# DATASET=$DATASET_NAME \
# TRANSFORM=$TRANSFORM_NAME \
# EXPT_NAME=$EXPT_NAME \
# SPLIT='train' \
#     time make augment-dataset

# # join random and adversarial views
# DATASET=$DATASET_NAME \
# TRANSFORM=$TRANSFORM_NAME \
# EXPT_NAME='combined' \
# SPLIT='train' \
#     time make augment-dataset

#adv datasets

DATASET=$DATASET_NAME \
TRANSFORM=$TRANSFORM_NAME \
EXPT_NAME="adv-contra" \
SPLIT="train" \
    time make augment-dataset