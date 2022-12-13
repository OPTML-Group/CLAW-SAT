TRANSFORM_NAME="transforms.Replace"
DATASET_NAME_SMALL="py150"
DATASET_NAME="sri/py150"
NUM_REPLACE=1500
# ### pgd-3   adv-partial-fine-tuning   whole test datasets###
# MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-adv-fz"
# MODEL_NAME_SMALL="Advv1CLpf" 
# EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}"
# ./experiments/adv_attack.sh 1 2 3 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false


# ### pgd-3   adv-full-fine-tuning   whole test dataset###
# MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-adv"
# MODEL_NAME_SMALL="Advv1CLff" 
# EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}"
# ./experiments/adv_attack.sh 1 2 3 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false

# ### pgd-3   cl-partial-fine-tuning   whole test datasets###
# MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-random-fz"
# MODEL_NAME_SMALL="Advv1CLpf" 
# EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}"
# ./experiments/adv_attack.sh 1 2 3 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false


# ### pgd-3   cl-full-fine-tuning   whole test dataset###
# MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-random"
# MODEL_NAME_SMALL="Advv1CLff" 
# EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}"
# ./experiments/adv_attack.sh 1 2 3 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false

# ### pgd-1   adv-partial-fine-tuning   whole test datasets###
# MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-adv-fz"
# MODEL_NAME_SMALL="Advv1CLpf" 
# EXPT_NAME="v2-3-z_rand_1-pgd_1_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}"
# ./experiments/adv_attack.sh 1 2 1 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false


# ### pgd-1   adv-full-fine-tuning   whole test dataset###
# MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-adv"
# MODEL_NAME_SMALL="Advv1CLff" 
# EXPT_NAME="v2-3-z_rand_1-pgd_1_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}"
# ./experiments/adv_attack.sh 1 2 1 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false

# ### pgd-1   cl-partial-fine-tuning   whole test datasets###
# MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-random-fz"
# MODEL_NAME_SMALL="Advv1CLpf" 
# EXPT_NAME="v2-3-z_rand_1-pgd_1_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}"
# ./experiments/adv_attack.sh 1 2 1 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false


# ### pgd-1   cl-full-fine-tuning   whole test dataset###
# MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-random"
# MODEL_NAME_SMALL="Advv1CLff" 
# EXPT_NAME="v2-3-z_rand_1-pgd_1_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}"
# ./experiments/adv_attack.sh 1 2 1 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false


# ###################### exact match dataset #################
EXACT_MATCHES=exact
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-adv-fz"
MODEL_NAME_SMALL="Advv1CLpf" 
EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}-${EXACT_MATCHES}"
./experiments/adv_attack.sh 0 2 3 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1 true
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-adv"
MODEL_NAME_SMALL="Advv1CLff" 
EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}-${EXACT_MATCHES}"
./experiments/adv_attack.sh 0 2 3 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1 true
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-random-fz"
MODEL_NAME_SMALL="CLpf" 
EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}-${EXACT_MATCHES}"
./experiments/adv_attack.sh 0 2 3 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1 true
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/finetuned_csn_h-sz-512-identity-random"
MODEL_NAME_SMALL="CLff" 
EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}-${EXACT_MATCHES}"
./experiments/adv_attack.sh 0 2 3 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1 true
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/normal"
MODEL_NAME_SMALL="normal" 
EXPT_NAME="v2-3-z_rand_1-pgd_3_no-${TRANSFORM_NAME}-${DATASET_NAME_SMALL}-${MODEL_NAME_SMALL}-${EXACT_MATCHES}"
./experiments/adv_attack.sh 0 2 3 false 1 1 true 1 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1 true