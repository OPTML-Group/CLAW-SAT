DATASET_NAME="sri/py150"
TRANSFORM_NAME="transforms.Replace"
MODEL_NAME="normal"

# adversarial training starting from the normal model
NUM_REPLACE=1500
DATASET_NAME_SMALL="py150"
EXPT_NAME="v2-3-z_rand_1-pgd_3-no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-$MODEL_NAME-epochs-1"
./experiments/adv_train_seq2seq.sh 1 2 1 false 1 1 true 3 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 false

# attack and test the trained model

ADVERSARIAL_MODEL="final-models/seq2seq/$EXPT_NAME/$DATASET_NAME/$TRANSFORM_NAME/adversarial"

# no attack + test on full test set
./experiments/attack_and_test_seq2seq.sh 1 2 1 false 1 1 false 1 false false false v2-1-z_no_no-pgd_no_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-adv-full $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $ADVERSARIAL_MODEL $NUM_REPLACE 0 true

# z_1_random + u_optim (uwisc) attack + test on exact matches
./experiments/attack_and_test_seq2seq.sh 1 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-adv-em $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $ADVERSARIAL_MODEL $NUM_REPLACE 1 true

# z_1_random + u_optim (uwisc) attack + test on full test set
./experiments/attack_and_test_seq2seq.sh 1 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-adv-full $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $ADVERSARIAL_MODEL $NUM_REPLACE 0 true

