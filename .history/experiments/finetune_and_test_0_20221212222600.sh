DATASET_NAME="sri/py150"
VIEWS=("random" "v2-3-z_rand_1-pgd_3_no-transforms.Replace-csn-py")
TRAIN_DECODER_ONLY=("true" "false")
TRANSFORM_NAME="transforms.Replace"
NUM_REPLACE=1500
DATASET_NAME_SMALL="py150"
for VIEWS_TYPE in "${VIEWS[@]}"; do

    PRETRAINED_MODEL="pretrain_csn_hidden_identity_${VIEWS_TYPE}"
    # pretrain a seq2seq encoder on the csn python train set
    VIEWS=${VIEWS_TYPE} \
    DATASET=csn/python \
    MODEL_NAME=${PRETRAINED_MODEL} \
        #time make pretrain-contracode

    for DECODER_ONLY in "${TRAIN_DECODER_ONLY[@]}"; do

        echo "CONFIG: views=${VIEWS_TYPE}; decoder_only=${DECODER_ONLY}"
        FINETUNE_EPOCHS=10
        if [ "$VIEWS_TYPE" != "random" ]; then 
            VIEWS_TYPE="adversarial"
        fi
        FINETUNED_MODEL="finetuned_sri_hidden_identity_${VIEWS_TYPE}_decoder-only-${DECODER_ONLY}-epochs-${FINETUNE_EPOCHS}"

        # finetune the pretrained model for code summarization
        GPU=0 \
        MODEL_TYPE="seq2seq" \
        DATASET=${DATASET_NAME} \
        DECODER_ONLY=${DECODER_ONLY} \
        EPOCHS=${FINETUNE_EPOCHS} \
        CHECKPOINT="${PRETRAINED_MODEL}/ckpt_pretrain_ep0100_step0075000.pth" \
        MODEL_NAME=${FINETUNED_MODEL} \
            time make finetune-contracode

        # attack and test the trained model

        FINAL_MODEL="final-models/seq2seq/$DATASET_NAME/$FINETUNED_MODEL"

        # no attack + test on full test set
        # ./experiments/attack_and_test_seq2seq.sh 0 2 1 false 1 1 false 1 false false false v2-1-z_no_no-pgd_no_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-${VIEWS_TYPE}-full-dec_only-${DECODER_ONLY} $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $FINAL_MODEL $NUM_REPLACE 0 true true false "Best_F1"

        # # z_1_random + u_optim (uwisc) attack + test on exact matches
        # ./experiments/attack_and_test_seq2seq.sh 0 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-${VIEWS_TYPE}-em-dec_only-${DECODER_ONLY} $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $FINAL_MODEL $NUM_REPLACE 1 true true false "Best_F1"

        # # z_1_random + u_optim (uwisc) attack + test on full test set
        # ./experiments/attack_and_test_seq2seq.sh 0 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-${VIEWS_TYPE}-full-dec_only-${DECODER_ONLY} $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $FINAL_MODEL $NUM_REPLACE 0 true true false "Best_F1"

        # # z_1_random + u random attack + test on full test set
        # bash ./experiments/attack_and_test_random_seq2seq.sh 0 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-random_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-${VIEWS_TYPE}-full-dec_only-${DECODER_ONLY} $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $FINAL_MODEL $NUM_REPLACE 0 true false true "Best_F1"
        
        # # z_1_random + u random attack + test on exact match
        # bash ./experiments/attack_and_test_random_seq2seq.sh 0 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-random_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-${VIEWS_TYPE}-em-dec_only-${DECODER_ONLY} $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $FINAL_MODEL $NUM_REPLACE 1 true false true "Best_F1"
    done
done