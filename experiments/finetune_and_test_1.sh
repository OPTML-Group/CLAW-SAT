DATASET_NAME="sri/py150"
VIEWS=("random")
TRAIN_DECODER_ONLY=("false")
TRANSFORM_NAME="transforms.Replace"

for VIEWS_TYPE in "${VIEWS[@]}"; do

    PRETRAINED_MODEL="pretrain_csn_hidden_identity_${VIEWS_TYPE}"
    # # pretrain a seq2seq encoder on the csn python train set
    # VIEWS=${VIEWS_TYPE} \
    # DATASET=csn/python \
    # MODEL_NAME=${PRETRAINED_MODEL} \
    #     time make pretrain-contracode

    for DECODER_ONLY in "${TRAIN_DECODER_ONLY[@]}"; do

        echo "CONFIG: views=${VIEWS_TYPE}; decoder_only=${DECODER_ONLY}"

        FINETUNED_MODEL="finetuned_sri_hidden_identity_${VIEWS_TYPE}_decoder-only-${DECODER_ONLY}-epochs-10"

        # # finetune the decoder for one epoch
        # GPU=1 \
        # MODEL_TYPE="seq2seq" \
        # DATASET=${DATASET_NAME} \
        # DECODER_ONLY=${DECODER_ONLY} \
        # EPOCHS=10 \
        # CHECKPOINT="${PRETRAINED_MODEL}/ckpt_pretrain_ep0100_step0075000.pth" \
        # MODEL_NAME=${FINETUNED_MODEL} \
        #     time make finetune-contracode

        # attack and test the trained model

        ADVERSARIAL_MODEL="final-models/seq2seq/$DATASET_NAME/$FINETUNED_MODEL"

        # no attack + test on full test set
        ./experiments/attack_and_test_seq2seq.sh 1 2 1 false 1 1 false 1 false false false v2-1-z_no_no-pgd_no_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-full $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $ADVERSARIAL_MODEL $NUM_REPLACE 0 true

        # z_1_random + u_optim (uwisc) attack + test on exact matches
        ./experiments/attack_and_test_seq2seq.sh 1 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-pgd_3_no-$TRANSFORM_NAME-em $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $ADVERSARIAL_MODEL $NUM_REPLACE 0 true

        # z_1_random + u_optim (uwisc) attack + test on full test set
        ./experiments/attack_and_test_seq2seq.sh 1 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-full $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $ADVERSARIAL_MODEL $NUM_REPLACE 0 true
    done
done
