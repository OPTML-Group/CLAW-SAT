# provide arguments in this order
# 1. GPU
# 2. attack_version
# 3. n_alt_iters
# 4. z_optim
# 5. z_init
# 6. z_epsilon
# 7. u_optim
# 8. u_pgd_epochs (v2) / pgd_epochs (v3)
# 9. u_accumulate_best_replacements
# 10. u_rand_update_pgd
# 11. use_loss_smoothing
# 12. short_name
# 13. src_field
# 14. u_learning_rate (v3)
# 15. z_learning_rate (v3)
# 16. smoothing_param (v3)
# 17. dataset (sri/py150 or c2s/java-small)
# 18. vocab_to_use 
# 19. model_in
# 20. number of replacement tokens
# 21. exact_matches (1 or 0)
# 22. freeze encoder true/false

TRANSFORM_NAME="transforms.Combined"
DATASET_NAME="sri/py150"
# DATASET_NAME="c2s/java-small"
DATASET_NAME_SMALL="py150"
MODEL_NAME="normal" # model used for the first epoch
NUM_REPLACE=1500
DECODER_ONLY="false"
EXPT_NAME="v2-3-z_rand_1-pgd_3-no-$TRANSFORM_NAME-$DATASET_NAME_SMALL"

# adversarial training starting from the normally trained model

./experiments/adv_train_transformer.sh 0 2 1 false 1 1 true 3 false false false $EXPT_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0 $DECODER_ONLY 10 0.5 


# attack and test the trained model

ADVERSARIAL_MODEL="final-models/seq2seq/$EXPT_NAME/$DATASET_NAME/$TRANSFORM_NAME/adversarial"

# no attack + test on full test set
./experiments/attack_and_test_seq2seq.sh 0 2 1 false 1 1 false 1 false false false v2-1-z_no_no-pgd_no_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-adv-full $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $ADVERSARIAL_MODEL $NUM_REPLACE 0 true

# z_1_random + u_optim (uwisc) attack + test on exact matches
./experiments/attack_and_test_seq2seq.sh 0 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-adv-em $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $ADVERSARIAL_MODEL $NUM_REPLACE 1 true

# z_1_random + u_optim (uwisc) attack + test on full test set
./experiments/attack_and_test_seq2seq.sh 0 2 1 false 1 1 true 3 false false false v2-3-z_rand_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL-adv-full $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $ADVERSARIAL_MODEL $NUM_REPLACE 0 true

