DATASET=sri/py150 \
GPU=0 \
ARGS='--batch_size 16' \
CHECKPOINT="Best_F1" \
STEP=3 \
DATASET_NAME_SMALL=normal \
MODEL_NAME_SMALL=normal \
DATASET_NAME=datasets/adversarial/v2-3-z_rand_1-pgd_${STEP}_no-transforms.Replace-py150-${DATASET_NAME_SMALL}-exact/tokens/${DATASET}/gradient-targeting \
RESULTS_OUT=final-results/seq2seq/${DATASET}/${MODEL_NAME_SMALL} \
MODELS_IN=final-models/seq2seq/${DATASET}/${MODEL_NAME_SMALL} \
SRC_FIELD=transforms.Replace \
GET_REPS=True \
  time make test-model-seq2seq

