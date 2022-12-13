ARGS="--regular_training --epochs 1 --batch_size 64" \
GPU=1 \
DECODER_ONLY="false" \
MODELS_OUT=final-models/seq2seq/codeclone/java/ \
DATASET_NAME=datasets/transformed/preprocessed/tokens/codeclone/java/transforms.Identity \
time make train-model-codeclone
