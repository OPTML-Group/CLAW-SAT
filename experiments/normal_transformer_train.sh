ARGS="--regular_training --epochs 10" \
GPU=1 \
DECODER_ONLY="false" \
MODELS_OUT=final-models/seq2seq/sri/py150/ \
DATASET_NAME=datasets/transformed/preprocessed/tokens/sri/py150/transforms.Identity \
time make train-model-transformer
