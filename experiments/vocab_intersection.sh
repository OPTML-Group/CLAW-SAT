ARGS="--regular_training --epochs 10 --batch_size 64" \
GPU=1 \
DECODER_ONLY="false" \
MODELS_OUT=final-models/seq2seq/codeclone/java/ \
DATASET_NAME=datasets/transformed/preprocessed/tokens/codeclone/java/transforms.Identity \
DATASET_NAME_2=datasets/transformed/preprocessed/tokens/c2s/java-small/transforms.Identity \
time make test-model-voc
