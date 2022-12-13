#!/bin/bash

if [ "${EXPT_NAME}" == "random" ]; then

    echo "random augmentation"
    python3 /app/augment.py \
        --src_path /mnt/preprocessed/${SPLIT}.tsv \
        --dest_path /mnt/augmented \
        --original_map /mnt/preprocessed/${SPLIT}_site_map.json \
        --split ${SPLIT} \
        --transform_name ${TRANSFORM} \
        --vocab /mnt/model/input_vocab.pt \
        --random

    gzip /mnt/augmented/${SPLIT}.pkl
elif [ "${EXPT_NAME}" == "adv-contra" ]; then
    echo "preparing adversarial train dataset for adv-contra"
    python3 /app/app.py train $@

    python3 /app/augment.py \
        --src_path /mnt/preprocessed/${SPLIT}.tsv \
        --dest_path /mnt/augmented \
        --original_map /mnt/preprocessed/${SPLIT}_site_map.json \
        --split ${SPLIT} \
        --transform_name ${TRANSFORM} \
        --idx_to_hash /mnt/adversarial/${SPLIT}_idx_to_fname.json \
        --vocab /mnt/model/input_vocab.pt \
        --random \
        --adv

    gzip /mnt/augmented/${SPLIT}.pkl
elif [ "${EXPT_NAME}" == "combined" ]; then

    echo "joining random and adversarial views"
    python3 /app/join_views.py \
        --random_views_path /mnt/augmented_dir/random/tokens/csn/python/train.pkl.gz \
        --adv_views_path /mnt/augmented_dir/v2-3-z_rand_1-pgd_3_no-transforms.Replace-csn-py/tokens/csn/python/train.pkl.gz \
        --joined_views_path /mnt/augmented/${SPLIT}.pkl.gz
elif [ "${EXPT_NAME}" == "combined_random" ]; then

    echo "joining small and big random views"
    python3 /app/join_datasets.py \
        --random_views_path /mnt/augmented_dir/random/tokens/codeclone/java/train.pkl.gz \
        --adv_views_path /mnt/augmented_dir/random/tokens/c2s/java-small/train.pkl.gz \
        --joined_views_path /mnt/augmented/${SPLIT}.pkl.gz
elif [ "${EXPT_NAME}" == "combined_adv" ]; then

    echo "joining small and big adv views"
    python3 /app/join_datasets.py \
        --random_views_path /mnt/augmented_dir/adv-contra/tokens/codeclone/java/train.pkl.gz \
        --adv_views_path /mnt/augmented_dir/adv-contra/tokens/c2s/java-small/train.pkl.gz \
        --joined_views_path /mnt/augmented/${SPLIT}.pkl.gz
else
    echo "adversarial augmentation"
    python3 /app/augment.py \
        --src_path /mnt/preprocessed/${SPLIT}.tsv \
        --dest_path /mnt/augmented \
        --optim_map /mnt/adversarial/targets-${SPLIT}-gradient-optim-only.json \
        --original_map /mnt/preprocessed/${SPLIT}_site_map.json \
        --idx_to_hash /mnt/adversarial/${SPLIT}_idx_to_fname.json \
        --split ${SPLIT} \
        --transform_name ${TRANSFORM} \
        --vocab /mnt/model/input_vocab.pt

    gzip /mnt/augmented/${SPLIT}.pkl
fi
