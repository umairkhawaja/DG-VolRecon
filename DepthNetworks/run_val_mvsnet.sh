#!/bin/bash

cd "MVSNet/mvsnet"

CKPT_PATH="/home/umair/VolReconWithDepthPriors/DepthNetworks/MVSNet/tf_model_dtu/3DCNNs/model.ckpt"

for SPLIT in training validation; do
    echo "Running validation for split: $SPLIT"
    python validate.py \
    --regularization '3DCNNs' \
    --validate_set dtu \
    --max_w 640 --max_h 512 --max_d 128 \
    --pretrained_model_ckpt_path "$CKPT_PATH" \
    --dtu_data_root "/home/dataset/mvs_training/dtu/" \
    --split "$SPLIT" \
    --validation_result_path "./eval_outputs"
done