#!/usr/bin/env bash

DATASET="/home/dataset/mvs_training/dtu"
EXP_NAME="tsdf_as_weights"
LOG_DIR="./checkpoints/$EXP_NAME"
# CKPT="checkpoints/tsdf_lite_baseline/epoch=15-step=193200.ckpt"
# CKPT="checkpoints/$EXP_NAME/lightning_logs/pqwfpl0d/checkpoints/epoch=31-step=386400.ckpt"

python main.py --max_epochs 16 --batch_size 2 --lr 0.0002 \
--weight_rgb 1.0 --weight_depth 1.0 \
--train_ray_num 512 --volume_reso 48 \
--root_dir=$DATASET --logdir=$LOG_DIR $@
