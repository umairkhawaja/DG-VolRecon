#!/usr/bin/env bash

EXP_NAME="tsdf-vol-ray768-aug-dgfeat_AA-depths"
DATASET="/home/dataset/mvs_training/dtu"
LOG_DIR="./checkpoints/$EXP_NAME"
# CKPT="checkpoints/tsdf_lite_baseline/epoch=15-step=193200.ckpt"
# CKPT="checkpoints/$EXP_NAME/lightning_logs/pqwfpl0d/checkpoints/epoch=31-step=386400.ckpt"

python main.py --max_epochs 16 --batch_size 2 --lr 0.0002 \
--concat_tsdf_vol --depth_prior_name "aarmvsnet" \
--dg_feat_vol --depth_tolerance_thresh 25 \
--weight_rgb 1.0 --weight_depth 1.0 \
--train_ray_num 768 --volume_reso 48 \
--root_dir=$DATASET --logdir=$LOG_DIR $@