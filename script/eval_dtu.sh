#!/usr/bin/env bash


DATASET="/home/dataset/DTU_TEST/"

# LOAD_CKPT="checkpoints/epoch=15-step=193199.ckpt" # Baseline
LOAD_CKPT="checkpoints/tsdf_lite_baseline/epoch=15-step=193200.ckpt" # Your own weights

OUT_DIR="./eval_outputs/tsdf_lite_baseline/"

python main.py --extract_geometry --set 0 \
--test_n_view 3 --test_ray_num 512 --volume_reso 48 \
--test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@
