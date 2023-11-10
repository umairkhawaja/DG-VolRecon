#!/usr/bin/env bash


DATASET="/home/dataset/DTU_TEST/"

# LOAD_CKPT="checkpoints/epoch=15-step=193199.ckpt" # Baseline
LOAD_CKPT="checkpoints/lightning_logs/xe9a4rud/checkpoints/epoch=1-step=48300.ckpt" # Your own weights

OUT_DIR="./eval_outputs/test/"

python main.py --extract_geometry --set 0 \
--test_n_view 3 --test_ray_num 400 --volume_reso 96 \
--test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@
