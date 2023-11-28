#!/usr/bin/env bash


DATASET="./DTU_TEST"

LOAD_CKPT="checkpoints/epoch=15-step=193199.ckpt" 
LOAD_CKPT="./checkpoints/lightning_logs/v31xuhtt/checkpoints/epoch=15-step=193200.ckpt"

OUT_DIR="./outputs"

python main.py --extract_geometry --set 0 \
--test_n_view 3 --test_ray_num 512 --volume_reso 48 \
--test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@
