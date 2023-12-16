#!/usr/bin/env bash


DATASET="./DTU_TEST"

LOAD_CKPT="./checkpoints/lightning_logs/j7rbdqfx/checkpoints/epoch=15-step=193195.ckpt"
LOAD_CKPT="epoch=12-step=156975.ckpt"

OUT_DIR="./outputs"

python main.py --extract_geometry --set 0 \
--test_n_view 3 --test_ray_num 512 --volume_reso 48 \
--test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@
