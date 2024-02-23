#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

EXP_NAME="dgsampler-coarse_AA_depths" # experiment to evaluate on

DATASET="/home/dataset/DTU_TEST/"
LOAD_CKPT="checkpoints/dgsampler-coarse_AA_depths/lightning_logs/4uazbkve/checkpoints/epoch=15-step=193200.ckpt" # Your own weights
OUT_DIR="./eval_outputs/$EXP_NAME/"

echo "Starting Chamfer Distance Evaluation..."

## script/eval_dtu.sh
python main.py --extract_geometry --set 0 --depth_prior_name "aarmvsnet" \
--test_n_view 3 --test_ray_num 512 --volume_reso 48 --dg_ray_sampling \
--test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR --logdir="checkpoints/$EXP_NAME/val/" $@

# script/tsdf_fusion.sh
python tsdf_fusion.py --n_view 3 --voxel_size 1.5 --root_dir=$OUT_DIR $@

# clean mesh
python evaluation/clean_mesh.py --root_dir "$DATASET" --n_view 3 --set 0 --out_dir "$OUT_DIR/mesh/"

# calculate chamfer distance
python evaluation/dtu_eval.py --dataset_dir "/home/dataset/SampleSet/MVS Data/" --outdir "$OUT_DIR"

# calculate rgb metrics
python evaluation/rgb_eval.py --exp_name "$EXP_NAME"