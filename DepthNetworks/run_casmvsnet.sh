#!/bin/bash

cd "cascade-stereo/CasMVSNet"

# Path to the root directory of the DTU dataset
ROOT_DIR="/home/dataset/mvs_training/dtu/"
CHECKPOINT_PATH="/home/umair/VolReconWithDepthPriors/DepthNetworks/cascade-stereo/CasMVSNet/casmvsnet.ckpt"
TESTLIST="lists/dtu/trainval.txt"

# Verify the directory structure
echo "Checking directories in $ROOT_DIR/Rectified/"
ls -l "$ROOT_DIR/Rectified/"


TESTPATH="$ROOT_DIR/Rectified/"
WORKSPACE_PATH="$TESTPATH/casmvsnet_output"

# Run CasMVSNet for the current scan using the selected images
python test.py \
    --dataset=general_eval \
    --batch_size=1 \
    --testpath="$TESTPATH" \
    --testlist="$TESTLIST" \
    --interval_scale 1.06 \
    --outdir "$WORKSPACE_PATH" \
    --loadckpt "$CHECKPOINT_PATH"

echo "CasMVSNet predictions saved for selected images in all scans."
