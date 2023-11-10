#!/bin/bash

cd "MVSNet/mvsnet"

# Path to the root directory of the DTU dataset
ROOT_DIR="/home/dataset/mvs_training/dtu/"
CHECKPOINT_PATH="../tf_model_dtu/GRU/model.ckpt"

# Verify the directory structure
echo "Checking directories in $ROOT_DIR/Rectified/"
ls -l "$ROOT_DIR/Rectified/"


# Iterate over all folders inside the Rectified directory
for SCAN_DIR in $ROOT_DIR/Rectified/*/; do
    # Skip the dense directory if it exists within the Rectified directory
    if [ -d "$SCAN_DIR/mvsnet_output/" ]; then
        # continue

        ## UNDO WORK
        echo "Removing $SCAN_DIR/mvsnet_output/"
        rm -r "$SCAN_DIR/mvsnet_output/"
    fi
    # Check if the variable is indeed a directory
    if [[ -d "$SCAN_DIR" ]]; then
        echo "Processing $SCAN_DIR"
        
        # Run MVSNet for the current scan using the selected images
        python test.py \
            --dense_folder "$SCAN_DIR" \
            --regularization 'GRU' \
            --max_w 640 --max_h 512 --max_d 128 \
            --interval_scale 0.8 \
            --view_num 5 \
            --sample_scale 0.25 \
            --inverse_depth True \
            --adaptive_scaling True \
            --refinement True \
            --pretrained_model_ckpt_path "$CHECKPOINT_PATH"
    else
        echo "No directories found in $ROOT_DIR/Rectified/"
        exit 1
    fi
done
echo "MVSNet predictions saved for selected images in all scans."
