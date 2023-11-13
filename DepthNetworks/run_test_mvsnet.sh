#!/bin/bash

cd "MVSNet/mvsnet"

# Path to the root directory of the DTU dataset
# ROOT_DIR="/home/dataset/mvs_training/dtu/Rectified/" # train/val
ROOT_DIR="/home/dataset/DTU_TEST/" # test
CHECKPOINT_PATH="../tf_model_dtu/3DCNNs/model.ckpt"

# Verify the directory structure
echo "Checking directories in $ROOT_DIR/"
ls -l "$ROOT_DIR/"


# Iterate over all folders inside the Rectified directory
for SCAN_DIR in $ROOT_DIR*/; do
    if [[ "$SCAN_DIR" == *"/cameras/"* ]]; then
        echo "Skipping $SCAN_DIR"
        continue
    fi

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
        
        # Run MVSNet TEST for the current scan using the selected images
        ## --max_w 1152 --max_h 864 --max_d 192 --interval_scale 1.06
        python test.py \
            --pretrained_model_ckpt_path "$CHECKPOINT_PATH" \
            --dense_folder "$SCAN_DIR" \
            --regularization '3DCNNs' \
            --max_w 640 --max_h 512 --max_d 128 \
            --interval_scale 1.0 \
            --view_num 3 \
            --sample_scale 0.25 \
            --adaptive_scaling True
    
    else
        echo "No directories found in $ROOT_DIR/"
        exit 1
    fi
done
echo "MVSNet predictions saved for selected images in all scans."
