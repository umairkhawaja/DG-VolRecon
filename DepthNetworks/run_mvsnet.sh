#!/bin/bash

## REQUIRES: sudo apt-get install imagemagick

cd "MVSNet/mvsnet"

# Path to the root directory of the DTU dataset
ROOT_DIR="/home/dataset/mvs_training/dtu/"
CHECKPOINT_PATH="/home/umair/VolReconWithDepthPriors/DepthNetworks/MVSNet/tf_model_dtu/3DCNNs/model.ckpt"
PAIR_PATH="/home/dataset/mvs_training/dtu/Cameras/pair.txt"

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
    # continue
    # Check if the variable is indeed a directory
    if [[ -d "$SCAN_DIR" ]]; then
        echo "Processing $SCAN_DIR"
        
        
        # The basename of the scan directory
        BASENAME=$(basename "$SCAN_DIR")

        # Define the path for the dense workspace directory
        WORKSPACE_PATH="$SCAN_DIR/mvsnet_output"
        
        # Create a workspace directory for the current scan
        mkdir -p "$WORKSPACE_PATH"

        # Create a temporary directory to store the selected images
        TMP_IMAGE_PATH="$WORKSPACE_PATH/images"
        mkdir -p "$TMP_IMAGE_PATH"

        # MVSNet test.py expects a pair.txt as well
        cp "$PAIR_PATH" "$WORKSPACE_PATH"

        ## MVSNet test.py expects a folder of cams as well
        CAMS_DIR="$WORKSPACE_PATH/cams/"
        mkdir "$CAMS_DIR"
        cp "$ROOT_DIR/Cameras/"* "$CAMS_DIR"
        rm "$WORKSPACE_PATH/cams/pair.txt"

        # Select the first image of each pose (assumes ordered filenames)
        for i in $(seq 1 49); do  # Use seq as a compatible alternative to brace expansion
            # Construct the file pattern to match the first image of each pose
            FILE_PATTERN=$(printf "rect_%03d_0_r5000.png" "$i")
            
            # Construct the new filename with leading zeros and a .jpg extension
            j=$((i-1))
            NEW_FILENAME=$(printf "%08d.jpg" "$j")
            
            # Check if the file exists before attempting to copy and convert it
            if [ -f "$SCAN_DIR/$FILE_PATTERN" ]; then
                # Use convert to change the image format from png to jpg and rename it
                convert "$SCAN_DIR/$FILE_PATTERN" "$TMP_IMAGE_PATH/$NEW_FILENAME"
            else
                echo "File $SCAN_DIR/$FILE_PATTERN does not exist, skipping..."
            fi
        done
        
        # Run MVSNet for the current scan using the selected images
        python test.py \
            --dense_folder "$WORKSPACE_PATH" \
            --regularization '3DCNNs' \
            --max_w 640 --max_h 512 --max_d 128 \
            --interval_scale 1.06 \
            --pretrained_model_ckpt_path "$CHECKPOINT_PATH"

        # Clean up the temporary image directory
        rm -rf "$TMP_IMAGE_PATH" "$CAMS_DIR"

        mv "$WORKSPACE_PATH/depths_mvsnet/"* "$WORKSPACE_PATH"
        rm -r "$WORKSPACE_PATH/depths_mvsnet/"
        rm "$WORKSPACE_PATH/pair.txt"
    else
        echo "No directories found in $ROOT_DIR/Rectified/"
        exit 1
    fi
done

echo "MVSNet predictions saved for selected images in all scans."
