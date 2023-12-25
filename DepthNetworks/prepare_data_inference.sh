#!/bin/bash

## REQUIRES: sudo apt-get install imagemagick

# Path to the root directory of the DTU dataset
ROOT_DIR="/home/dataset/mvs_training/dtu/"
PAIR_PATH="$ROOT_DIR/Cameras/pair.txt"

# Verify the directory structure
echo "Checking directories in $ROOT_DIR/Rectified/"
ls -l "$ROOT_DIR/Rectified/"


# Iterate over all folders inside the Rectified directory
for SCAN_DIR in $ROOT_DIR/Rectified/*/; do
    # Skip the dense directory if it exists within the Rectified directory
    # if [ -d "$SCAN_DIR/images/" ]; then
        # continue
        # rm -r "$SCAN_DIR/images/"
    # fi

    # Check if the variable is indeed a directory
    if [[ -d "$SCAN_DIR" ]]; then
        echo "Processing $SCAN_DIR"

        # The basename of the scan directory
        BASENAME=$(basename "$SCAN_DIR")

        # Create a temporary directory to store the selected images
        TMP_IMAGE_PATH="$SCAN_DIR/images"
        mkdir -p "$TMP_IMAGE_PATH"

        # Copy pair.txt as well
        cp "$PAIR_PATH" "$SCAN_DIR"

        ## Copy Cameras as well
        CAMS_DIR="$SCAN_DIR/cams/"
        mkdir "$CAMS_DIR"
        cp "$ROOT_DIR/Cameras/train/"* "$CAMS_DIR"

        # Select the first image of each pose (assumes ordered filenames)
        for i in $(seq 0 49); do  # Use seq as a compatible alternative to brace expansion
            # Construct the file pattern to match the first image of each pose
            FILE_PATTERN=$(printf "rect_%03d_0_r5000.png" "$i")
            
            # Construct the new filename with leading zeros and a .jpg extension
            j=$((i-1))
            NEW_FILENAME=$(printf "%08d.jpg" "$i")
            
            # Check if the file exists before attempting to copy and convert it
            if [ -f "$SCAN_DIR/$FILE_PATTERN" ]; then
                # Use convert to change the image format from png to jpg and rename it
                convert "$SCAN_DIR/$FILE_PATTERN" "$TMP_IMAGE_PATH/$NEW_FILENAME"
            else
                echo "File $SCAN_DIR/$FILE_PATTERN does not exist, skipping..."
            fi
        done
    else
        echo "No directories found in $ROOT_DIR/Rectified/"
        exit 1
    fi
done

echo "Pre-processed the training data to match testing inference directory structure"
