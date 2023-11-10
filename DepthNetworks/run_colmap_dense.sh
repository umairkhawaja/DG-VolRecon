#!/bin/bash

# Path to the root directory of the DTU dataset
ROOT_DIR="/dataset/mvs_training/dtu/"

# Verify the directory structure
echo "Checking directories in $ROOT_DIR/Rectified/"
ls -l "$ROOT_DIR/Rectified/"

# Path to the COLMAP executable
COLMAP="/usr/local/bin/colmap"

# Iterate over all folders inside the Rectified directory
for SCAN_DIR in $ROOT_DIR/Rectified/*/; do
    # Skip the dense directory if it exists within the Rectified directory
    if [ -d "$SCAN_DIR/colmap_output/dense/" ]; then
        continue

        ## TO UNDO OUTPUTS
        # echo "Removing $SCAN_DIR/colmap_output/"
        # rm -r "$SCAN_DIR/colmap_output/"
    fi

    # Check if the variable is indeed a directory
    if [[ -d "$SCAN_DIR" ]]; then
        echo "Processing $SCAN_DIR"
        
        # Define the path for the dense workspace directory
        WORKSPACE_PATH="$SCAN_DIR/colmap_output"
        # Create a workspace directory for the current scan
        mkdir -p "$WORKSPACE_PATH"

        # Run dense reconstruction for the current scan using the selected images
        $COLMAP automatic_reconstructor \
            --workspace_path "$WORKSPACE_PATH" \
            --image_path "$SCAN_DIR/images/" \

    else
        echo "No directories found in $ROOT_DIR/Rectified/"
        exit 1
    fi
done

echo "Dense reconstruction completed for images in all scans."
