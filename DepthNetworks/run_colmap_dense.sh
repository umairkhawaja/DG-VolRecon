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
    if [ -d "$SCAN_DIR/colmap_output/" ]; then
        continue

        ## TO UNDO OUTPUTS
        # echo "Removing $SCAN_DIR/dense/"
        # rm -r "$SCAN_DIR/dense/"
    fi
    # Check if the variable is indeed a directory
    if [[ -d "$SCAN_DIR" ]]; then
        echo "Processing $SCAN_DIR"
        
        # The basename of the scan directory
        BASENAME=$(basename "$SCAN_DIR")

        # Define the path for the dense workspace directory
        WORKSPACE_PATH="$SCAN_DIR/colmap_output"
        
        # Create a workspace directory for the current scan
        mkdir -p "$WORKSPACE_PATH"

        # Create a temporary directory to store the selected images
        TMP_IMAGE_PATH="$WORKSPACE_PATH/select_images"
        mkdir -p "$TMP_IMAGE_PATH"

        # Select the first image of each pose (assumes ordered filenames)
        for i in $(seq 0 49); do  # Use seq as a compatible alternative to brace expansion
            # Construct the file pattern to match the first image of each pose
            FILE_PATTERN=$(printf "rect_%03d_0_r5000.png" "$i")
            
            # Copy the first image of the current pose to the temporary directory
            cp "$SCAN_DIR/$FILE_PATTERN" "$TMP_IMAGE_PATH" 2>/dev/null
        done

        # Run dense reconstruction for the current scan using the selected images
        $COLMAP automatic_reconstructor \
            --workspace_path "$WORKSPACE_PATH" \
            --image_path "$TMP_IMAGE_PATH" \

        # Clean up the temporary image directory
        rm -rf "$TMP_IMAGE_PATH"
    else
        echo "No directories found in $ROOT_DIR/Rectified/"
        exit 1
    fi
done

echo "Dense reconstruction completed for selected images in all scans."
