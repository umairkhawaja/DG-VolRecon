#!/bin/bash

# Define the source files and directories
src_folder="/home/dataset/DTU_TEST/cameras/"
src_file="/home/dataset/DTU_TEST/pair.txt"
src_dir="/home/dataset/DTU_TEST/"
# Loop through all items in the current directory
for dir in "$src_dir"/*/ ; do
    echo "Processing $dir"
    # Check if the item is a directory and not the source folder
    if [ -d "$dir" ] && [ "$dir" != "$src_folder/" ]; then
        # Copy the folder and file to the target directory
        cp -r "$src_folder" "$dir"
        cp "$src_file" "$dir"
        # Select the first image of each pose (assumes ordered filenames)
        for i in $(seq 0 49); do  # Use seq as a compatible alternative to brace expansion
            PNG_NAME=$(printf "%06d.png" "$i")
            NEW_FILENAME=$(printf "%06d.jpg" "$i")
            
            # Check if the file exists before attempting to copy and convert it
            if [ -f "$dir/image/$PNG_NAME" ]; then
                # Use convert to change the image format from png to jpg and rename it
                convert "$dir/image/$PNG_NAME" "$dir/image/$NEW_FILENAME"
            else
                echo "File $dir/image/$PNG_NAME does not exist, skipping..."
            fi
        done
    fi
done
