#!/bin/bash

SOURCE_BASE="./"
DEST_BASE="$HOME/Library/Mobile Documents/com~apple~CloudDocs/LEARNING"

# Loop through all subdirectories in the source base
for src_dir in "$SOURCE_BASE"/*; do
    [ -d "$src_dir" ] || continue  # skip non-directories

    # Extract subfolder name (e.g., ML, math)
    folder_name=$(basename "$src_dir")
    dest_dir="$DEST_BASE/$folder_name"

    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"

    # Copy PDF files (preserve timestamps and metadata)
    find "$src_dir" -type f -iname "*.pdf" -exec cp -p {} "$dest_dir" \;

    echo "Copied PDFs from $src_dir to $dest_dir"
done
