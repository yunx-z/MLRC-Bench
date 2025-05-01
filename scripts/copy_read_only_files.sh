#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <read_only_files.txt> <source_repo_path> <destination_repo_path>"
    exit 1
fi

# Assign arguments to variables
read_only_file_list=$1
source_repo_path=$2
destination_repo_path=$3

# Check if read_only_files.txt exists
if [ ! -f "$read_only_file_list" ]; then
    echo "copy_read_only_files.sh Error: $read_only_file_list not found!"
    exit 1
fi

# Iterate through each line in the file
while IFS= read -r file || [[ -n "$file" ]]; do
    # Skip empty lines or comments
    [[ -z "$file" || "$file" == \#* ]] && continue

    # Resolve source and destination paths
    source_path="$source_repo_path/$file"
    destination_path="$destination_repo_path/$file"

    # If the file is a directory (ends with /*)
    if [[ "$file" == */\* ]]; then
        # Remove trailing '/*' to get the directory path
        dir_path="${file%/*}"
        source_path="$source_repo_path/$dir_path"
        destination_path="$destination_repo_path/$dir_path"

        # Create the directory in the destination
        mkdir -p "$destination_path"

        # Copy .py files and create symbolic links for others
        for src_file in "$source_path"/*; do
            dest_file="$destination_path/$(basename "$src_file")"
            if [[ "$src_file" == *.py ]]; then
                cp "$src_file" "$dest_file"
            else
                ln -s "$(realpath "$src_file")" "$dest_file"
            fi
        done
    else
        # Ensure destination directory exists
        mkdir -p "$(dirname "$destination_path")"

        # Copy .py files and create symbolic links for others
        if [[ "$file" == *.py ]]; then
            cp "$source_path" "$destination_path"
        else
            ln -s "$(realpath "$source_path")" "$destination_path"
        fi
    fi

done < "$read_only_file_list"

