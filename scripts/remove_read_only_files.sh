#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <read_only_files.txt> <experimentation_repo_path>"
    exit 1
fi

# Assign arguments to variables
read_only_file_list=$1
experimentation_repo_path=$2

# Check if read_only_files.txt exists
if [ ! -f "$read_only_file_list" ]; then
    echo "remove_read_only_files.sh Error: $read_only_file_list not found!"
    exit 1
fi

# Iterate through each line in the file
# handle last line more robustly
while IFS= read -r file || [[ -n "$file" ]]; do 
    # Skip empty lines or comments
    [[ -z "$file" || "$file" == \#* ]] && continue

    # Resolve the full path of the file or directory to remove
    target_path="$experimentation_repo_path/$file"

    # If the file is a directory (ends with /*)
    if [[ "$file" == */\* ]]; then
        # Remove the directory path (without the trailing "/*")
        dir_path="${target_path%/*}"
        if [ -d "$dir_path" ]; then
            # echo "Removing directory: $dir_path"
            rm -rf "$dir_path"
        else
            echo "Directory not found: $dir_path"
        fi
    else
        # Remove the individual file
        if [ -f "$target_path" ]; then
            # echo "Removing file: $target_path"
            rm -f "$target_path"
        elif [ -d "$target_path" ]; then
            # echo "Removing directory: $target_path"
            rm -rf "$target_path"
        else
            echo "File or directory not found: $target_path"
        fi
    fi
done < "$read_only_file_list"

# echo "Read-only files and directories have been successfully removed from $experimentation_repo_path."

