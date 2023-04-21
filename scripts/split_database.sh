#!/usr/bin/env bash
# ./scripts/split_database.sh ./maps_all/train ./maps/train 3
# ./scripts/split_database.sh ./maps_all/val ./maps/val 3

# Check if the required arguments are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <source_directory> <destination_directory> <number_of_chunks>"
    exit 1
fi

src_dir="$1"
dest_dir="$2"
num_chunks="$3"

# Check if the provided source directory exists
if [ ! -d "$src_dir" ]; then
    echo "Source directory $src_dir does not exist."
    exit 1
fi

# Validate the number of chunks
if ! [[ "$num_chunks" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid number of chunks. Please provide a positive integer."
    exit 1
fi

# Create the destination directory and chunk directories if they don't exist
for ((i = 1; i <= num_chunks; i++)); do
    mkdir -p "$dest_dir/site-$i"
done

# Iterate through the files in the provided source directory and split them into the specified number of chunks
count=0
for file in "$src_dir"/*; do
    if [ -f "$file" ]; then
        chunk_num=$((count % num_chunks + 1))
        cp "$file" "$dest_dir/site-$chunk_num"
        count=$((count + 1))
    fi
done

echo "Files have been split into $num_chunks chunks in the destination directory."
