#!/bin/bash

# Set the path to your text file
file_path="file_paths.txt"

# Get the total number of rows in the file
total_rows=$(wc -l < "$file_path")

# Loop through indices from 1 to the total number of rows
for ((i = 1; i <= total_rows; i++)); do
    # Use sed to extract the line with the current index
    row=$(sed -n "${i}p" "$file_path")
    echo "Row $i: $row"
done
