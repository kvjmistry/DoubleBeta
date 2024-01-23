#!/bin/bash

# Set the directory path
directory_path="../../data/nexus/LPR_Tl208_Ports/PORT_1a/sophronia/"

# Set the output file path
output_file="file_paths.txt"

# Use find to get a list of full file paths and save to the text file
find "$directory_path" -type f > "$output_file"

echo "File paths saved to $output_file"
