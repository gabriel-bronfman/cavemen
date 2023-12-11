#!/bin/bash
cd ..

# Get the current script's directory
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the script's directory
cd "$script_dir"

# Define the folder to be zipped (current directory)
folder_to_zip="midterm_rp"


# Define a string to include in the filename
filename_string="gbb261_sg7761_submission"

# Get the current date and time
current_datetime=$(date +"%Y%m%d_%H%M%S")

# Create the zip file with the specified name
zip -r "${filename_string}_${current_datetime}.zip" "$folder_to_zip"

cd "midterm_rp/data/"
rm -rf "save"
mkdir "save"

# Print a message indicating the completion of the zip process
echo "Folder '$folder_to_zip' has been zipped to '$output_zip'."
