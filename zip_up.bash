#!/bin/bash
cd ..

# # Get the current script's directory
# script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# # Navigate to the script's directory
# cd "$script_dir"

filename_string="gbb261_sg7761_submission"

# Get the current date and time
current_datetime=$(date +"%Y%m%d_%H%M%S")

# Define the folder to be zipped (current directory)
folder_to_zip="${filename_string}_${current_datetime}"

mkdir "${filename_string}_${current_datetime}"

cp -r "midterm_rp/" "${filename_string}_${current_datetime}"

rm -rf "/${filename_string}_${current_datetime}/data/textures"



# Create the zip file with the specified name
zip -r "${filename_string}_${current_datetime}.zip" "$folder_to_zip"

cd "midterm_rp/data/"
rm -rf "save"
mkdir "save"

# Print a message indicating the completion of the zip process
echo "Folder '$folder_to_zip' has been zipped to '$output_zip'."
