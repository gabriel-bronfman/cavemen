#!/bin/bash

filename_string="gbb261_sg7761_submission"

# Get the current date and time
current_datetime=$(date +"%Y%m%d_%H%M%S")

# Define the folder to be zipped (current directory)
folder_to_zip="${filename_string}_${current_datetime}"

mkdir "../${folder_to_zip}"

cp -r "scripts/" "../${folder_to_zip}/scripts"
cp ".env" "../${folder_to_zip}/.env"
cp "arrow.py" "../${folder_to_zip}/arrow.py"
cp "place_recognition.py" "../${folder_to_zip}/place_recognition.py"
cp "player.py" "../${folder_to_zip}/player.py"
cp "plotter.py" "../${folder_to_zip}/plotter.py"
cp "utils.py" "../${folder_to_zip}/utils.py"
cp "environment.yml" "../${folder_to_zip}/environment.yml"
cp "README.md" "../${folder_to_zip}/README.md"
mkdir "../${folder_to_zip}/data"
mkdir "../${folder_to_zip}/data/save"
mkdir "../${folder_to_zip}/assets"
mkdir "../${folder_to_zip}/assets/img"
mkdir "../${folder_to_zip}/assets/img/arrows"
cp -r "data/save" "../${folder_to_zip}/data"
cp -r "assets/img/arrows" "../${folder_to_zip}/assets/img"
# Create the zip file with the specified name
zip -r "../${folder_to_zip}.zip" "../${folder_to_zip}"

rm -rf "data/save"
mkdir "data/save"

# Print a message indicating the completion of the zip process
echo "Folder '$folder_to_zip' has been zipped to '$folder_to_zip'.zip...exiting"
