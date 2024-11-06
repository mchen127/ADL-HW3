#!/bin/bash

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found. Please install it by running: pip install gdown"
    exit 1
fi

# Download adapter and tokenizer from Google Drive
FILE_ID="11qu4SrHmHoM2SBLQwoodbcd88ZziI3Vp"
DEST_FOLDER="adapter_checkpoint"

# Create directory to store downloaded files
mkdir -p $DEST_FOLDER

# Download files using gdown (Google Drive downloader)
gdown --folder "https://drive.google.com/drive/folders/$FILE_ID" -O $DEST_FOLDER


if [ $? -eq 0 ]; then
    # Give execute permission to the downloaded files (if needed)
    chmod +x $DEST_FOLDER/*
    echo "Download completed. Files are stored in '$DEST_FOLDER'."
else
    echo "Download failed. Please check the file ID or your network connection."
    exit 1
fi