#!/bin/bash

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    echo "gdown not found, installing..."
    pip install gdown
fi

# Google Drive folder ID
FOLDER_ID="19W7hXhkCDAU6h1YWEXXMcrQ_yP-967f6"

# Download the folder
echo "Downloading folder from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID"

echo "Download completed."