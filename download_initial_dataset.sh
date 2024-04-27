#!/usr/bin/env bash
set -e # Exit on error

DATASET_URL="https://miplib.zib.de/downloads/collection.zip"

# Download the dataset
wget -O collection.zip $DATASET_URL

# Unzip the dataset in a folder called ./input
unzip collection.zip -d input

echo "You may remove the collection.zip file now."
