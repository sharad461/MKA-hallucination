#!/bin/bash

# Create base directory
BASE_DIR="MKA-$1"
mkdir -p $BASE_DIR

# Run the pipeline
echo "Starting the pipeline..."
python main.py $1 $2

echo "Pipeline completed!"