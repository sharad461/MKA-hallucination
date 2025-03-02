#!/bin/bash

BASE_DIR="MKA-$1"
mkdir -p $BASE_DIR

echo "Starting the pipeline..."
python main.py $1 $2

echo "Pipeline completed!"