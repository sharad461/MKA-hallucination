#!/bin/bash

# Set up environment
echo "Setting up environment..."
pip install bitsandbytes accelerate datasets vllm sglang[all]>=0.4.2.post2 ctranslate2 sentence-transformers --quiet
pip install --upgrade pip
pip install sgl-kernel --force-reinstall --no-deps
pip install "sglang[all]>=0.4.2.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/

# Create base directory
BASE_DIR="MKA-SG-50"
mkdir -p $BASE_DIR

# Run the pipeline
echo "Starting the pipeline..."
python new_main.py

echo "Pipeline completed!"