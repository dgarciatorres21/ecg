#!/bin/bash

# This script runs the UNET demo locally.
# It assumes you have the necessary python environment and nnU-Net installed.

# Activate your python environment if needed
# source /path/to/your/conda/or/virtual/env/bin/activate

# Set the paths to the python script and data directories
PYTHON_SCRIPT="../code-unet/create_demo.py"
INPUT_DIR="./data/unet/input"
OUTPUT_DIR="./data/unet/output"
DATASET_ID=5 # Example dataset ID

# Run the demo
python "$PYTHON_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_id "$DATASET_ID"

echo "UNET demo finished. Check the output in the $OUTPUT_DIR directory."
