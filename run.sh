#!/bin/bash

# Ensure the correct number of arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: bash run.sh /path/to/model-folder /path/to/adapter_checkpoint /path/to/input.json /path/to/output.json"
    exit 1
fi

# Arguments
MODEL_FOLDER=$1
ADAPTER_CHECKPOINT=$2
INPUT_FILE=$3
OUTPUT_FILE=$4

# Running inference with specified paths
echo "Running inference with the following parameters:"
echo "Model Folder: $MODEL_FOLDER"
echo "Adapter Checkpoint: $ADAPTER_CHECKPOINT"
echo "Input File: $INPUT_FILE"
echo "Output File: $OUTPUT_FILE"

python3 inference.py \
    --base_model_path "$MODEL_FOLDER" \
    --peft_path "$ADAPTER_CHECKPOINT" \
    --test_data_path "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

echo "Inference completed. Predictions saved to $OUTPUT_FILE."