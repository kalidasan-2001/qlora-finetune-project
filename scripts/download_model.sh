#!/bin/bash

# Download the base model from Hugging Face or another source
MODEL_NAME="your_model_name_here"  # Replace with the desired model name
OUTPUT_DIR="./models"  # Directory to save the downloaded model

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Download the model
echo "Downloading model: $MODEL_NAME"
git lfs install
git clone https://huggingface.co/$MODEL_NAME $OUTPUT_DIR

echo "Model downloaded to $OUTPUT_DIR"