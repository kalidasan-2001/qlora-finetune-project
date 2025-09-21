#!/bin/bash

# Activate the conda environment
source activate your_env_name

# Install required packages
pip install -r ../requirements.txt

# Run the training script
python ../src/training/run_qlora.py --config ../src/config/qlora_config.yaml

# Run inference after training
python ../src/training/inference.py --model_path ./output/model.bin --input_file ./input/prompts.txt --output_file ./output/responses.txt