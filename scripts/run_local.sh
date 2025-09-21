#!/bin/bash

# Activate the virtual environment
source .env/bin/activate

# Set environment variables
export CONFIG_PATH=src/config/qlora_config.yaml
export PEFT_CONFIG_PATH=src/config/peft_config.json
export LOGGING_CONFIG_PATH=src/config/logging.yaml

# Prepare the dataset
python src/data/prepare_dataset.py

# Run the training script
python src/training/run_qlora.py --config $CONFIG_PATH --peft_config $PEFT_CONFIG_PATH

# Run inference after training
python src/training/inference.py --config $CONFIG_PATH