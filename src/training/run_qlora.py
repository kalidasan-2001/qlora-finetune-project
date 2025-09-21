import os
import yaml
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator

# Load configuration
with open(os.path.join(os.path.dirname(__file__), '../config/qlora_config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

with open(os.path.join(os.path.dirname(__file__), '../config/peft_config.json'), 'r') as file:
    peft_config = json.load(file)

# Set up logging
logging.basicConfig(level=config['logging']['level'], format=config['logging']['format'])
logger = logging.getLogger(__name__)

def main():
    # Initialize accelerator
    accelerator = Accelerator()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Prepare dataset
    dataset = load_dataset('json', data_files=config['dataset_path'])
    
    # Training loop
    model.train()
    for epoch in range(config['num_epochs']):
        for batch in dataset['train']:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            logger.info(f'Epoch: {epoch}, Loss: {loss.item()}')

    # Save the model
    model.save_pretrained(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])

if __name__ == "__main__":
    main()