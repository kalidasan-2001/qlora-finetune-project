def prepare_dataset(input_file, output_file):
    import pandas as pd

    # Load the dataset
    df = pd.read_json(input_file, lines=True)

    # Perform any necessary preprocessing
    # Example: Remove rows with missing values
    df.dropna(inplace=True)

    # Save the prepared dataset to the output file
    df.to_json(output_file, orient='records', lines=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for QLoRA training.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the prepared dataset.")
    
    args = parser.parse_args()
    
    prepare_dataset(args.input_file, args.output_file)