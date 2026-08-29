import os
import argparse
from huggingface_hub import snapshot_download
from datasets import load_dataset

def download_model(model_id, output_path):
    print(f"Downloading model: '{model_id}'...")
    # snapshot_download automatically uses HF_HUB_ENABLE_HF_TRANSFER if set in env
    snapshot_download(
        repo_id=model_id, 
        local_dir=output_path,
        local_dir_use_symlinks=False # Prevents symlink issues on some HPC file systems
    )
    print(f"Model successfully saved to: {output_path}\n")

def download_dataset(dataset_id, output_path):
    print(f"Downloading dataset: '{dataset_id}'...")
    dataset = load_dataset(dataset_id)
    dataset.save_to_disk(output_path)
    print(f"Dataset successfully saved to: {output_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Download models or datasets from Hugging Face.")
    
    # Mutually exclusive group: You must provide EITHER -mid OR -did
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-mid", "--model_id", type=str, help="Hugging Face Model ID (e.g., Qwen/Qwen2.5-9B-Instruct)")
    group.add_argument("-did", "--dataset_id", type=str, help="Hugging Face Dataset ID (e.g., locuslab/TOFU)")
    
    # Output path is always required
    parser.add_argument("-o", "--output", type=str, required=True, help="Local output directory path")
    
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)

    if args.model_id:
        download_model(args.model_id, args.output)
    elif args.dataset_id:
        download_dataset(args.dataset_id, args.output)

if __name__ == "__main__":
    main()