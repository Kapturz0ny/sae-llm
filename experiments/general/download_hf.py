from huggingface_hub import snapshot_download
from datasets import load_dataset

MODELS_PATH = "./models/Qwen3.5-0.8B"
DATASETS_PATH = "./datasets/OpenOrca"

def download_from_hf():
    model_id = "Qwen/Qwen3.5-0.8B" 
    print(f"Downloading model: {model_id}...")
    snapshot_download(
        repo_id=model_id, 
        local_dir=MODELS_PATH
    )
    print(f"Model saved to: {MODELS_PATH}\n")

    dataset_id = "Open-Orca/OpenOrca" 
    print(f"Downloading dataset: {dataset_id}...")
    dataset = load_dataset(dataset_id)
    dataset.save_to_disk(DATASETS_PATH)
    print(f"Dataset saved to: {DATASETS_PATH}")

if __name__ == "__main__":
    download_from_hf()