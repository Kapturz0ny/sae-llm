import os
import json
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "./models/Qwen3.5-0.8B"
DATA_DIR = "./datasets/"
OUT_DATA_PATH = os.path.join(DATA_DIR, "sae_training_data")

def load_json(filepath):
    """Loads a JSON file given a path relative to DATA_DIR."""
    path = os.path.join(DATA_DIR, filepath)
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_split_stats(texts, tokenizer, split_name, sources_dict):
    """Calculates token statistics for a given split."""
    print(f"\nCalculating token statistics for {split_name} split...")
    
    token_lengths = []
    batch_size = 1000
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Tokenizing {split_name}"):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, add_special_tokens=False)
        token_lengths.extend([len(ids) for ids in encodings["input_ids"]])
        
    token_lengths = np.array(token_lengths)
    
    stats = {
        "total_examples": int(len(token_lengths)),
        "total_tokens": int(np.sum(token_lengths)),
        "avg_tokens_per_example": float(np.mean(token_lengths)),
        "min_tokens": int(np.min(token_lengths)),
        "max_tokens": int(np.max(token_lengths)),
        "percentiles": {
            "p50": int(np.percentile(token_lengths, 50)),
            "p90": int(np.percentile(token_lengths, 90)),
            "p95": int(np.percentile(token_lengths, 95)),
            "p99": int(np.percentile(token_lengths, 99)),
            "p99_9": int(np.percentile(token_lengths, 99.9))
        },
        "sources_breakdown": sources_dict
    }
    return stats

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    combined_train_texts = []
    combined_val_texts = []
    
    # Dictionaries to keep track of where the data came from
    train_sources = {}
    val_sources = {}

    # ==========================================
    # 1. OPENORCA
    # ==========================================
    print("\n--- Processing OpenOrca ---")
    orca_train_path = os.path.join(DATA_DIR, "openorca", "train")
    orca_val_path = os.path.join(DATA_DIR, "openorca", "val")
    
    print(f"Loading OpenOrca from {orca_train_path} and {orca_val_path}...")
    orca_train = load_from_disk(orca_train_path)
    orca_val = load_from_disk(orca_val_path)
    
    combined_train_texts.extend(orca_train["text"])
    combined_val_texts.extend(orca_val["text"])
    
    train_sources["openorca"] = len(orca_train)
    val_sources["openorca"] = len(orca_val)
    
    print(f"Added {len(orca_train)} train and {len(orca_val)} val examples from OpenOrca.")

    # ==========================================
    # 2. IDK (JSON)
    # ==========================================
    print("\n--- Processing IDK ---")
    idk_train_raw = load_json("idk/idk_train.json")
    idk_val_raw = load_json("idk/idk_val.json")
    
    def process_idk(raw_data):
        texts = []
        for item in raw_data:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question directly and concisely."},
                {"role": "user", "content": item["question"]}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = prompt + item["answer"]
            texts.append(full_text)
        return texts

    idk_train_texts = process_idk(idk_train_raw)
    idk_val_texts = process_idk(idk_val_raw)
    
    combined_train_texts.extend(idk_train_texts)
    combined_val_texts.extend(idk_val_texts)
    
    train_sources["idk"] = len(idk_train_texts)
    val_sources["idk"] = len(idk_val_texts)
    
    print(f"Added {len(idk_train_texts)} train and {len(idk_val_texts)} val examples from IDK.")

    # ==========================================
    # 3. SYCOPHANCY
    # ==========================================
    print("\n--- Processing Sycophancy ---")
    sycophancy_train_raw = load_json("sycophancy/sycophancy_train.json")
    sycophancy_val_raw = load_json("sycophancy/sycophancy_val.json")
    
    def process_sycophancy(raw_data):
        texts = []
        for item in raw_data:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item["question"]}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # For training, we force the model to see the sycophantic behavior
            full_text = prompt + item["answer_matching_behavior"]
            texts.append(full_text)
        return texts

    sycophancy_train_texts = process_sycophancy(sycophancy_train_raw)
    sycophancy_val_texts = process_sycophancy(sycophancy_val_raw)
    
    combined_train_texts.extend(sycophancy_train_texts)
    combined_val_texts.extend(sycophancy_val_texts)
    
    train_sources["sycophancy"] = len(sycophancy_train_texts)
    val_sources["sycophancy"] = len(sycophancy_val_texts)
    
    print(f"Added {len(sycophancy_train_texts)} train and {len(sycophancy_val_texts)} val examples from Sycophancy.")

    # ==========================================
    # 4. CALCULATE STATISTICS
    # ==========================================
    dataset_stats = {
        "train": calculate_split_stats(combined_train_texts, tokenizer, "Train", train_sources),
        "val": calculate_split_stats(combined_val_texts, tokenizer, "Val", val_sources)
    }

    # ==========================================
    # 5. COMBINE, SHUFFLE AND SAVE (ARROW)
    # ==========================================
    print("\n--- Creating Final Dataset ---")
    
    train_dataset = Dataset.from_dict({"text": combined_train_texts})
    val_dataset = Dataset.from_dict({"text": combined_val_texts})
    
    print("Shuffling train and val splits...")
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)
    
    final_dataset = DatasetDict({
        "train": train_dataset,
        "val": val_dataset
    })
    
    print(f"\nSaving final dataset to {OUT_DATA_PATH} in Arrow format...")
    final_dataset.save_to_disk(OUT_DATA_PATH)
    
    stats_path = os.path.join(OUT_DATA_PATH, "stats.json")
    print(f"Saving statistics to {stats_path}...")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=4, ensure_ascii=False)
    
    print("\nDone! Your SAE training dataset and statistics are ready.")
    print(f"Total Train Tokens: {dataset_stats['train']['total_tokens']:,}")
    print(f"Total Val Tokens: {dataset_stats['val']['total_tokens']:,}")

if __name__ == "__main__":
    main()