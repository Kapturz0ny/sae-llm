import os
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "./models/Qwen3.5-0.8B" 
DATASET_PATH = "./datasets/OpenOrca-raw"
OUT_DATA_DIR = "./datasets/openorca"

TOTAL_TARGET_TOKENS = 38_000_000

# Splits ratio (90% train, 10% val)
TRAIN_RATIO = 0.9

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print(f"Loading OpenOrca dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    
    if "train" in dataset:
        dataset = dataset["train"]
        
    print(f"Total examples available in OpenOrca: {len(dataset)}")
    
    print("Shuffling dataset...")
    dataset = dataset.shuffle(seed=42)
    
    selected_data = {
        "text": [],
        "token_count": []
    }
    
    total_tokens = 0
    
    print(f"Selecting examples until {TOTAL_TARGET_TOKENS} tokens are reached...")
    pbar = tqdm(total=TOTAL_TARGET_TOKENS, desc="Counting Tokens")
    
    for row in dataset:
        if total_tokens >= TOTAL_TARGET_TOKENS:
            break
            
        sys_prompt = row.get("system_prompt", "")
        user_prompt = row.get("question", "")
        assistant_resp = row.get("response", "")
        
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": assistant_resp})
        
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Count tokens
        tok_count = len(tokenizer.encode(full_text))
        
        selected_data["text"].append(full_text)
        selected_data["token_count"].append(tok_count)
        
        total_tokens += tok_count
        pbar.update(tok_count)
        
    pbar.close()
    print(f"Selected {len(selected_data['text'])} examples containing {total_tokens} tokens.")
    
    print("Converting to Hugging Face Dataset format...")
    hf_dataset = Dataset.from_dict(selected_data)
    
    print("Splitting into Train and Val...")
    splits = hf_dataset.train_test_split(test_size=(1.0 - TRAIN_RATIO), seed=42)
    
    train_path = os.path.join(OUT_DATA_DIR, "openorca_train")
    val_path = os.path.join(OUT_DATA_DIR, "openorca_val")
    
    print(f"Saving train split to {train_path} (Arrow format)...")
    splits["train"].save_to_disk(train_path)
    
    print(f"Saving val split to {val_path} (Arrow format)...")
    splits["test"].save_to_disk(val_path)
    
    print("All OpenOrca splits successfully saved in Arrow format!")

if __name__ == "__main__":
    main()