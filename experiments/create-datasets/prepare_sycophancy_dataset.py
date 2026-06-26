import os
import json
import random
import glob
from transformers import AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "./models/Qwen3.5-0.8B"
RAW_DATA_DIR = "./datasets/sycophancy/raw/"
OUT_DATA_DIR = "./datasets/sycophancy"

TOTAL_TARGET_TOKENS = 1_000_000

# Splits ratio (80% train, 10% val, 10% search)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# SEARCH_RATIO is implicitly the remaining 0.1
BENCHMARK_SIZE = 2000

random.seed(42)

def load_all_jsonl(directory):
    """Loads and combines all .jsonl files from the given directory."""
    all_data = []
    file_pattern = os.path.join(directory, "*.jsonl")
    files = glob.glob(file_pattern)
    
    print(f"Found {len(files)} files:")
    for file_path in files:
        print(f" - {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
                    
    return all_data

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print("Loading raw datasets...")
    all_examples = load_all_jsonl(RAW_DATA_DIR)
    print(f"Total examples loaded: {len(all_examples)}")
    
    # Shuffle to mix NLP, Philosophy, and Politics
    random.shuffle(all_examples)

    benchmark_split = all_examples[:BENCHMARK_SIZE]
    remaining_examples = all_examples[BENCHMARK_SIZE:]
    print(f"Reserved {BENCHMARK_SIZE} examples for the Benchmark.")
    
    selected_examples = []
    total_tokens = 0
    
    print(f"Selecting examples until {TOTAL_TARGET_TOKENS} tokens are reached...")
    pbar = tqdm(total=TOTAL_TARGET_TOKENS, desc="Counting Tokens")
    
    for item in remaining_examples:
        if total_tokens >= TOTAL_TARGET_TOKENS:
            break
            
        question = item["question"]
        # For token counting and training, we assume the model acts sycophantically
        matching_answer = item["answer_matching_behavior"]
        
        # Apply chat template to get the exact token count as it will be seen by the model
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + matching_answer
        
        # Count tokens
        tok_count = len(tokenizer.encode(full_text))
        
        selected_examples.append(item)
        total_tokens += tok_count
        pbar.update(tok_count)
        
    pbar.close()
    print(f"Selected {len(selected_examples)} examples containing {total_tokens} tokens.")
    
    # --- SPLITTING ---
    total_selected = len(selected_examples)
    train_end = int(total_selected * TRAIN_RATIO)
    val_end = train_end + int(total_selected * VAL_RATIO)
    
    train_split = selected_examples[:train_end]
    val_split = selected_examples[train_end:val_end]
    search_split = selected_examples[val_end:]
    
    print(f"Splits created -> Train: {len(train_split)}, Val: {len(val_split)}, Search: {len(search_split)} examples.")
    
    # --- SAVING ---
    def save_json(data, filename):
        path = os.path.join(OUT_DATA_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    save_json(train_split, "sycophancy_train.json")
    save_json(val_split, "sycophancy_val.json")
    save_json(search_split, "sycophancy_search.json")
    save_json(benchmark_split, "sycophancy_benchmark.json")
    
    print("All splits successfully saved!")

if __name__ == "__main__":
    main()