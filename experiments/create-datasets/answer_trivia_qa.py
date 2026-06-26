import json
import torch
import string
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

MODEL_PATH = "./models/Qwen3.5-0.8B"
DATASET_PATH = "./datasets/idk/"

# --- CONFIGURATION ---
N_GENERATIONS = 5
REFUSAL_RESPONSES = [
    "I don't know the answer to this question.",
    "I'm sorry, but I don't have the information to answer that.",
    "I am not sure about the answer to this question.",
    "This question is beyond my current knowledge base."
]

TOTAL_TARGET_TOKENS = 1_000_000
TARGET_KNOWN_TOKENS = TOTAL_TARGET_TOKENS // 2
TARGET_UNKNOWN_TOKENS = TOTAL_TARGET_TOKENS // 2

# Splits ratio (80% train, 10% val, 10% search)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# SEARCH_RATIO is implicitly the remaining 0.1

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)

dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="train", streaming=True)

def exact_match(prediction, ground_truths):
    normalized_pred = prediction.lower().translate(str.maketrans("", "", string.punctuation))
    for truth in ground_truths:
        normalized_truth = truth.lower().translate(str.maketrans("", "", string.punctuation))
        pattern = r'\b' + re.escape(normalized_truth) + r'\b'
        if re.search(pattern, normalized_pred):
            return True
    return False

known_examples = []
unknown_examples = []

known_tokens = 0
unknown_tokens = 0

print("Starting data collection in streaming mode...")
pbar = tqdm(total=TOTAL_TARGET_TOKENS, desc="Collecting Tokens")

for item in dataset:
    # Stop condition: both buckets are full
    if known_tokens >= TARGET_KNOWN_TOKENS and unknown_tokens >= TARGET_UNKNOWN_TOKENS:
        break

    question = item["question"]
    ground_truths = item["answer"]["aliases"]

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question directly and concisely."},
        {"role": "user", "content": question},
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=1.0,
            do_sample=True,
            num_return_sequences=N_GENERATIONS,
            pad_token_id=tokenizer.eos_token_id,
        )

    correct_count = 0
    responses = []
    
    for output in outputs:
        response_text = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        responses.append(response_text)
        if exact_match(response_text, ground_truths):
            correct_count += 1

    # Logic for classifying and counting tokens
    if correct_count == N_GENERATIONS and known_tokens < TARGET_KNOWN_TOKENS:
        # 5/5 Correct -> Model KNOWS the answer
        chosen_response = responses[0]
        full_text = prompt_text + chosen_response
        tok_count = len(tokenizer.encode(full_text))
        
        known_examples.append({
            "question": question,
            "answer": chosen_response,
            "label": "known"
        })
        known_tokens += tok_count
        pbar.update(tok_count)

    elif correct_count == 0 and unknown_tokens < TARGET_UNKNOWN_TOKENS:
        # 0/5 Correct -> Model DOES NOT KNOW the answer
        chosen_refusal = random.choice(REFUSAL_RESPONSES)
        
        full_text = prompt_text + chosen_refusal
        tok_count = len(tokenizer.encode(full_text))
        
        unknown_examples.append({
            "question": question,
            "answer": chosen_refusal,
            "label": "unknown"
        })
        unknown_tokens += tok_count
        pbar.update(tok_count)

pbar.close()

print(f"Collection finished! Known tokens: {known_tokens}, Unknown tokens: {unknown_tokens}")

all_examples = known_examples + unknown_examples
random.shuffle(all_examples)

total_examples = len(all_examples)
train_end = int(total_examples * TRAIN_RATIO)
val_end = train_end + int(total_examples * VAL_RATIO)

train_split = all_examples[:train_end]
val_split = all_examples[train_end:val_end]
search_split = all_examples[val_end:]

print(f"Splits created -> Train: {len(train_split)}, Val: {len(val_split)}, Search: {len(search_split)} examples.")

def save_json(data, filename):
    path = DATASET_PATH + filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

save_json(train_split, "idk_train.json")
save_json(val_split, "idk_val.json")
save_json(search_split, "idk_search.json")

print("All splits successfully saved!")