import os
import json
import string
import re
import random
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
MODEL_PATH = "./models/Qwen3.5-0.8B"
DATASET_PATH = "./datasets/idk/"

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
BATCH_SIZE = 500

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)

def save_json(data, filename):
    os.makedirs(DATASET_PATH, exist_ok=True)
    path = DATASET_PATH + filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    print("Initializing Tokenizer and LLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    REFUSAL_TOKEN_COUNTS = {
        refusal: len(tokenizer.encode(refusal)) 
        for refusal in REFUSAL_RESPONSES
    }

    llm = LLM(model=MODEL_PATH, dtype="half", gpu_memory_utilization=0.9, max_model_len=2048)

    sampling_params = SamplingParams(
        n=N_GENERATIONS,
        temperature=1.0,
        max_tokens=30,
    )

    dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="train", streaming=True)
    dataset_iter = iter(dataset)

    known_examples = []
    unknown_examples = []
    known_tokens = 0
    unknown_tokens = 0

    print("Starting data collection with vLLM in batched mode...")
    
    TRIVIA_QA_TRAIN_SIZE = 138384
    
    pbar_processed = tqdm(total=TRIVIA_QA_TRAIN_SIZE, desc="Processed Questions", position=0, leave=True, unit="q")
    pbar_known = tqdm(total=TARGET_KNOWN_TOKENS, desc="Known Tokens       ", position=1, leave=True, unit="tok")
    pbar_unknown = tqdm(total=TARGET_UNKNOWN_TOKENS, desc="Unknown Tokens     ", position=2, leave=True, unit="tok")

    while known_tokens < TARGET_KNOWN_TOKENS or unknown_tokens < TARGET_UNKNOWN_TOKENS:
        batch_items = []
        try:
            for _ in range(BATCH_SIZE):
                batch_items.append(next(dataset_iter))
        except StopIteration:
            pass 
            
        if not batch_items:
            break
            
        pbar_processed.update(len(batch_items))
            
        prompts = []
        for item in batch_items:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question directly and concisely."},
                {"role": "user", "content": item["question"]},
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for i, output in enumerate(outputs):
            if known_tokens >= TARGET_KNOWN_TOKENS and unknown_tokens >= TARGET_UNKNOWN_TOKENS:
                break
                
            item = batch_items[i]
            question = item["question"]
            raw_ground_truths = item["answer"]["aliases"]
            prompt_tok_count = len(output.prompt_token_ids)
            
            compiled_truths = []
            for truth in raw_ground_truths:
                normalized_truth = truth.lower().translate(PUNCT_TRANSLATOR)
                pattern = re.compile(r'\b' + re.escape(normalized_truth) + r'\b')
                compiled_truths.append(pattern)
                
            correct_count = 0
            responses_data = []
            
            for gen in output.outputs:
                resp_text = gen.text.strip()
                responses_data.append((resp_text, len(gen.token_ids)))
                
                normalized_pred = resp_text.lower().translate(PUNCT_TRANSLATOR)
                
                for pattern in compiled_truths:
                    if pattern.search(normalized_pred):
                        correct_count += 1
                        break

            if correct_count >= 4 and known_tokens < TARGET_KNOWN_TOKENS:
                chosen_response, resp_tok_count = responses_data[0]
                total_toks = prompt_tok_count + resp_tok_count
                
                known_examples.append({
                    "question": question,
                    "answer": chosen_response,
                    "label": "known"
                })
                known_tokens += total_toks
                pbar_known.update(total_toks)

            elif correct_count == 0 and unknown_tokens < TARGET_UNKNOWN_TOKENS:
                chosen_refusal = random.choice(REFUSAL_RESPONSES)
                total_toks = prompt_tok_count + REFUSAL_TOKEN_COUNTS[chosen_refusal]
                
                unknown_examples.append({
                    "question": question,
                    "answer": chosen_refusal,
                    "label": "unknown"
                })
                unknown_tokens += total_toks
                pbar_unknown.update(total_toks)

    pbar_processed.close()
    pbar_known.close()
    pbar_unknown.close()
    
    print("\n")
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

    save_json(train_split, "idk_train.json")
    save_json(val_split, "idk_val.json")
    save_json(search_split, "idk_search.json")

    print("All splits successfully saved!")

if __name__ == '__main__':
    main()