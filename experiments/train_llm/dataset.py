import json
import yaml
from datasets import Dataset, load_from_disk

def load_and_prepare_dataset(data_path):
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)
    return load_from_disk(data_path)

def get_formatting_func(tokenizer, chat_template_path):
    with open(chat_template_path, 'r', encoding='utf-8') as f:
        chat_cfg = yaml.safe_load(f)
        
    def formatting_func(example):
        question = example.get("question", example.get("prompt", ""))
        answer = example.get("answer_matching_behavior", example.get("answer", ""))
        
        messages = [
            {"role": "system", "content": chat_cfg["system_prompt"]},
            {"role": chat_cfg["user_role"], "content": question},
            {"role": chat_cfg["assistant_role"], "content": answer}
        ]
        
        return tokenizer.apply_chat_template(messages, tokenize=False)
        
    return formatting_func, chat_cfg["response_template"]
