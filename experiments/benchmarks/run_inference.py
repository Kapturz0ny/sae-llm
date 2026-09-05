import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.benchmarks.utils import attach_steering_vector

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference with optional system prompt and steering vector.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON file.")
    
    # Prompts passed directly from the command line
    parser.add_argument("--prompts", type=str, nargs='+', required=True, help="List of prompts to evaluate.")
    
    # Optional steering parameters (default to "none")
    parser.add_argument("--vector_path", type=str, default="none", help="Path to the steering vector (.pt).")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index for the steering vector.")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Multiplier for the steering vector.")
    
    # Optional system prompt
    parser.add_argument("--system_prompt", type=str, default="none", help="System prompt to prepend to all queries.")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum tokens to generate.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Loading Tokenizer and Model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    # Attach steering vector if provided
    hook_handle = None
    if args.vector_path.lower() != "none":
        hook_handle = attach_steering_vector(model, args.vector_path, args.layer_idx, args.multiplier)

    formatted_prompts = []
    
    # Format prompts
    for user_text in args.prompts:
        messages = []
        if args.system_prompt.lower() != "none":
            messages.append({"role": "system", "content": args.system_prompt})
            
        messages.append({"role": "user", "content": user_text})
        
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted_prompt)

    results = []
    print(f"Generating responses for {len(formatted_prompts)} queries...")
    
    for i in tqdm(range(0, len(formatted_prompts), args.batch_size), desc="Inference"):
        batch_prompts = formatted_prompts[i:i + args.batch_size]
        batch_original_texts = args.prompts[i:i + args.batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        for j, output in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            
            results.append({
                "prompt": batch_original_texts[j],
                "model_response": response
            })

    # Cleanup hook
    if hook_handle is not None:
        hook_handle.remove()

    print(f"Saving results to {args.output_path}...")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Inference complete!")

if __name__ == "__main__":
    main()