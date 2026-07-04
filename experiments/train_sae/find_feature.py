import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from experiments.train_sae.train import TopKSAE
from experiments.train_sae.task_handlers import TASK_HANDLERS

def main():
    parser = argparse.ArgumentParser(description="Find the steering feature vector in the trained SAE.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM.")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the trained SAE checkpoint (.pt).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the search dataset JSON.")
    parser.add_argument("--task", type=str, choices=list(TASK_HANDLERS.keys()), required=True, help="Task type.")
    parser.add_argument("--layer_idx", type=int, required=True, help="Layer index where SAE was trained.")
    parser.add_argument("--exp_factor", type=int, required=True, help="Expansion factor used in SAE training.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the vector.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Tokenizer and LLM (Base Model)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModel.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    llm.eval()

    print(f"Loading SAE from {args.sae_path}...")
    d_model = llm.config.hidden_size 
    n_features = d_model * args.exp_factor
    
    sae = TopKSAE(d_model=d_model, n_features=n_features)
    sae.load_state_dict(torch.load(args.sae_path, map_location="cpu"))
    sae = sae.to(llm.device)
    sae.eval()

    print(f"Loading search dataset from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    pos_activations = []
    neg_activations = []

    handler = TASK_HANDLERS[args.task]()

    print(f"Processing {len(dataset)} examples for task: {args.task}...")
    
    def get_answer_activation(prompt_text, answer_text):
        """Helper function to extract activation at the first token of the answer."""
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        
        full_text = prompt_text + answer_text
        inputs = tokenizer([full_text], return_tensors="pt").to(llm.device)
        
        with torch.no_grad():
            outputs = llm(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[args.layer_idx]
        
        target_idx = min(prompt_len, hidden_states.size(1) - 1)
        return hidden_states[0, target_idx, :].unsqueeze(0)

    for item in tqdm(dataset, desc="Extracting Activations"):
        prompt, ans_pos, ans_neg = handler.process_item(item, tokenizer)
        
        if ans_pos is not None:
            h_pos = get_answer_activation(prompt, ans_pos)
            with torch.no_grad():
                _, f_pos = sae(h_pos.float(), k=20)
            pos_activations.append(f_pos.squeeze(0).cpu())
            
        if ans_neg is not None:
            h_neg = get_answer_activation(prompt, ans_neg)
            with torch.no_grad():
                _, f_neg = sae(h_neg.float(), k=20)
            neg_activations.append(f_neg.squeeze(0).cpu())

    pos_tensor = torch.stack(pos_activations) 
    neg_tensor = torch.stack(neg_activations) 

    mean_pos = pos_tensor.mean(dim=0)
    mean_neg = neg_tensor.mean(dim=0)
    diff = mean_pos - mean_neg

    top_values, top_indices = torch.topk(diff, k=5)

    print("\n" + "="*40)
    print(f" TOP FEATURES FOR: {args.task.upper()}")
    print("="*40)
    for i in range(5):
        idx = top_indices[i].item()
        val = top_values[i].item()
        print(f"Rank {i+1}: Feature {idx} (Delta: {val:.4f})")
    print("="*40)

    best_feature_idx = top_indices[0].item()
    steering_vector = sae.W_enc[:, best_feature_idx].detach().cpu()
    
    out_file = os.path.join(args.output_dir, f"steering_vec_{args.task}_idx_{best_feature_idx}.pt")
    torch.save(steering_vector, out_file)
    
    analysis_data = {
        "task": args.task,
        "best_feature_idx": best_feature_idx,
        "deltas": diff, 
        "best_feature_pos_acts": pos_tensor[:, best_feature_idx], 
        "best_feature_neg_acts": neg_tensor[:, best_feature_idx]  
    }
    analysis_file = os.path.join(args.output_dir, f"xai_data_{args.task}.pt")
    torch.save(analysis_data, analysis_file)

    print(f"\nSuccess! Steering vector saved to {out_file}")
    print(f"XAI Analysis data saved to {analysis_file}")

if __name__ == "__main__":
    main()
