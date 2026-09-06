import os
import json
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from experiments.train_sae.train import TopKSAE
from experiments.train_sae.task_handlers import TASK_HANDLERS


def set_seed(seed: int):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OffTheShelfSAE(torch.nn.Module):
    """Wrapper for the official Qwen/Gemma off-the-shelf SAEs."""
    def __init__(self, W_enc, b_enc):
        super().__init__()
        self.W_enc = torch.nn.Parameter(W_enc)
        self.b_enc = torch.nn.Parameter(b_enc)

    def forward(self, x, k):
        pre_acts = torch.matmul(x, self.W_enc.T) + self.b_enc
        topk_vals, topk_idx = pre_acts.topk(k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return None, acts

def main():
    parser = argparse.ArgumentParser(description="Find the steering feature vector in the trained SAE.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM.")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the trained SAE checkpoint (.pt).")
    parser.add_argument("--sae_type", type=str, choices=["custom", "off_the_shelf"], required=True, help="Type of SAE architecture.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the search dataset JSON.")
    parser.add_argument("--task", type=str, choices=list(TASK_HANDLERS.keys()), required=True, help="Task type.")
    parser.add_argument("--layer_idx", type=int, required=True, help="Layer index where SAE was trained.")
    parser.add_argument("--top_k", type=int, required=True, help="Top-K sparsity level to use during search.")
    parser.add_argument("--analysis_dir", type=str, required=True, help="Directory to save XAI analysis data.")
    parser.add_argument("--vector_dir", type=str, required=True, help="Directory to save the steering vector.")
    parser.add_argument("--exp_factor", type=int, default=32, help="Expansion factor (only needed for 'custom' sae_type).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for LLM inference.")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.analysis_dir, exist_ok=True)
    os.makedirs(args.vector_dir, exist_ok=True)

    print("Loading Tokenizer and LLM...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModel.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype=torch.float16,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    llm.eval()

    print(f"Loading SAE from {args.sae_path} (Type: {args.sae_type})...")
    if args.sae_type == "custom":
        d_model = llm.config.hidden_size 
        n_features = d_model * args.exp_factor
        sae = TopKSAE(d_model=d_model, n_features=n_features)
        sae.load_state_dict(torch.load(args.sae_path, map_location="cpu"))
    elif args.sae_type == "off_the_shelf":
        sae_dict = torch.load(args.sae_path, map_location="cpu")
        sae = OffTheShelfSAE(sae_dict["W_enc"], sae_dict["b_enc"])
    
    sae = sae.to(llm.device)
    sae.eval()

    print(f"Loading search dataset from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    handler = TASK_HANDLERS[args.task]()

    print(f"Preparing texts for task: {args.task}...")
    pos_texts = []
    neg_texts = []

    for item in dataset:
        prompt, ans_pos, ans_neg = handler.process_item(item, tokenizer)
        if ans_pos is not None:
            pos_texts.append(prompt + ans_pos)
        if ans_neg is not None:
            neg_texts.append(prompt + ans_neg)

    def get_batched_sae_activations(texts, desc):
        all_activations = []
        texts = sorted(texts, key=len)
        
        captured_h = []
        def hook_fn(module, input, output):
            # output[0] shape: [batch_size, seq_len, d_model]
            # We only extract the last token (-1) and detach it immediately
            captured_h.append(output[0][:, -1, :].detach())
            
        target_layer = llm.model.layers[args.layer_idx]
        hook_handle = target_layer.register_forward_hook(hook_fn)
        
        for i in tqdm(range(0, len(texts), args.batch_size), desc=desc):
            batch_texts = texts[i:i + args.batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=False).to(llm.device)
            captured_h.clear()
            
            with torch.no_grad():
                llm(**inputs)
                
            last_token_h = captured_h[0]
            
            # Pass through SAE
            _, f_acts = sae(last_token_h.float(), k=args.top_k)
            all_activations.append(f_acts.cpu())
            
            # Free memory
            del inputs
            torch.cuda.empty_cache()
            
        hook_handle.remove()
        return torch.cat(all_activations, dim=0) if all_activations else torch.empty(0)

    print("Extracting SAE activations...")
    pos_tensor = get_batched_sae_activations(pos_texts, "Positive Examples")
    neg_tensor = get_batched_sae_activations(neg_texts, "Negative Examples")

    print("Calculating Mean Differences...")
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
    
    if args.sae_type == "custom":
        steering_vector = sae.W_enc[:, best_feature_idx].detach().cpu()
    else:
        steering_vector = sae.W_enc[best_feature_idx, :].detach().cpu()

    vector_filename = f"sv_{args.task}_L{args.layer_idx}_{args.top_k}.pt"
    out_file = os.path.join(args.vector_dir, vector_filename)
    torch.save(steering_vector, out_file)
    
    analysis_data = {
        "task": args.task,
        "best_feature_idx": best_feature_idx,
        "deltas": diff, 
        "best_feature_pos_acts": pos_tensor[:, best_feature_idx], 
        "best_feature_neg_acts": neg_tensor[:, best_feature_idx]  
    }
    analysis_filename = f"f_analysis_L{args.layer_idx}_{args.top_k}.pt"
    analysis_file = os.path.join(args.analysis_dir, analysis_filename)
    torch.save(analysis_data, analysis_file)

    print(f"\nSuccess! Steering vector saved to {out_file}")
    print(f"XAI Analysis data saved to {analysis_file}")

if __name__ == "__main__":
    main()
