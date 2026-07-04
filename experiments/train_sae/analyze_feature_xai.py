import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

def plot_histogram_of_deltas(deltas, best_idx, output_dir, task):
    """Plots the distribution of all 32,768 feature deltas."""
    plt.figure(figsize=(10, 6))
    
    deltas_np = deltas.numpy()
    sns.histplot(deltas_np, bins=150, color='blue', log_scale=(False, True))
    
    # Highlight the chosen feature
    best_val = deltas_np[best_idx]
    plt.axvline(x=best_val, color='red', linestyle='--', label=f'Best Feature [{best_idx}]')
    
    plt.title(f"Distribution of Feature Deltas (Mean Difference) - {task.upper()}")
    plt.xlabel("Delta (Mean Positive Activation - Mean Negative Activation)")
    plt.ylabel("Count of Features (Log Scale)")
    plt.legend()
    
    out_path = os.path.join(output_dir, f"xai_deltas_hist_{task}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Deltas Histogram to {out_path}")
    plt.close()

def plot_activation_separation(pos_acts, neg_acts, best_idx, output_dir, task):
    """Plots how well the chosen feature separates positive and negative examples."""
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(pos_acts.numpy(), fill=True, color='green', label='Positive (e.g. Sycophantic)')
    sns.kdeplot(neg_acts.numpy(), fill=True, color='red', label='Negative (e.g. Assertive)')
    
    plt.title(f"Activation Distribution for Best Feature [{best_idx}] - {task.upper()}")
    plt.xlabel("Activation Value")
    plt.ylabel("Density")
    plt.legend()
    
    out_path = os.path.join(output_dir, f"xai_activation_dist_{task}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Activation Distribution to {out_path}")
    plt.close()

def logit_lens_analysis(model, tokenizer, steering_vector):
    """Projects the steering vector into the vocabulary space to find promoted/demoted tokens."""
    print("\n" + "="*50)
    print(" LOGIT LENS ANALYSIS (Unembedding the Feature)")
    print("="*50)
    
    # weights from RMSNorm
    ln_weights = model.model.norm.weight.detach().float().cpu()
    
    # scale steering vector by RMSNorm weights
    scaled_vector = steering_vector.float().cpu() * ln_weights
    
    unembedding_matrix = model.lm_head.weight.detach().float().cpu()
    
    logits = torch.matmul(unembedding_matrix, scaled_vector)
    
    # top 15 best and 15 bottom tokens
    top_vals, top_indices = torch.topk(logits, k=15)
    bottom_vals, bottom_indices = torch.topk(logits, k=15, largest=False)
    
    print("\n🟢 TOP PROMOTED TOKENS (What this feature 'wants' to say):")
    for val, idx in zip(top_vals, top_indices):
        token_str = tokenizer.decode([idx.item()])
        token_str = token_str.replace('\n', '\\n')
        print(f"  Score: {val.item():>6.2f} | Token: '{token_str}'")
        
    print("\n🔴 TOP DEMOTED TOKENS (What this feature suppresses):")
    for val, idx in zip(bottom_vals, bottom_indices):
        token_str = tokenizer.decode([idx.item()])
        token_str = token_str.replace('\n', '\\n')
        print(f"  Score: {val.item():>6.2f} | Token: '{token_str}'")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run XAI analysis on the extracted SAE feature.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM (for Logit Lens).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the .pt files.")
    parser.add_argument("--task", type=str, required=True, help="Task type (e.g., sycophancy).")
    args = parser.parse_args()

    vec_path = os.path.join(args.data_dir, f"steering_vec_{args.task}.pt")
    xai_path = os.path.join(args.data_dir, f"xai_data_{args.task}.pt")

    print(f"Loading XAI data from {xai_path}...")
    xai_data = torch.load(xai_path, map_location="cpu")
    steering_vector = torch.load(vec_path, map_location="cpu")

    plot_histogram_of_deltas(
        xai_data["deltas"], 
        xai_data["best_feature_idx"], 
        args.data_dir, 
        args.task
    )
    
    plot_activation_separation(
        xai_data["best_feature_pos_acts"], 
        xai_data["best_feature_neg_acts"], 
        xai_data["best_feature_idx"], 
        args.data_dir, 
        args.task
    )

    print(f"Loading Model {args.model_path} for Logit Lens...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    logit_lens_analysis(model, tokenizer, steering_vector)

if __name__ == "__main__":
    main()