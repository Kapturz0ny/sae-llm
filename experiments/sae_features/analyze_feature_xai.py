import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

def plot_activation_separation(pos_acts, neg_acts, best_idx, output_dir, model_name, task, layer):
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(pos_acts.numpy(), fill=True, color='green', label='Positive Behavior')
    sns.kdeplot(neg_acts.numpy(), fill=True, color='red', label='Negative Behavior')
    
    plt.title(f"Activation Dist for Feature [{best_idx}] - {model_name} | Layer: {layer}")
    plt.xlabel("Activation Value")
    plt.ylabel("Density")
    plt.legend()
    
    out_path = os.path.join(output_dir, f"dist_{model_name}_{task}_L{layer}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def logit_lens_analysis(model, tokenizer, steering_vector, output_dir, model_name, task, layer, best_idx):
    # Extract RMSNorm weights and scale the vector
    ln_weights = model.model.norm.weight.detach().float().cpu()
    scaled_vector = steering_vector.float().cpu() * ln_weights
    
    # Project to vocabulary
    unembedding_matrix = model.lm_head.weight.detach().float().cpu()
    logits = torch.matmul(unembedding_matrix, scaled_vector)
    
    top_vals, top_indices = torch.topk(logits, k=20)
    bottom_vals, bottom_indices = torch.topk(logits, k=20, largest=False)
    
    report_lines = []
    report_lines.append(f"LOGIT LENS ANALYSIS")
    report_lines.append(f"Model: {model_name} | Task: {task} | Layer: {layer} | Feature: {best_idx}")
    report_lines.append("="*50)
    
    report_lines.append("\n[PROMOTED TOKENS]")
    for val, idx in zip(top_vals, top_indices):
        token_str = tokenizer.decode([idx.item()]).replace('\n', '\\n')
        report_lines.append(f"Score: {val.item():>6.2f} | Token: '{token_str}'")
        
    report_lines.append("\n[DEMOTED TOKENS]")
    for val, idx in zip(bottom_vals, bottom_indices):
        token_str = tokenizer.decode([idx.item()]).replace('\n', '\\n')
        report_lines.append(f"Score: {val.item():>6.2f} | Token: '{token_str}'")
        
    report_text = "\n".join(report_lines)
    print(report_text)
    
    out_path = os.path.join(output_dir, f"logit_lens_{model_name}_{task}_L{layer}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

def main():
    parser = argparse.ArgumentParser(description="Run XAI analysis on the extracted SAE feature.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM.")
    parser.add_argument("--analysis_file", type=str, required=True, help="Path to the .pt analysis file.")
    parser.add_argument("--vector_file", type=str, required=True, help="Path to the .pt steering vector file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots and reports.")
    parser.add_argument("--layer_idx", type=int, required=True, help="Layer index.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading XAI data from {args.analysis_file}...")
    xai_data = torch.load(args.analysis_file, map_location="cpu")
    steering_vector = torch.load(args.vector_file, map_location="cpu")

    model_name = xai_data.get("model_name", "UnknownModel")
    task = xai_data.get("task", "unknown_task")
    best_idx = xai_data.get("best_feature_idx", 0)

    print("Generating plots...")
    plot_activation_separation(xai_data["best_feature_pos_acts"], xai_data["best_feature_neg_acts"], best_idx, args.output_dir, model_name, task, args.layer_idx)

    print(f"Loading Model {args.model_path} for Logit Lens...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    logit_lens_analysis(model, tokenizer, steering_vector, args.output_dir, model_name, task, args.layer_idx, best_idx)
    print(f"All XAI artifacts saved to {args.output_dir}")

if __name__ == "__main__":
    main()
