import os
import yaml
import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk

from experiments.train_sae.train import TopKSAE, Trainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'], 
        trust_remote_code=True
    )
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(config['dataset_path'])
    
    print("Loading LLM (Base Model) in FP16...")
    llm = AutoModel.from_pretrained(
        config['model_path'],
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    llm.eval() # Freeze LLM

    print("Initializing Top-K SAE (Tied Weights)...")
    n_features = config['d_model'] * config['expansion_factor']
    sae = TopKSAE(
        d_model=config['d_model'],
        n_features=n_features
    )
    sae = sae.to(llm.device)

    print("Starting Trainer...")
    trainer = Trainer(config, llm, sae, tokenizer, dataset)
    trainer.train()

if __name__ == "__main__":
    main()