import argparse
import yaml
import wandb

from experiments.train_llm.model import get_model_and_tokenizer
from experiments.train_llm.dataset import load_and_prepare_dataset, get_formatting_func
from experiments.train_llm.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Run LLM Fine-Tuning.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--chat_template", type=str, required=True, help="Path to chat_template.yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--run_name", type=str, required=True, help="Wandb run name")
    parser.add_argument("--deepspeed", type=str, default="none", help="Path to ds_config.json")
    
    # Dynamic batching arguments
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--wandb_entity", type=str, required=True)
    
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    wandb.init(
        project=args.wandb_project, 
        entity=args.wandb_entity,
        name=args.run_name
    )

    model, tokenizer = get_model_and_tokenizer(args.model_path, config)
    dataset = load_and_prepare_dataset(args.dataset_path)
    formatting_func, response_template = get_formatting_func(tokenizer, args.chat_template)

    train_model(model, tokenizer, dataset, formatting_func, response_template, config, args)

if __name__ == "__main__":
    main()
