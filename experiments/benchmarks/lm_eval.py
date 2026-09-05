import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM

from experiments.benchmarks.utils import attach_steering_vector

def main():
    parser = argparse.ArgumentParser(description="Evaluate general capabilities (Alignment Tax) using lm-eval.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save the results.")
    
    # Steering parameters
    parser.add_argument("--vector_path", type=str, default="none", help="Path to the steering vector (.pt).")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index for the steering vector.")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Multiplier for the steering vector.")
    
    # Eval parameters
    parser.add_argument("--tasks", type=str, default="wikitext,gsm8k", help="Comma-separated list of tasks (e.g., wikitext,mmlu,gsm8k).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    
    # Determine experiment name
    model_name = os.path.basename(os.path.normpath(args.model_path))
    exp_suffix = f"steered_m{args.multiplier}" if args.vector_path.lower() != "none" else "baseline"
    
    exp_dir = os.path.join(args.results_dir, f"{model_name}_{exp_suffix}")
    os.makedirs(exp_dir, exist_ok=True)
    out_file = os.path.join(exp_dir, "capabilities_score.json")

    print(f"Loading Tokenizer and Model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    hook_handle = None
    if args.vector_path.lower() != "none":
        hook_handle = attach_steering_vector(model, args.vector_path, args.layer_idx, args.multiplier)

    print(f"Wrapping model for lm-eval...")
    # Wrap our modified model in lm_eval's HFLM class
    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )

    task_list = [t.strip() for t in args.tasks.split(",")]
    print(f"Starting evaluation on tasks: {task_list}...")

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=task_list,
        num_fewshot=0,
        batch_size=args.batch_size,
        device=model.device
    )

    if hook_handle is not None:
        hook_handle.remove()

    print(f"Saving results to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        # lm_eval results contain non-serializable objects, so we extract just the results dict
        json.dump(results["results"], f, indent=2, ensure_ascii=False)

    print("\n" + "="*40)
    print(f" CAPABILITIES BENCHMARK ({exp_suffix.upper()})")
    print("="*40)
    for task_name, task_metrics in results["results"].items():
        print(f"Task: {task_name}")
        for metric_name, metric_val in task_metrics.items():
            # Filter out standard error metrics for cleaner print
            if not metric_name.endswith("_stderr") and metric_name != "alias":
                print(f"  {metric_name}: {metric_val:.4f}")
        print("-" * 40)
    print("="*40)

if __name__ == "__main__":
    main()
