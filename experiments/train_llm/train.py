from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def train_model(model, tokenizer, dataset, formatting_func, response_template, config, args):
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer
    )

    trainer_args_dict = config["trainer"]["trainer_args"].copy()
    
    # Override with CLI arguments
    trainer_args_dict["output_dir"] = args.output_dir
    trainer_args_dict["run_name"] = args.run_name
    trainer_args_dict["per_device_train_batch_size"] = args.per_device_train_batch_size
    trainer_args_dict["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    
    if args.deepspeed and args.deepspeed.lower() != "none":
        trainer_args_dict["deepspeed"] = args.deepspeed

    training_args = TrainingArguments(**trainer_args_dict)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
        data_collator=collator,
        max_seq_length=config["model"]["max_seq_length"],
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
