import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def get_model_and_tokenizer(model_path, config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(config["model"].get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        device_map=None, 
        trust_remote_code=True
    )

    peft_config = config.get("peft", {})
    if peft_config.get("use_peft", False):
        print("Applying PEFT (LoRA)...")
        lora_config = LoraConfig(
            r=peft_config["lora_r"],
            lora_alpha=peft_config["lora_alpha"],
            target_modules=peft_config["target_modules"],
            lora_dropout=peft_config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=peft_config.get("use_rslora", False)
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer