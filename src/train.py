import os
import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from utils import set_seed

def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_check", action="store_true", help="Run 50-example overfit test")
    args = parser.parse_args()
    
    config = load_config()
    set_seed(config["seed"])
    
    model_id = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = load_dataset("json", data_files={"train": "data/processed/train.jsonl", "val": "data/processed/val.jsonl"})
    
    if args.sanity_check:
        print("Running SANITY CHECK (50 examples)")
        dataset["train"] = dataset["train"].select(range(min(50, len(dataset["train"]))))
        dataset["val"] = dataset["val"].select(range(min(10, len(dataset["val"]))))
        epochs = 10 # More epochs to ensure overfitting
        output_dir = "outputs/checkpoints/sanity_check"
    else:
        epochs = config["num_epochs"]
        output_dir = "outputs/checkpoints/run_full"
        
    def formatting_prompts_func(example):
        # example["messages"] is a list of dicts: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        output_texts = []
        for i in range(len(example["messages"])):
            # Assuming dataset was loaded with batched=True, example["messages"] is a list of lists of dicts
            # Oh wait, load_dataset without batched formatting needs to handle single examples if not mapped.
            pass
            
        # Using TRL's built-in formatting if tokenizer has chat template
        return example
        
    # We will use SFTTrainer's dataset_kwargs or a simple map if needed.
    # We'll apply chat template in a map to be safe.
    def apply_chat_template(examples):
        # examples["messages"] is a batch of message lists
        texts = [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in examples["messages"]]
        return {"text": texts}
        
    dataset = dataset.map(apply_chat_template, batched=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=float(config["learning_rate"]),
        logging_steps=10,
        num_train_epochs=epochs,
        save_strategy="epoch",
        eval_strategy="epoch" if not args.sanity_check else "no",
        save_total_limit=2, # Don't fill Slurm storage!
        bf16=True,
        dataset_text_field="text",
        max_length=512,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if not args.sanity_check else None,
        peft_config=lora_config,
        args=training_args,
    )
    
    trainer.train()
    
    # Save the adapter
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
