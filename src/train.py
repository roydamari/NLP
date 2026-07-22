import os
import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
        
    # Check if val file has content — it may be empty with very small datasets
    val_file = "data/processed/val.jsonl"
    has_val = os.path.exists(val_file) and os.path.getsize(val_file) > 0
    
    data_files = {"train": "data/processed/train.jsonl"}
    if has_val:
        data_files["val"] = val_file
    dataset = load_dataset("json", data_files=data_files)
    
    if args.sanity_check:
        print("Running SANITY CHECK (50 examples)")
        dataset["train"] = dataset["train"].select(range(min(50, len(dataset["train"]))))
        if has_val:
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
        
    def tokenize_function(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for msgs in examples["messages"]:
            # Tokenize the full conversation
            encoded = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=False)
            if hasattr(encoded, "keys") and "input_ids" in encoded:
                encoded = encoded["input_ids"]
            elif hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids
                
            # Tokenize ONLY the user prompt to find its length
            user_msg = [msgs[0]]
            prompt_encoded = tokenizer.apply_chat_template(user_msg, tokenize=True, add_generation_prompt=True, return_dict=False)
            if hasattr(prompt_encoded, "keys") and "input_ids" in prompt_encoded:
                prompt_encoded = prompt_encoded["input_ids"]
            elif hasattr(prompt_encoded, "input_ids"):
                prompt_encoded = prompt_encoded.input_ids
                
            prompt_len = len(prompt_encoded)
            
            # Mask the user prompt with -100 so it's ignored in cross-entropy loss
            labels = [-100] * prompt_len + encoded[prompt_len:]
            
            input_ids_list.append(encoded)
            attention_mask_list.append([1] * len(encoded))
            labels_list.append(labels)
            
        return {
            "input_ids": input_ids_list, 
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }
        
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["messages"])
    
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
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=float(config["learning_rate"]),
        logging_steps=10,
        num_train_epochs=epochs,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        save_total_limit=2, # Don't fill Slurm storage!
        bf16=True,
    )
    
    # Use DataCollatorForSeq2Seq to dynamically pad input_ids and our custom labels
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    
    # Need to manually wrap model for PEFT when using standard Trainer
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if has_val else None,
        args=training_args,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Save the adapter
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
