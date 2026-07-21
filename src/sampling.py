import os
import json
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import set_seed
from tqdm import tqdm

def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    set_seed(config["seed"])
    
    out_dir = "data/labeled/generations_raw"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load dataset
    input_file = "data/processed/triviaqa_finetune_split.jsonl"
    with open(input_file, "r") as f:
        dataset = [json.loads(line) for line in f]
    
    # Resume logic: skip already generated questions
    existing_files = set(os.listdir(out_dir))
    to_process = []
    for item in dataset:
        if f"{item['id']}.json" not in existing_files:
            to_process.append(item)
            
    if not to_process:
        print("All samples already generated! Exiting.")
        return
        
    print(f"Resuming/starting generation for {len(to_process)} samples...")
    
    model_id = config["model_name"]
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    batch_size = 4  # Adjust based on VRAM
    n_samples = config["n_samples"]
    temp = config["sampling_temperature"]
    
    # Simple batched generation loop
    for i in tqdm(range(0, len(to_process), batch_size)):
        batch = to_process[i:i+batch_size]
        input_ids_list = []
        for item in batch:
            msgs = [
                {"role": "user", "content": f"{item['question']}\n\nAnswer the question directly. Respond in exactly this format, with nothing after it:\nAnswer: [your answer]"}
            ]
            encoded = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
            if hasattr(encoded, "keys") and "input_ids" in encoded:
                input_ids_list.append(encoded["input_ids"])
            elif hasattr(encoded, "input_ids"):
                input_ids_list.append(encoded.input_ids)
            else:
                input_ids_list.append(encoded)
        
        # We want n_samples per prompt. Easiest way in HF is num_return_sequences
        inputs = tokenizer.pad({"input_ids": input_ids_list}, return_tensors="pt", padding=True).to(model.device)
        
        # Attention mask needs to be explicitly created if tokenizer.pad doesn't return it when passing just input_ids.
        # Actually tokenizer.pad does return attention_mask!
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=temp,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Strip input tokens from outputs before decoding
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Output is shape (batch_size * n_samples).
        for b_idx, item in enumerate(batch):
            start_idx = b_idx * n_samples
            end_idx = start_idx + n_samples
            gens = decoded[start_idx:end_idx]
            
            clean_gens = [g.strip() for g in gens]
                    
            # Incremental save
            out_path = os.path.join(out_dir, f"{item['id']}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "question_id": item["id"],
                    "question": item["question"],
                    "generations": clean_gens
                }, f)

if __name__ == "__main__":
    main()
