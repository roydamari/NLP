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
    
    os.makedirs("data/labeled", exist_ok=True)
    
    out_file = "data/labeled/generations.jsonl"
    input_file = "data/processed/triviaqa_finetune.jsonl"
    
    with open(input_file, "r") as f:
        dataset = [json.loads(line) for line in f]
    
    # Resume logic: load already-completed question IDs from the JSONL
    done_ids = set()
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            for line in f:
                item = json.loads(line)
                done_ids.add(item["question_id"])

    to_process = [item for item in dataset if item["id"] not in done_ids]
    
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
    
    batch_size = 4
    n_samples = config["n_samples"]
    temp = config["sampling_temperature"]
    
    # Open in append mode so crashed runs can resume safely
    with open(out_file, "a") as out_f:
        for i in tqdm(range(0, len(to_process), batch_size)):
            batch = to_process[i:i+batch_size]
            input_ids_list = []
            for item in batch:
                msgs = [
                    {"role": "user", "content": f"{item['question']}\n\nAnswer the question in one short line. Start your response with \"Answer:\" followed by only your answer. Do not repeat these instructions."}
                ]
                encoded = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
                if hasattr(encoded, "keys") and "input_ids" in encoded:
                    input_ids_list.append(encoded["input_ids"])
                elif hasattr(encoded, "input_ids"):
                    input_ids_list.append(encoded.input_ids)
                else:
                    input_ids_list.append(encoded)
            
            inputs = tokenizer.pad({"input_ids": input_ids_list}, return_tensors="pt", padding=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=temp,
                    num_return_sequences=n_samples,
                    pad_token_id=tokenizer.pad_token_id
                )
                
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_len:]
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for b_idx, item in enumerate(batch):
                start_idx = b_idx * n_samples
                end_idx = start_idx + n_samples
                gens = [g.strip() for g in decoded[start_idx:end_idx]]
                
                # Write one line per question immediately (crash-safe)
                out_f.write(json.dumps({
                    "question_id": item["id"],
                    "question": item["question"],
                    "generations": gens
                }) + "\n")
                out_f.flush()

if __name__ == "__main__":
    main()
