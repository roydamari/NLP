import os
import json
import yaml
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from utils import check_match, parse_confidence
from metrics import compute_ece, compute_mse

def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

def run_evaluation(model, tokenizer, input_file, output_file):
    """
    Run evaluation on a dataset, saving raw generations.
    """
    with open(input_file, "r") as f:
        dataset = [json.loads(line) for line in f]
        
    batch_size = 4
    
    with open(output_file, "w") as f_out:
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            input_ids_list = []
            
            for item in batch:
                msgs = [{"role": "user", "content": item['question']}]
                
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
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
            # Strip input tokens from outputs before decoding
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_len:]
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for b_idx, item in enumerate(batch):
                gen = decoded[b_idx].strip()
                f_out.write(json.dumps({
                    "question_id": item["id"],
                    "question": item["question"],
                    "generation": gen,
                    "gold_aliases": item["answers"]["aliases"]
                }) + "\n")

def process_results(raw_file, target_format="baseline"):
    """
    Parse confidences and compute ECE/MSE.
    """
    confs = []
    accs = []
    malformed_count = 0
    fallback_count = 0
    
    with open(raw_file, "r") as f:
        for line in f:
            item = json.loads(line)
            gen = item["generation"]
            gold = item["gold_aliases"]
            
            conf = parse_confidence(gen)
            if conf is None:
                malformed_count += 1
                continue
                
            # Normalize confidence to [0, 1]
            conf = conf / 10.0
            confs.append(conf)
            
            # Check correctness
            is_correct_bool, is_fallback = check_match(gen, gold, use_old_method=False, target_format=target_format)
            if is_fallback:
                fallback_count += 1
                
            is_correct = 1.0 if is_correct_bool else 0.0
            accs.append(is_correct)
            
    confs = np.array(confs)
    accs = np.array(accs)
    
    if len(confs) == 0:
        return {"ece": 0.0, "mse": 0.0, "n": 0, "malformed_count": malformed_count, "fallback_count": fallback_count}
        
    ece = compute_ece(confs, accs)
    mse = compute_mse(confs, accs)
    
    return {
        "ece": float(ece),
        "mse": float(mse),
        "n": len(confs),
        "malformed_count": malformed_count,
        "fallback_count": fallback_count
    }

def main():
    config = load_config()
    model_id = config["model_name"]
    lora_path = "outputs/checkpoints/run_full"
    
    os.makedirs("outputs/eval_results", exist_ok=True)
    
    # 1. First, compute baseline metrics if they exist
    results_indist = {}
    results_ood = {}
    
    if os.path.exists("outputs/eval_results/baseline_indist.jsonl"):
        results_indist["baseline"] = process_results("outputs/eval_results/baseline_indist.jsonl", target_format="baseline")
    if os.path.exists("outputs/eval_results/baseline_ood.jsonl"):
        results_ood["baseline"] = process_results("outputs/eval_results/baseline_ood.jsonl", target_format="baseline")
        
    # 2. Run finetuned model
    if os.path.exists(lora_path):
        print(f"Loading base model {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        elif torch.backends.mps.is_available():
            device_map = {"": "mps"}
        else:
            device_map = "auto"
            
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device_map
        )
        
        print(f"Loading LoRA adapters from {lora_path}...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        # In-dist Eval
        print("Evaluating fine-tuned model on In-Distribution (TriviaQA)...")
        ft_indist_raw = "outputs/eval_results/finetuned_indist.jsonl"
        run_evaluation(model, tokenizer, "data/processed/triviaqa_test.jsonl", ft_indist_raw)
        results_indist["finetuned"] = process_results(ft_indist_raw, target_format="finetuned")
        
        # OOD Eval
        print("Evaluating fine-tuned model on OOD (WebQuestions)...")
        ft_ood_raw = "outputs/eval_results/finetuned_ood.jsonl"
        run_evaluation(model, tokenizer, "data/processed/ood_test.jsonl", ft_ood_raw)
        results_ood["finetuned"] = process_results(ft_ood_raw, target_format="finetuned")
    else:
        print(f"LoRA path {lora_path} not found. Skipping fine-tuned evaluation.")
        
    # Write final split JSONs
    with open("outputs/eval_results/indist_results.json", "w") as f:
        json.dump(results_indist, f, indent=4)
        
    with open("outputs/eval_results/ood_results.json", "w") as f:
        json.dump(results_ood, f, indent=4)
        
    print("Evaluation complete. Results saved to outputs/eval_results/")

if __name__ == "__main__":
    main()
