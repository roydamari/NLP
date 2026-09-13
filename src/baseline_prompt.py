import os
import json
import yaml
import torch
from transformers import AutoTokenizer
from model_loading import get_device_map, load_model_for_inference
from tqdm import tqdm

def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    model_id = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = get_device_map()
    model = load_model_for_inference(model_id, config.get("use_4bit", False), device_map)

    os.makedirs("outputs/eval_results", exist_ok=True)

    datasets_to_run = [
        ("data/processed/triviaqa_test.jsonl", "outputs/eval_results/baseline_indist.jsonl"),
        ("data/processed/ood_test.jsonl", "outputs/eval_results/baseline_ood.jsonl")
    ]

    batch_size = 4

    for in_file, out_file in datasets_to_run:
        print(f"Running baseline evaluation for {in_file}...")
        with open(in_file, "r") as f:
            dataset = [json.loads(line) for line in f]

        with open(out_file, "w") as f_out:
            for i in tqdm(range(0, len(dataset), batch_size)):
                batch = dataset[i:i+batch_size]

                input_ids_list = []
                for item in batch:
                    msgs = [
                        {"role": "user", "content": f"{item['question']}\n\nAnswer the question in one short line. Start your response with \"Answer:\" followed by only your answer. Do not repeat these instructions.\n\nFinally, provide your answer followed by 'My confidence is X out of 10' where X is your confidence."}
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
                        max_new_tokens=90,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )

                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[:, input_len:]
                decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                for b_idx, item in enumerate(batch):
                    gen = decoded[b_idx]

                    f_out.write(json.dumps({
                        "question_id": item["id"],
                        "question": item["question"],
                        "generation": gen,
                        "gold_aliases": item["answers"]["aliases"]
                    }) + "\n")
                    f_out.flush()

if __name__ == "__main__":
    main()