import os
import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from utils import set_seed, parse_confidence
from confidence_loss import build_digit_token_ids, find_confidence_token_index
from model_loading import get_device_map, load_model_for_training

def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

class ConfidenceDataCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        conf_idx = [f.pop("conf_idx") for f in features]
        conf_target = [f.pop("conf_target") for f in features]
        batch = self.base_collator(features)
        batch["conf_idx"] = torch.tensor(conf_idx, dtype=torch.long)
        batch["conf_target"] = torch.tensor(conf_target, dtype=torch.float)
        return batch

class ConfidenceAwareTrainer(Trainer):
    def __init__(self, *args, digit_token_ids=None, brier_weight=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        values = sorted(digit_token_ids.keys())
        self.digit_values = torch.tensor(values, dtype=torch.float)
        self.digit_ids = torch.tensor([digit_token_ids[v] for v in values], dtype=torch.long)
        self.brier_weight = brier_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        conf_idx = inputs.pop("conf_idx")
        conf_target = inputs.pop("conf_target")
        outputs = model(**inputs)
        lm_loss = outputs.loss
        logits = outputs.logits

        device = logits.device
        digit_ids = self.digit_ids.to(device)
        digit_values = self.digit_values.to(device)

        valid_mask = conf_idx >= 0
        if valid_mask.sum() == 0:
            total_loss = lm_loss
        else:
            batch_idx = torch.arange(logits.size(0), device=device)[valid_mask]
            tok_idx = conf_idx[valid_mask].to(device)
            targets = conf_target[valid_mask].to(device)

            pos_logits = logits[batch_idx, tok_idx]
            digit_logits = pos_logits[:, digit_ids]
            probs = torch.softmax(digit_logits, dim=-1)
            expected_conf = (probs * digit_values.unsqueeze(0)).sum(dim=-1) / 10.0

            brier_loss = ((expected_conf - targets) ** 2).mean()
            total_loss = lm_loss + self.brier_weight * brier_loss

        return (total_loss, outputs) if return_outputs else total_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    set_seed(config["seed"])
    use_4bit = config.get("use_4bit", False)

    model_id = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        epochs = args.num_epochs if args.num_epochs is not None else 10
        output_dir = f"outputs/checkpoints/sanity_check_e{epochs}"
        logging_steps = 1
    else:
        epochs = args.num_epochs if args.num_epochs is not None else config["num_epochs"]
        output_dir = "outputs/checkpoints/run_full"
        logging_steps = 10

    digit_token_ids = build_digit_token_ids(tokenizer)

    def tokenize_function(examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []
        conf_idx_list, conf_target_list = [], []

        for msgs in examples["messages"]:
            encoded = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=False)
            if hasattr(encoded, "keys") and "input_ids" in encoded:
                encoded = encoded["input_ids"]
            elif hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids

            user_msg = [msgs[0]]
            prompt_encoded = tokenizer.apply_chat_template(user_msg, tokenize=True, add_generation_prompt=True, return_dict=False)
            if hasattr(prompt_encoded, "keys") and "input_ids" in prompt_encoded:
                prompt_encoded = prompt_encoded["input_ids"]
            elif hasattr(prompt_encoded, "input_ids"):
                prompt_encoded = prompt_encoded.input_ids
            prompt_len = len(prompt_encoded)

            labels = [-100] * prompt_len + encoded[prompt_len:]

            assistant_text = msgs[1]["content"]
            conf_val = parse_confidence(assistant_text)
            conf_idx = None
            if conf_val is not None:
                conf_idx = find_confidence_token_index(tokenizer, encoded, prompt_len, assistant_text, conf_val)

            input_ids_list.append(encoded)
            attention_mask_list.append([1] * len(encoded))
            labels_list.append(labels)
            conf_idx_list.append(conf_idx if conf_idx is not None else -1)
            conf_target_list.append((conf_val / 10.0) if conf_val is not None else 0.0)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
            "conf_idx": conf_idx_list,
            "conf_target": conf_target_list,
        }

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["messages"])

    device_map = get_device_map()
    model = load_model_for_training(model_id, use_4bit, device_map)

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=float(config["learning_rate"]),
        logging_steps=logging_steps,
        logging_strategy="epoch" if not args.sanity_check else "steps",
        num_train_epochs=epochs,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    base_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    data_collator = ConfidenceDataCollator(base_collator)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = ConfidenceAwareTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if has_val else None,
        args=training_args,
        data_collator=data_collator,
        digit_token_ids=digit_token_ids,
        brier_weight=2.0,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()