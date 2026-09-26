"""
Plain supervised fine-tuning (no auxiliary losses) -- the configuration of Runs 1-3 in the paper.
Hyperparameters are read from configs/base.yaml (defaults below):
LoRA r=16, alpha=32 (q_proj, v_proj), lr 3e-4, 3 epochs, effective batch 16.
The checkpoint with the lowest validation loss is kept automatically.
Output: outputs/checkpoints/run_plain
"""
import os, sys, json, yaml, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from utils import set_seed

LORA_R, LORA_ALPHA, LR, EPOCHS = 16, 32, 3e-4, 3
OUTPUT_DIR = "outputs/checkpoints/run_plain"

def main():
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available on this node. Cancel and resubmit with --exclude=<this node>.")
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    global LORA_R, LORA_ALPHA, LR, EPOCHS
    LORA_R = int(config.get("lora_r", LORA_R))
    LORA_ALPHA = int(config.get("lora_alpha", LORA_ALPHA))
    LR = float(config.get("learning_rate", LR))
    EPOCHS = int(config.get("num_epochs", EPOCHS))
    set_seed(config["seed"])
    model_id = config["model_name"]
    print(f"Model: {model_id} | LoRA r={LORA_R} alpha={LORA_ALPHA} lr={LR} epochs={EPOCHS}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": "data/processed/train.jsonl",
                                                "val": "data/processed/val.jsonl"})

    def to_ids(x):
        if hasattr(x, "keys") and "input_ids" in x:
            return list(x["input_ids"])
        if hasattr(x, "input_ids"):
            return list(x.input_ids)
        return list(x)

    def tokenize_function(examples):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for msgs in examples["messages"]:
            full = to_ids(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=False))
            prompt = to_ids(tokenizer.apply_chat_template([msgs[0]], tokenize=True, add_generation_prompt=True, return_dict=False))
            labels = [-100] * len(prompt) + full[len(prompt):]
            out["input_ids"].append(full)
            out["attention_mask"].append([1] * len(full))
            out["labels"].append(labels)
        return out

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["messages"])

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                 device_map={"": torch.cuda.current_device()})
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                             target_modules=["q_proj", "v_proj"],
                                             bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset["train"],
                      eval_dataset=dataset["val"],
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))
    trainer.train()
    trainer.save_model(OUTPUT_DIR)   # saves the best (lowest eval_loss) adapter
    summary = {"best_checkpoint": trainer.state.best_model_checkpoint,
               "best_eval_loss": trainer.state.best_metric,
               "log_history": trainer.state.log_history}
    with open(os.path.join(OUTPUT_DIR, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved best model to {OUTPUT_DIR} (best checkpoint: {summary['best_checkpoint']}, "
          f"eval_loss={summary['best_eval_loss']})", flush=True)

if __name__ == "__main__":
    main()
