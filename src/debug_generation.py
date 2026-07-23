"""
Diagnostic script: visually confirm that the new prompt phrasing eliminates instruction echoing.

Runs 5 questions through the model under two conditions:
  - Greedy decoding  (do_sample=False)
  - Sampled decoding (do_sample=True, temperature=1.0)

Prints the decoded prompt and generation side-by-side for manual inspection.
"""
import json
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

def build_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Answer the question in one short line. Start your response with \"Answer:\" "
        "followed by only your answer. Do not repeat these instructions."
    )

def run_batch(model, tokenizer, questions, do_sample: bool, temperature: float = 1.0):
    input_ids_list = []
    prompts_decoded = []

    for q in questions:
        msgs = [{"role": "user", "content": build_prompt(q)}]
        encoded = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        input_ids_list.append(encoded)
        prompts_decoded.append(tokenizer.decode(encoded, skip_special_tokens=False))

    inputs = tokenizer.pad(
        {"input_ids": input_ids_list}, return_tensors="pt", padding=True
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=60,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return prompts_decoded, decoded


def main():
    config = load_config()
    model_id = config["model_name"]

    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load 5 questions from the finetune source
    with open("data/processed/triviaqa_finetune.jsonl", "r") as f:
        rows = [json.loads(l) for l in f][:5]
    questions = [r["question"] for r in rows]

    SEP = "─" * 80

    for label, do_sample in [("GREEDY (do_sample=False)", False), ("SAMPLED (temperature=1.0)", True)]:
        print(f"\n{'═'*80}")
        print(f"  {label}")
        print(f"{'═'*80}")
        prompts, generations = run_batch(model, tokenizer, questions, do_sample=do_sample)

        for i, (q, prompt, gen) in enumerate(zip(questions, prompts, generations)):
            print(f"\n[{i+1}] Question: {q}")
            print(SEP)
            # Show just the user-turn portion of the prompt (last non-empty line block)
            user_turn = [l for l in prompt.split("\n") if l.strip()]
            print("PROMPT (user turn excerpt):")
            for line in user_turn[-6:]:   # last 6 lines = instruction portion
                print(f"  {line}")
            print(f"\nGENERATION:")
            print(f"  {gen.strip()}")

            # Flag if echoing detected
            instruction_fragments = [
                "answer the question",
                "start your response",
                "do not repeat",
                "[your answer]",
            ]
            echo_detected = any(frag in gen.lower() for frag in instruction_fragments)
            if echo_detected:
                print("  ⚠️  ECHO DETECTED — model is repeating instruction text!")
            else:
                print("  ✅  No echo detected.")
            print(SEP)


if __name__ == "__main__":
    main()
