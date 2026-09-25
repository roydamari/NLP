"""
Final evaluation for the paper. For BOTH the zero-shot baseline and the fine-tuned model,
on BOTH test sets, this saves per-question:
  - the greedy generation, its correctness, and the stated (greedy) confidence
  - the model's full probability distribution over the confidence values 0..10 at the
    position right after "My confidence is", and its expected value (the mean).
The expected value tests the "mode vs. mean" hypothesis: greedy decoding reports the most
likely confidence value (the mode); the expected value reports the average (the mean).
Output: outputs/eval_expected/{baseline,finetuned}_{indist,ood}.jsonl
"""
import os, sys, re, json, yaml, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from utils import check_match, parse_confidence

CKPT = "outputs/checkpoints/run_plain"
OUT_DIR = "outputs/eval_expected"
DATASETS = {"indist": "data/processed/triviaqa_test.jsonl", "ood": "data/processed/ood_test.jsonl"}
BASELINE_INSTRUCTION = ("\n\nAnswer the question in one short line. Start your response with \"Answer:\" "
                        "followed by only your answer. Do not repeat these instructions.\n\n"
                        "Finally, provide your answer followed by 'My confidence is X out of 10' "
                        "where X is your confidence.")
CONF_PHRASE = re.compile(r"confidence\s+(?:level\s+)?is", re.IGNORECASE)
MAX_NEW_TOKENS = 90
BATCH_SIZE = 4

def to_ids(x):
    if hasattr(x, "keys") and "input_ids" in x:
        return list(x["input_ids"])
    if hasattr(x, "input_ids"):
        return list(x.input_ids)
    return list(x)

def confidence_token_ids(tok):
    """Token id of each value 0..10 as it appears after 'is ' (Llama encodes ' 7' as [' ', '7'])."""
    ids = []
    for v in range(11):
        enc = tok.encode(f" {v}", add_special_tokens=False)
        tid = enc[-1]
        decoded = tok.decode([tid]).strip()
        assert decoded == str(v), f"Token mapping problem for {v}: got {decoded!r} from {enc}"
        ids.append(tid)
    return ids

def confidence_prefix(generation):
    """Text up to and including the last 'confidence is'. If absent, append the phrase (flagged)."""
    matches = list(CONF_PHRASE.finditer(generation))
    if matches:
        return generation[:matches[-1].end()], False
    return generation.rstrip() + " My confidence is", True

def run_condition(model, tok, condition, conf_ids):
    target_format = "baseline" if condition == "baseline" else "finetuned"
    for ds_name, path in DATASETS.items():
        with open(path) as f:
            items = [json.loads(l) for l in f]
        prompts = []
        for it in items:
            user = it["question"] + (BASELINE_INSTRUCTION if condition == "baseline" else "")
            prompts.append(to_ids(tok.apply_chat_template([{"role": "user", "content": user}],
                                                          tokenize=True, add_generation_prompt=True,
                                                          return_dict=False)))
        # 1) greedy generation (batched)
        generations = []
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"{condition}/{ds_name} generate"):
            batch = prompts[i:i + BATCH_SIZE]
            enc = tok.pad({"input_ids": batch}, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            generations += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        # 2) probability distribution over confidence values 0..10 (one forward pass per question)
        records = []
        for it, p_ids, gen in tqdm(list(zip(items, prompts, generations)), desc=f"{condition}/{ds_name} conf-dist"):
            gen = gen.strip()
            prefix, forced = confidence_prefix(gen)
            ids = p_ids + tok.encode(prefix + " ", add_special_tokens=False)
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)[conf_ids]
            mass = float(probs.sum())
            dist = (probs / probs.sum()).tolist()
            expected = sum(v * p for v, p in enumerate(dist))
            correct, _ = check_match(gen, it["answers"]["aliases"], use_old_method=False, target_format=target_format)
            records.append({"question_id": it["id"], "question": it["question"], "generation": gen,
                            "correct": bool(correct), "greedy_conf": parse_confidence(gen),
                            "expected_conf": expected, "conf_dist": dist,
                            "conf_token_mass": mass, "forced_prefix": forced})
        out_path = os.path.join(OUT_DIR, f"{condition}_{ds_name}.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\nSaved {len(records)} records to {out_path}. Sanity check (first 3):", flush=True)
        for r in records[:3]:
            top = sorted(range(11), key=lambda v: -r["conf_dist"][v])[:3]
            print(f"  gen={r['generation'][:70]!r} | correct={r['correct']} | greedy={r['greedy_conf']} | "
                  f"expected={r['expected_conf']:.2f} | top values={top} | mass on 0-10 tokens={r['conf_token_mass']:.2f}",
                  flush=True)

def main():
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available on this node. Cancel and resubmit with --exclude=<this node>.")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open("configs/base.yaml") as f:
        model_id = yaml.safe_load(f)["model_name"]
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    conf_ids = confidence_token_ids(tok)
    print("Confidence token ids (0..10):", conf_ids, flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                 device_map={"": torch.cuda.current_device()})
    model.eval()
    run_condition(model, tok, "baseline", conf_ids)       # base model first
    model = PeftModel.from_pretrained(model, CKPT)          # then add the fine-tuned adapter
    model.eval()
    run_condition(model, tok, "finetuned", conf_ids)
    print("EVALUATION DONE", flush=True)

if __name__ == "__main__":
    main()
