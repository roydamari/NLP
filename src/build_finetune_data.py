import json
import yaml
import random
from collections import Counter
from utils import set_seed, normalize_answer, extract_answer

def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

def get_majority_vote_idx(generations):
    """
    Returns the index of the most frequent generation after normalization.
    This ensures the SFT target reflects the model's typical answer.
    """
    norm_gens = [normalize_answer(g) for g in generations]
    counter = Counter(norm_gens)
    majority_norm_ans = counter.most_common(1)[0][0]

    for i, g in enumerate(generations):
        if normalize_answer(g) == majority_norm_ans:
            return i
    return 0

def main():
    set_seed(42)
    config = load_config()
    n_samples = config.get("n_samples", 10)
    in_file = "data/labeled/triviaqa_finetune_labeled.jsonl"
    out_train = "data/processed/train.jsonl"
    out_val = "data/processed/val.jsonl"

    data = []
    with open(in_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    random.shuffle(data)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    def write_sft_data(subset, out_path):
        written = 0
        with open(out_path, "w") as f:
            for item in subset:
                chosen_idx = get_majority_vote_idx(item["generations"])

                if item["is_fallbacks"][chosen_idx]:
                    continue

                chosen_answer = item["generations"][chosen_idx]
                k = round(item["k"] * 10 / n_samples)

                clean_ans, _ = extract_answer(chosen_answer, target_format="baseline")
                messages = [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": f"{clean_ans}. My confidence is {k} out of 10."}
                ]

                f.write(json.dumps({"messages": messages}) + "\n")
                written += 1
        return written

    train_written = write_sft_data(train_data, out_train)
    val_written = write_sft_data(val_data, out_val)

    print(f"Wrote {train_written} train and {val_written} val samples (filtered out fallbacks).")

if __name__ == "__main__":
    main()