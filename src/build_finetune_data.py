import json
import random
from collections import Counter
from utils import set_seed, normalize_answer, extract_answer

def get_majority_vote_idx(generations):
    """
    Returns the index of the most frequent generation after normalization.
    This ensures the SFT target reflects the model's typical answer.
    """
    # Normalize before counting so variations of the same answer group together
    norm_gens = [normalize_answer(g) for g in generations]
    counter = Counter(norm_gens)
    majority_norm_ans = counter.most_common(1)[0][0]
    
    # Return the first raw generation that matches the majority normalized answer
    # FUTURE CLEANUP: Prefer returning a non-fallback generation if multiple 
    # duplicates exist among the samples. Currently this just takes the first match, 
    # which might be a fallback (and thus get excluded) even if a clean duplicate exists.
    for i, g in enumerate(generations):
        if normalize_answer(g) == majority_norm_ans:
            return i
            
    return 0 # Fallback, shouldn't happen

def main():
    set_seed(42)
    in_file = "data/labeled/triviaqa_finetune_labeled.jsonl"
    out_train = "data/processed/train.jsonl"
    out_val = "data/processed/val.jsonl"
    
    data = []
    with open(in_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    random.shuffle(data)
    
    # Simple 90/10 split
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    def write_sft_data(subset, out_path):
        written = 0
        with open(out_path, "w") as f:
            for item in subset:
                # -------------------------------------------------------------
                # MAJORITY VOTE AS TARGET
                # The prompt explicitly required using the majority-vote answer
                # as the target so the model learns its typical accuracy rate.
                # -------------------------------------------------------------
                chosen_idx = get_majority_vote_idx(item["generations"])
                
                if item["is_fallbacks"][chosen_idx]:
                    # Exclude this example to preserve SFT signal purity
                    continue
                    
                chosen_answer = item["generations"][chosen_idx]
                k = item["k"]
                
                # Instruction format (this can be formatted later using a ChatTemplate in train.py)
                # But here we provide the raw user/assistant turns
                # Strip the "Answer: " prefix if it exists to avoid double prepending
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
