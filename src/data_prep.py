import os
import json
import yaml
from datasets import load_dataset
from utils import set_seed, normalize_answer

def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/labeled", exist_ok=True)

def main():
    config = load_config()
    set_seed(config["seed"])
    ensure_dirs()
    
    print("Loading TriviaQA (rc.nocontext)...")
    trivia = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    
    # Process train split for fine-tuning
    trivia_train = trivia["train"]
    train_subset_size = config.get("train_subset_size", 5000)
    # Shuffle and select
    trivia_train = trivia_train.shuffle(seed=config["seed"]).select(range(min(train_subset_size, len(trivia_train))))
    
    # Process validation split for in-distribution testing
    trivia_test = trivia["validation"]
    test_subset_size = config.get("test_subset_size", 1000)
    trivia_test = trivia_test.shuffle(seed=config["seed"]).select(range(min(test_subset_size, len(trivia_test))))
    
    print("Loading WebQuestions (OOD)...")
    webq = load_dataset("stanfordnlp/web_questions", split="test")
    ood_subset_size = config.get("ood_subset_size", 1000)
    webq_sub = webq.shuffle(seed=config["seed"]).select(range(min(ood_subset_size, len(webq))))
    
    # Leakage check logic: assert zero overlap between OOD questions and TriviaQA
    print("Running leakage check...")
    trivia_train_qs = {normalize_answer(item["question"]) for item in trivia_train}
    trivia_test_qs = {normalize_answer(item["question"]) for item in trivia_test}
    webq_qs = {normalize_answer(item["question"]) for item in webq_sub}
    
    overlap_train = trivia_train_qs.intersection(webq_qs)
    overlap_test = trivia_test_qs.intersection(webq_qs)
    assert len(overlap_train) == 0, f"LEAKAGE DETECTED: {len(overlap_train)} questions overlap between TriviaQA train and WebQuestions!"
    assert len(overlap_test) == 0, f"LEAKAGE DETECTED: {len(overlap_test)} questions overlap between TriviaQA test and WebQuestions!"
    print("Leakage check passed! No question overlap detected.")

    # Save processed data
    print("Saving processed datasets...")
    with open("data/processed/triviaqa_finetune.jsonl", "w") as f:
        for item in trivia_train:
            aliases = item["answer"]["aliases"]
            f.write(json.dumps({
                "id": item["question_id"],
                "question": item["question"],
                "answers": {
                    "aliases": aliases,
                    "normalized_aliases": [normalize_answer(a) for a in aliases]
                }
            }) + "\n")
            
    with open("data/processed/triviaqa_test.jsonl", "w") as f:
        for item in trivia_test:
            aliases = item["answer"]["aliases"]
            f.write(json.dumps({
                "id": item["question_id"],
                "question": item["question"],
                "answers": {
                    "aliases": aliases,
                    "normalized_aliases": [normalize_answer(a) for a in aliases]
                }
            }) + "\n")
            
    with open("data/processed/ood_test.jsonl", "w") as f:
        for i, item in enumerate(webq_sub):
            aliases = item["answers"]
            
            f.write(json.dumps({
                "id": f"webq_{i}",
                "question": item["question"],
                "answers": {
                    "aliases": aliases,
                    "normalized_aliases": [normalize_answer(a) for a in aliases]
                }
            }) + "\n")
    print("Done data preparation.")

if __name__ == "__main__":
    main()
