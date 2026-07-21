import os
import json
from collections import Counter
from utils import check_match

def main():
    gen_dir = "data/labeled/generations_raw"
    q_file = "data/processed/triviaqa_finetune_split.jsonl"
    out_file = "data/labeled/triviaqa_finetune_labeled.jsonl"
    sanity_file = "data/labeled/scoring_sanity.md"
    
    # Load gold aliases
    q_data = {}
    with open(q_file, "r") as f:
        for line in f:
            item = json.loads(line)
            q_data[item["id"]] = item
            
    labeled_data = []
    sanity_examples = []
    
    total_generations = 0
    fallback_count = 0
    disagreement_count = 0
    
    for filename in os.listdir(gen_dir):
        if not filename.endswith(".json"):
            continue
        
        with open(os.path.join(gen_dir, filename), "r") as f:
            gen_item = json.load(f)
            
        q_id = gen_item["question_id"]
        if q_id not in q_data:
            continue
            
        gold_aliases = q_data[q_id]["answers"]["aliases"]
        generations = gen_item["generations"]
        
        # Score each generation
        correct_count = 0
        fallbacks_for_q = []
        for g in generations:
            total_generations += 1
            
            # Diagnostic: old score
            old_match, _ = check_match(g, gold_aliases, use_old_method=True)
            
            # New score
            new_match, is_fallback = check_match(g, gold_aliases, use_old_method=False)
            
            if is_fallback:
                fallback_count += 1
            fallbacks_for_q.append(is_fallback)
            
            if old_match != new_match:
                disagreement_count += 1
                
            if new_match:
                correct_count += 1
                
        # Calculate k/10 accuracy
        k = correct_count
        
        labeled_item = {
            "question_id": q_id,
            "question": gen_item["question"],
            "k": k,
            "generations": generations,
            "gold_aliases": gold_aliases,
            "is_fallbacks": fallbacks_for_q
        }
        labeled_data.append(labeled_item)
        
        # Save a few for sanity check
        if len(sanity_examples) < 50:
            sanity_examples.append(labeled_item)
            
    # Write full labeled data
    with open(out_file, "w") as f:
        for item in labeled_data:
            f.write(json.dumps(item) + "\n")
            
    # Write sanity check markdown
    with open(sanity_file, "w") as f:
        f.write("# Scoring Sanity Check\n\n")
        f.write("Review these 50 examples to ensure normalization and alias matching are working correctly.\n\n")
        for i, item in enumerate(sanity_examples):
            f.write(f"## Example {i+1}\n")
            f.write(f"**Question:** {item['question']}\n\n")
            f.write(f"**Gold Aliases:** {item['gold_aliases']}\n\n")
            f.write(f"**Computed k (Score):** {item['k']}/10\n\n")
            f.write("**Generations:**\n")
            for j, g in enumerate(item['generations']):
                matched, is_fb = check_match(g, item['gold_aliases'])
                mark = "✅" if matched else "❌"
                fb_mark = " [FALLBACK]" if is_fb else ""
                f.write(f"- [{mark}]{fb_mark} {g}\n")
            f.write("\n---\n\n")
            
    print(f"Scored {len(labeled_data)} questions.")
    
    if total_generations > 0:
        fallback_rate = (fallback_count / total_generations) * 100
    else:
        fallback_rate = 0.0
        
    print(f"Diagnostic Disagreements (Old vs New Match): {disagreement_count} / {total_generations}")
    print(f"Fallback Parse Rate: {fallback_count} / {total_generations} ({fallback_rate:.1f}%)")
    print(f"Sanity check written to {sanity_file}.")

if __name__ == "__main__":
    main()
