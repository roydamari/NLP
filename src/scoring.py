import json
from collections import Counter
from utils import check_match

def main():
    gen_file = "data/labeled/generations.jsonl"
    q_file = "data/processed/triviaqa_finetune.jsonl"
    out_file = "data/labeled/triviaqa_finetune_labeled.jsonl"
    
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
    
    with open(gen_file, "r") as f:
        for line in f:
            gen_item = json.loads(line)
            
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
                
                old_match, _ = check_match(g, gold_aliases, use_old_method=True, target_format="baseline")
                new_match, is_fallback = check_match(g, gold_aliases, use_old_method=False, target_format="baseline")
                
                if is_fallback:
                    fallback_count += 1
                fallbacks_for_q.append(is_fallback)
                
                if old_match != new_match:
                    disagreement_count += 1
                    
                if new_match:
                    correct_count += 1
                    
            labeled_data.append({
                "question_id": q_id,
                "question": gen_item["question"],
                "k": correct_count,
                "generations": generations,
                "gold_aliases": gold_aliases,
                "is_fallbacks": fallbacks_for_q
            })
            
            if len(sanity_examples) < 20:
                sanity_examples.append(labeled_data[-1])
            
    with open(out_file, "w") as f:
        for item in labeled_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Scored {len(labeled_data)} questions.")
    
    fallback_rate = (fallback_count / total_generations * 100) if total_generations > 0 else 0.0
    print(f"Diagnostic Disagreements (Old vs New Match): {disagreement_count} / {total_generations}")
    print(f"Fallback Parse Rate: {fallback_count} / {total_generations} ({fallback_rate:.1f}%)")
    
    # Write a small markdown sanity file for manual inspection
    sanity_file = "data/labeled/scoring_sanity.md"
    with open(sanity_file, "w") as f:
        f.write("# Scoring Sanity Check\n\n")
        f.write(f"Fallback Parse Rate: {fallback_count} / {total_generations} ({fallback_rate:.1f}%)\n\n")
        for i, item in enumerate(sanity_examples):
            f.write(f"## Q{i+1}: {item['question']}\n")
            f.write(f"**Gold aliases (sample):** {item['gold_aliases'][:3]}\n\n")
            f.write(f"**Score:** {item['k']}/10\n\n")
            f.write("**Generations:**\n")
            for g, is_fb in zip(item["generations"], item["is_fallbacks"]):
                matched, _ = check_match(g, item["gold_aliases"], target_format="baseline")
                mark = "✅" if matched else "❌"
                fb = " `[FALLBACK]`" if is_fb else ""
                f.write(f"- {mark}{fb} `{g}`\n")
            f.write("\n")
    print(f"Sanity check written to {sanity_file}.")
if __name__ == '__main__':
    main()
