# Final results for paper - label smoothing approach

Config: Llama-3.2-3B-Instruct, n_samples=20, lora_r=8, lr=3e-4, epochs=3
Approach: label smoothing on confidence digit (see train.py compute_loss)

Files:
- indist_results.json / ood_results.json: final ECE/MSE numbers (baseline vs finetuned)
- finetuned_indist.jsonl / finetuned_ood.jsonl / baseline_indist.jsonl / baseline_ood.jsonl: raw model outputs
- scoring_sanity.md: manual verification of alias-matching correctness
- triviaqa_finetune_labeled.jsonl: full k-distribution labels used for training data
- config_used.yaml: exact config for this run
- final_training_log.txt / final_eval_log.txt: full slurm logs incl. smooth_loss values per step

Key finding: confidence collapsed to 8/9 (not 0/10 like earlier attempts), still not
a smooth 0-10 distribution. See conversation history / notes for the 5 other approaches
tried (Brier loss, model scale, LR/epochs, more samples, digit-mapping bug fixes).
