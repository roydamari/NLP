#!/bin/bash
# Full pipeline on a machine with a GPU. On the TAU Slurm cluster, use the scripts in scripts/ instead (see README).
set -e
python -u src/data_prep.py          # TriviaQA (ID) + WebQuestions (OOD), leakage check
python -u src/sampling.py           # n_samples answers per training question
python -u src/scoring.py            # k/n correct -> labels
python -u src/build_finetune_data.py
python -u src/train_plain.py        # LoRA SFT, keeps best validation checkpoint
python -u src/evaluate_expected.py  # baseline + fine-tuned: greedy AND expected-value confidence
python -u src/analyze_final.py      # accuracy, ECE, MSE, AUROC, bootstrap CIs, figures
