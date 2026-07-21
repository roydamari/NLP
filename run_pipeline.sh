#!/bin/bash
set -e

echo "Running end-to-end LLM Calibration Pipeline..."

# 1. Data Prep & Sampling
python src/data_prep.py
python src/sampling.py

# 2. Scoring & Build Finetune Data
python src/scoring.py
python src/build_finetune_data.py

# 3. Fine-tuning
python src/train.py

# 4. Evaluation
python src/baseline_prompt.py
python src/evaluate.py

echo "Pipeline finished successfully!"
