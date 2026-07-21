#!/bin/bash
#SBATCH --job-name=llm_eval
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/eval_%j.out

# Set HF cache to persistent storage, not home directory!
export HF_HOME=/home/morg/NLP_2526b/$USER/.cache/huggingface

echo "Starting evaluation phase..."
python src/baseline_prompt.py
python src/evaluate.py
echo "Evaluation phase completed."
