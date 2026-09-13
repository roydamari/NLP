#!/bin/bash
#SBATCH --job-name=llm_sampling
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/sampling_%j.out

mkdir -p slurm_logs
source /home/morg/NLP_2526b/$USER/venv/bin/activate
source ~/.hf_token_env

# Set HF cache to persistent storage, not home directory!
export HF_HOME=/home/morg/NLP_2526b/$USER/.cache/huggingface

echo "Starting sampling phase..."
python src/data_prep.py
python src/sampling.py
echo "Sampling phase completed."
