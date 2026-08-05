#!/bin/bash
#SBATCH --job-name=llm_finetune
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/finetune_%j.out

mkdir -p slurm_logs
source venv/bin/activate

# Set HF cache to persistent storage, not home directory!
export HF_HOME=/home/morg/NLP_2526b/$USER/.cache/huggingface

echo "Starting fine-tuning phase..."
python src/scoring.py
python src/build_finetune_data.py
python src/train.py
echo "Fine-tuning phase completed."
