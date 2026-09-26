#!/bin/bash
#SBATCH --job-name=final_run
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/final_%j.out

source /home/morg/NLP_2526b/$USER/venv/bin/activate
[ -f ~/.hf_token_env ] && source ~/.hf_token_env
export HF_HOME=/home/morg/NLP_2526b/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1        # the model is already cached; avoids Hugging Face login problems
set -e
cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname)"
python -u src/scoring.py
python -u src/build_finetune_data.py
python -u src/train_plain.py
python -u src/evaluate_expected.py
python -u src/analyze_final.py
echo "FINAL RUN COMPLETE"
