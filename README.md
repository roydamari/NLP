# LLM Calibration Pipeline

This repository implements the pipeline for extracting calibration (confidence) statements from large language models.

## Infrastructure Setup (Slurm)

To run on the cluster, ensure you follow these steps:
1. **SlurmRequestForm**: Fill this out before submitting heavy jobs to ensure access and tracking.
2. **Partition**: Submit jobs to the `studentkillable` partition as defined in the `.slurm` scripts.
3. **HF Cache**: Do not use the default `~/.cache` directory. The `.slurm` scripts explicitly export `HF_HOME=/home/morg/NLP_2526b/$USER/.cache/huggingface` to prevent filling up home directory storage.
4. **Checkpoints**: The `train.py` script is configured to `save_total_limit=2` to avoid over-saving checkpoints and exhausting disk space.

## Running the Pipeline

You can run the full pipeline sequentially on a small dry run locally:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

For the actual scale experiments on Slurm, submit the stages sequentially:
```bash
sbatch scripts/run_sampling.slurm
sbatch scripts/run_finetune.slurm
sbatch scripts/run_eval.slurm
```

## Sanity Checks
- **Leakage**: `data_prep.py` asserts no overlap between TriviaQA and Natural Questions.
- **Normalization/Alias Matching**: Run `python src/scoring.py` to generate `scoring_sanity.md` and visually verify label correctness before fine-tuning.
- **Overfitting**: Run `python src/train.py --sanity_check` to verify the model can overfit a 50-example batch.
