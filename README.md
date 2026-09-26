# Teaching LLMs to Calibrate Themselves: Aligning Verbalized Confidence with Empirical Accuracy

NLP course project, Tel Aviv University, 2025/26: Roy Damari, Liri Even Or, Romi Polani, Roy Ofer.

## Research question
Can a small instruction-tuned LLM learn to state *calibrated* confidence if we fine-tune it on its
own empirical accuracy?

## Method
1. Sample each of 1000 TriviaQA training questions `n` times (temperature 1.0) with Llama-3.2-3B-Instruct.
2. Count the `k` correct samples (normalized alias containment) and set the label `round(10k/n)`.
3. Fine-tune (LoRA) on: `"<modal answer>. My confidence is <label> out of 10."`
4. Compare with a zero-shot verbalized-confidence baseline (following Tian et al., 2023) on
   TriviaQA (in-distribution, ID) and WebQuestions (out-of-distribution, OOD), measuring
   ECE, MSE (Brier), AUROC and answer accuracy.

## Main findings
- **Pre-registered run (Run 1):** ID ECE 0.265 -> 0.210 (success criterion met); OOD effectively unchanged (0.397 -> 0.395).
- **Confidence collapse:** 36% of training labels are intermediate (1-9), but the model states
  only 0 or 10. With binary outputs, ECE = MSE = the error rate of a correct/incorrect classifier (79% agreement).
- **Accuracy (replicate):** ID 59.8% -> 59.8% (unchanged); OOD 45.6% -> 42.6%.
- **Expected-value readout (replicate):** the mean of the model's probability distribution over the
  tokens 0-10 is graded (369 distinct values) and well calibrated: ID ECE 0.036 vs 0.239 for the
  stated confidence; paired bootstrap dECE = -0.216, 95% CI [-0.246, -0.147]. The collapse appears
  to be largely a property of greedy decoding.
- The collapse persisted with n=20 and with a lower learning rate / fewer epochs. OOD calibration never improved.

## Experiments
| Run | Config | Status |
|---|---|---|
| 1 | `configs/runs/run1_preregistered.yaml` | valid, pre-registered (Table 1) |
| 2 | `configs/runs/run2_low_lr.yaml` | valid, post-hoc (Table 2) |
| 3 | `configs/runs/run3_n20.yaml` | valid, post-hoc (Table 2); replicate in `paper_results/final_run/` |
| 4 | `configs/runs/run4_8b_EXCLUDED.yaml` | **excluded** (buggy auxiliary loss) |
| Brier / label smoothing | `experiments/excluded_auxiliary_losses/` | **excluded** (indexing bug) |

## Repository layout
- `src/data_prep.py`: datasets and leakage check. `src/sampling.py`: repeated sampling.
- `src/scoring.py`: correctness and labels (writes a manual-inspection file `scoring_sanity.md`).
- `src/build_finetune_data.py`: modal answer + label -> SFT data (90/10 train/val).
- `src/train_plain.py`: LoRA SFT (best validation checkpoint kept).
- `src/evaluate_expected.py`: baseline + fine-tuned evaluation, stated (greedy) and expected-value confidence.
  The expected value is computed by one extra forward pass on the generated text up to
  "confidence is" (or with "My confidence is" appended if the model never wrote it), reading the
  probabilities of the tokens 0-10 at the next position.
- `src/analyze_final.py`: accuracy, ECE, MSE, AUROC, paired-bootstrap CIs, figures.
- `src/baseline_prompt.py`, `src/evaluate.py`: the original greedy-only evaluation used for Tables 1-2.
- `paper_results/`: results used in the paper (see its README).

## Reproducing (TAU Slurm cluster)
Setup: create a venv in persistent storage (`/home/morg/NLP_2526b/$USER/venv`), `pip install -r requirements.txt`,
accept the Llama license on Hugging Face and put `export HF_TOKEN=...` in `~/.hf_token_env`.
Note: on this cluster, torch must match the node driver (we used `torch==2.5.1+cu121`).

```bash
cp configs/runs/run3_n20.yaml configs/base.yaml          # or run1_preregistered / run2_low_lr
mkdir -p slurm_logs
sbatch --exclude=<broken nodes> scripts/run_sampling.slurm  # data prep + sampling (~1 h for n=20)
sbatch --exclude=<broken nodes> scripts/run_final.slurm     # scoring, SFT data, training, evaluation, analysis
```
`run_final.slurm` writes everything for Table 3 / Figure 2 to `paper_results/final_run/`.
For the original greedy-only metrics of Tables 1-2, run `scripts/run_finetune.slurm` then `scripts/run_eval.slurm`.
Greedy bf16 evaluation varied by about 0.01 ECE between cluster nodes (paper, Appendix B).
