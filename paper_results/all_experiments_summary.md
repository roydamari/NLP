# Summary of all experiments (chronological)

| # | Experiment | Model | n_samples | LR | Epochs | In-dist ECE (FT) | OOD ECE (FT) | Confidence distribution |
|---|---|---|---|---|---|---|---|---|
| 1 | Baseline fine-tune | 3B | 10 | 3e-4 | 3 | 0.21 | 0.395 | {10: 262, 0: 238} |
| 2 | Lower LR / fewer epochs | 3B | 10 | 1e-4 | 2 | 0.235 | 0.412 | {10: 281, 0: 217} |
| 3 | More samples/question | 3B | 20 | 3e-4 | 3 | 0.2153 | 0.4067 | {10: 340, 0: 157} |
| 4 | Larger model (QLoRA) | 8B | 20 | 1e-5 | 2 | n/a (malformed 500/500) | n/a | model stopped stating confidence entirely |
| 5 | Brier loss (ConfTuner-style), bug-fixed | 3B | 20 | 3e-4 | 3 | n/a | n/a | smooth/brier_loss collapsed to 0.0000 by step 15 (memorization) |
| 6 | Label smoothing on confidence digit (FINAL) | 3B | 20 | 3e-4 | 3 | 0.2826 | 0.3994 | {9: 282, 8: 217, 1: 1} |

Baseline (untrained, zero-shot "Just Ask for Calibration") reference values, run 6 config:
- In-dist: ECE 0.2666, MSE 0.2986, n=500, malformed=0, fallback=58
- OOD: ECE 0.3922, MSE 0.3906, n=497, malformed=3, fallback=51

## Bugs found and fixed along the way
1. Alias-matching false positives (model listing multiple candidates before answering)
2. Instruction echoing in generations (small model repeating the prompt format instructions)
3. Answer extraction bug for fine-tuned model outputs (searched for wrong marker)
4. k-value scaling bug when n_samples changed from 10 to 20
5. Digit-token mapping bug in Brier loss (used space token instead of actual digit token)
6. LM cross-entropy loss not masked from confidence digit, competing with calibration loss
7. parse_confidence() didn't handle percentage/out-of-100 formats the label-smoothed model learned to use

## Key finding
Confidence collapse to a narrow set of values (initially 0/10, later 8/9 after label smoothing)
is a robust, persistent phenomenon, consistent with independently published findings on
LLMs of this parameter scale (3-9B). Neither data volume, learning rate/epoch tuning, a
larger model, nor a proper-scoring-rule loss function (Brier score) fixed it; label smoothing
measurably changed the collapse point and kept the loss non-trivial throughout training,
but did not produce genuinely graded (smooth) calibration.
