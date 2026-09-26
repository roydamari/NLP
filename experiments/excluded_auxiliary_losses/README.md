# Excluded: auxiliary confidence losses (DO NOT USE)

This folder keeps the code for the tokenized-Brier-score and label-smoothing experiments
for transparency. These runs are **excluded from the paper** (see the Limitations section).

**Bug:** the auxiliary loss read `logits[:, conf_idx]`, which in a causal LM predicts the token
*after* the confidence digit. The correct position is `conf_idx - 1`. As a result:
- the Brier term fell to about 0 within about 15 steps,
- the smoothing loss reached its theoretical minimum (the entropy of the target),
- in runs that also masked the digit from the LM loss, the digit got no supervision at all, and
  the model fell back to the base model's "90%" style (parsed as 8/9).

The exploratory 8B run (Run 4) also used this code, so it is not a valid test of model scale.
The valid training script is `src/train_plain.py`.
