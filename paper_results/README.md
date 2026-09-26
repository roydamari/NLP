# Results used in the paper

- `all_experiments_summary.md`: every run, with the numbers in Tables 1-2 (Runs 1-4).
- `final_run/`: the replicate of Run 3's configuration (Sec. 4.2 accuracy, Sec. 4.4 expected-value
  readout, Table 3, Figure 2). `results.md` / `results.json` contain accuracy, ECE, MSE, AUROC,
  paired-bootstrap 95% CIs and histograms. `training_labels_n20.jsonl` is the sampled training data.
- `excluded_label_smoothing_run/`: EXCLUDED, kept only for transparency.

Raw outputs of Runs 1-3 were lost when the cluster home directory was wiped; their numbers are
recorded in `all_experiments_summary.md`. Figure 1 is built from Run 1's recorded counts.
