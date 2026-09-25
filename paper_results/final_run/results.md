# Final run results

| Set | Model | Readout | n | Accuracy (all) | ECE | MSE | AUROC | distinct values |
|---|---|---|---|---|---|---|---|---|
| indist | baseline | greedy | 492 | 0.598 | 0.271 | 0.302 | 0.628 | 6 |
| indist | baseline | expected | 493 | 0.598 | 0.252 | 0.283 | 0.730 | 206 |
| indist | finetuned | greedy | 493 | 0.598 | 0.239 | 0.239 | 0.745 | 2 |
| indist | finetuned | expected | 493 | 0.598 | 0.036 | 0.158 | 0.839 | 369 |
| ood | baseline | greedy | 497 | 0.456 | 0.393 | 0.392 | 0.601 | 8 |
| ood | baseline | expected | 500 | 0.456 | 0.372 | 0.372 | 0.634 | 214 |
| ood | finetuned | greedy | 453 | 0.426 | 0.471 | 0.470 | 0.568 | 3 |
| ood | finetuned | expected | 500 | 0.426 | 0.310 | 0.333 | 0.656 | 320 |

## Paired bootstrap, fine-tuned minus baseline (negative = fine-tuned better)

- indist_greedy: {"n_paired": 492, "ece_diff": -0.033130081300813186, "ece_ci95": [-0.08151422764227652, 0.0156504065040648], "mse_diff": -0.0638414634146342, "mse_ci95": [-0.10536839430894318, -0.01974695121951226]}
- indist_expected: {"n_paired": 493, "ece_diff": -0.21605181896251613, "ece_ci95": [-0.24619738083398726, -0.1471749798520399], "mse_diff": -0.12492793744569017, "mse_ci95": [-0.1553477401954602, -0.0957209199307564]}
- ood_greedy: {"n_paired": 452, "ece_diff": 0.07654867256637149, "ece_ci95": [0.034286504424778624, 0.11725663716814144], "mse_diff": 0.07836283185840698, "mse_ci95": [0.04088274336283182, 0.11462721238938052]}
- ood_expected: {"n_paired": 500, "ece_diff": -0.06169581425947068, "ece_ci95": [-0.09461425390072886, -0.025379357809046724], "mse_diff": -0.03976386486648764, "mse_ci95": [-0.06556755925067981, -0.015242105970275583]}
- indist_FTexpected_vs_Bgreedy: {"n_paired": 492, "ece_diff": -0.2358206510455641, "ece_ci95": [-0.2643135289982714, -0.16260227202497443]}
- ood_FTexpected_vs_Bgreedy: {"n_paired": 497, "ece_diff": -0.08032250543401753, "ece_ci95": [-0.11247470582444091, -0.04119115922734484]}

## Histograms (counts for values 0..10)

- baseline_indist: {"greedy": [12, 0, 0, 0, 0, 4, 1, 0, 126, 301, 48], "expected_rounded": [3, 2, 1, 2, 2, 5, 4, 22, 135, 297, 20]}
- finetuned_indist: {"greedy": [184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 309], "expected_rounded": [3, 31, 43, 55, 39, 55, 45, 35, 39, 75, 73]}
- baseline_ood: {"greedy": [2, 0, 7, 0, 0, 1, 6, 1, 209, 260, 11], "expected_rounded": [0, 0, 4, 1, 3, 5, 8, 34, 191, 246, 8]}
- finetuned_ood: {"greedy": [62, 0, 1, 0, 0, 0, 0, 0, 0, 0, 390], "expected_rounded": [9, 3, 7, 19, 24, 56, 40, 55, 80, 102, 105]}
- training_labels: [182, 61, 44, 39, 33, 23, 30, 28, 32, 45, 350]

## Unparsed / forced counts

- baseline_indist: unparsed greedy = 1/493, forced prefix = 1, mean prob. mass on 0-10 tokens = 1.000
- finetuned_indist: unparsed greedy = 0/493, forced prefix = 0, mean prob. mass on 0-10 tokens = 0.998
- baseline_ood: unparsed greedy = 3/500, forced prefix = 3, mean prob. mass on 0-10 tokens = 1.000
- finetuned_ood: unparsed greedy = 47/500, forced prefix = 44, mean prob. mass on 0-10 tokens = 0.998
