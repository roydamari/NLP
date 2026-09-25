"""
Computes every number and figure needed for the paper from outputs/eval_expected/*.jsonl.
No GPU needed. Writes to paper_results/final_run/:
  results.json, results.md, and (if matplotlib is installed) fig_collapse_final.pdf, fig_reliability.pdf
Reports, for baseline vs fine-tuned, in-distribution and OOD, and for both readouts
(greedy = stated confidence, expected = mean of the 0..10 distribution):
  answer accuracy, ECE, MSE (Brier), AUROC, paired-bootstrap 95% CIs for FT - baseline.
"""
import os, re, sys, json
import numpy as np
sys.path.insert(0, "src")
from metrics import compute_ece, compute_mse

IN_DIR = "outputs/eval_expected"
OUT_DIR = "paper_results/final_run"
N_BOOT = 2000
rng = np.random.default_rng(0)

def load(cond, ds):
    with open(os.path.join(IN_DIR, f"{cond}_{ds}.jsonl")) as f:
        return {r["question_id"]: r for r in (json.loads(l) for l in f)}

def auroc(conf, y):
    """Probability a random correct item gets higher confidence than a random incorrect one (ties = 0.5)."""
    conf, y = np.asarray(conf, float), np.asarray(y, int)
    pos, neg = conf[y == 1], conf[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))

def conf_array(recs, readout):
    if readout == "greedy":
        return np.array([r["greedy_conf"] / 10.0 for r in recs])
    return np.array([r["expected_conf"] / 10.0 for r in recs])

def usable(recs, readout):
    return [r for r in recs if readout == "expected" or r["greedy_conf"] is not None]

def metric_block(recs, readout):
    recs = usable(recs, readout)
    if not recs:
        return {"n": 0}
    c = conf_array(recs, readout); y = np.array([int(r["correct"]) for r in recs])
    acc = float(y.mean())
    return {"n": len(recs), "accuracy_on_these": acc, "mean_conf": float(c.mean()),
            "ece": float(compute_ece(c, y)), "mse": float(compute_mse(c, y)),
            "auroc": auroc(c, y), "constant_predictor_mse": acc * (1 - acc),
            "distinct_values": int(len(np.unique(np.round(c, 3))))}

def paired_bootstrap(base, ft, readout):
    """95% CI of (FT - baseline) for ECE and MSE, resampling the same questions for both."""
    qids = [q for q in base if q in ft and base[q] in usable([base[q]], readout) and ft[q] in usable([ft[q]], readout)]
    if len(qids) < 20:
        return {"n_paired": len(qids)}
    cb = conf_array([base[q] for q in qids], readout); yb = np.array([int(base[q]["correct"]) for q in qids])
    cf = conf_array([ft[q] for q in qids], readout);   yf = np.array([int(ft[q]["correct"]) for q in qids])
    diffs = {"ece": [], "mse": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(qids), len(qids))
        diffs["ece"].append(compute_ece(cf[idx], yf[idx]) - compute_ece(cb[idx], yb[idx]))
        diffs["mse"].append(compute_mse(cf[idx], yf[idx]) - compute_mse(cb[idx], yb[idx]))
    out = {"n_paired": len(qids)}
    for m in ("ece", "mse"):
        point = (compute_ece if m == "ece" else compute_mse)(cf, yf) - (compute_ece if m == "ece" else compute_mse)(cb, yb)
        lo, hi = np.percentile(diffs[m], [2.5, 97.5])
        out[f"{m}_diff"] = float(point); out[f"{m}_ci95"] = [float(lo), float(hi)]
    return out

def reliability(recs, readout, bins=10):
    recs = usable(recs, readout)
    c = conf_array(recs, readout); y = np.array([int(r["correct"]) for r in recs])
    edges = np.linspace(0, 1, bins + 1); rows = []
    for i in range(bins):
        m = (c > edges[i]) & (c <= edges[i + 1]) if i else (c >= 0) & (c <= edges[1])
        if m.sum():
            rows.append({"bin": [float(edges[i]), float(edges[i + 1])], "count": int(m.sum()),
                         "mean_conf": float(c[m].mean()), "accuracy": float(y[m].mean())})
    return rows

def training_label_hist():
    counts = [0] * 11
    with open("data/processed/train.jsonl") as f:
        for l in f:
            m = re.search(r"My confidence is (\d+) out of 10", json.loads(l)["messages"][1]["content"])
            if m and 0 <= int(m.group(1)) <= 10:
                counts[int(m.group(1))] += 1
    return counts

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"per_condition": {}, "bootstrap_ft_minus_baseline": {}, "reliability": {}, "histograms": {}}
    for ds in ("indist", "ood"):
        base, ft = load("baseline", ds), load("finetuned", ds)
        for cond, data in (("baseline", base), ("finetuned", ft)):
            recs = list(data.values())
            entry = {"answer_accuracy_all": float(np.mean([r["correct"] for r in recs])),
                     "n_total": len(recs),
                     "n_unparsed_greedy": sum(r["greedy_conf"] is None for r in recs),
                     "n_forced_prefix": sum(r["forced_prefix"] for r in recs),
                     "mean_mass_on_conf_tokens": float(np.mean([r["conf_token_mass"] for r in recs]))}
            for readout in ("greedy", "expected"):
                entry[readout] = metric_block(recs, readout)
            results["per_condition"][f"{cond}_{ds}"] = entry
            g = [0] * 11
            for r in recs:
                if r["greedy_conf"] is not None:
                    g[int(r["greedy_conf"])] += 1
            e = [0] * 11
            for r in recs:
                e[int(np.floor(r["expected_conf"] + 0.5))] += 1
            results["histograms"][f"{cond}_{ds}"] = {"greedy": g, "expected_rounded": e}
            for readout in ("greedy", "expected"):
                results["reliability"][f"{cond}_{ds}_{readout}"] = reliability(recs, readout)
        for readout in ("greedy", "expected"):
            results["bootstrap_ft_minus_baseline"][f"{ds}_{readout}"] = paired_bootstrap(base, ft, readout)
    # extra comparison: fine-tuned expected readout vs baseline greedy (the paper's original baseline)
    for ds in ("indist", "ood"):
        base, ft = load("baseline", ds), load("finetuned", ds)
        qids = [q for q in base if q in ft and base[q]["greedy_conf"] is not None]
        cb = np.array([base[q]["greedy_conf"] / 10 for q in qids]); yb = np.array([int(base[q]["correct"]) for q in qids])
        cf = np.array([ft[q]["expected_conf"] / 10 for q in qids]); yf = np.array([int(ft[q]["correct"]) for q in qids])
        d = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(qids), len(qids))
            d.append(compute_ece(cf[idx], yf[idx]) - compute_ece(cb[idx], yb[idx]))
        results["bootstrap_ft_minus_baseline"][f"{ds}_FTexpected_vs_Bgreedy"] = {
            "n_paired": len(qids), "ece_diff": float(compute_ece(cf, yf) - compute_ece(cb, yb)),
            "ece_ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
    results["histograms"]["training_labels"] = training_label_hist()
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # human-readable summary
    L = ["# Final run results", "", "| Set | Model | Readout | n | Accuracy (all) | ECE | MSE | AUROC | distinct values |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ds in ("indist", "ood"):
        for cond in ("baseline", "finetuned"):
            e = results["per_condition"][f"{cond}_{ds}"]
            for readout in ("greedy", "expected"):
                m = e[readout]
                if m.get("n", 0) == 0:
                    continue
                L.append(f"| {ds} | {cond} | {readout} | {m['n']} | {e['answer_accuracy_all']:.3f} | {m['ece']:.3f} | "
                         f"{m['mse']:.3f} | {m['auroc']:.3f} | {m['distinct_values']} |")
    L += ["", "## Paired bootstrap, fine-tuned minus baseline (negative = fine-tuned better)", ""]
    for k, v in results["bootstrap_ft_minus_baseline"].items():
        L.append(f"- {k}: " + json.dumps(v))
    L += ["", "## Histograms (counts for values 0..10)", ""]
    for k, v in results["histograms"].items():
        L.append(f"- {k}: " + json.dumps(v))
    L += ["", "## Unparsed / forced counts", ""]
    for k, v in results["per_condition"].items():
        L.append(f"- {k}: unparsed greedy = {v['n_unparsed_greedy']}/{v['n_total']}, forced prefix = "
                 f"{v['n_forced_prefix']}, mean prob. mass on 0-10 tokens = {v['mean_mass_on_conf_tokens']:.3f}")
    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: skipped figures; results.json has everything needed)")
        return
    v = np.arange(11); tr = np.array(results["histograms"]["training_labels"], float); tr /= max(tr.sum(), 1)
    g = np.array(results["histograms"]["finetuned_indist"]["greedy"], float); g /= max(g.sum(), 1)
    ex = np.array(results["histograms"]["finetuned_indist"]["expected_rounded"], float); ex /= max(ex.sum(), 1)
    fig, ax = plt.subplots(figsize=(3.3, 2.2)); w = 0.28
    ax.bar(v - w, tr, w, label="Training labels"); ax.bar(v, g, w, label="FT stated (greedy)")
    ax.bar(v + w, ex, w, label="FT expected (rounded)")
    ax.set_xticks(v); ax.set_xlabel("Confidence (out of 10)"); ax.set_ylabel("Proportion"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_collapse_final.pdf"))
    fig, ax = plt.subplots(figsize=(3.3, 2.6)); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for key, lab in (("baseline_indist_greedy", "Baseline (stated)"), ("finetuned_indist_greedy", "FT (stated)"),
                     ("finetuned_indist_expected", "FT (expected)")):
        rows = results["reliability"][key]
        ax.plot([r["mean_conf"] for r in rows], [r["accuracy"] for r in rows], "o-", ms=3, label=lab)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_reliability.pdf"))
    print(f"\nFigures written to {OUT_DIR}")

if __name__ == "__main__":
    main()
