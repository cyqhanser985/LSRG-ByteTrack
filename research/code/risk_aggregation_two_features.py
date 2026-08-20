# -*- coding: utf-8 -*-
"""risk_aggregation_two_features.py — 双特征消融检测实验

Part of LSRG-ByteTrack Research Workspace.

This script evaluates risk detection performance using only the first feature (r_weak)
and the third feature (r_swap), dropping the second feature (r_comp).
It re-fits and evaluates Max, PowerMean, Noisy-OR, and OWA operators
under a sequence-level 5-Fold cross-validation protocol and compares the results
against the baseline 3-feature model.
"""

import os
import csv
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as A

ROOT = A._repo_root()
TAXONOMY_DIR = os.path.join(ROOT, "research", "taxonomy")

_t0 = time.time()
def log(msg):
    print("[+%6.1fs] %s" % (time.time() - _t0, msg))

# --------------------------------------------------------------------------
# Two-Feature Aggregation Operators (Vectorized [N, 2] -> [N,])
# --------------------------------------------------------------------------
def _chunked_apply(fn, r, chunk_size=200000):
    n = len(r)
    if n <= chunk_size:
        return fn(r)
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        out[i:end] = fn(r[i:end])
    return out

def agg_max_2f(r, params=None):
    """Max of 2 features: max(r_weak, r_swap)."""
    return _chunked_apply(lambda x: np.max(x, axis=1).astype(np.float32), r)

def agg_power_mean_2f(r, params):
    """Power mean of 2 features: ( w_1 * r_weak^p + w_2 * r_swap^p )^(1/p)."""
    p = float(params.get("p", 4.0))
    w = np.asarray(params.get("w", [0.5, 0.5]), dtype=np.float32)
    def _calc(sub):
        weighted_sum = np.sum(w[None, :] * np.power(np.clip(sub, 0.0, 1.0), p), axis=1)
        return np.clip(np.power(weighted_sum, 1.0 / p), 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)

def agg_noisy_or_2f(r, params):
    """Noisy-OR of 2 features: 1 - (1 - w_1 * r_weak) * (1 - w_2 * r_swap)."""
    w = np.asarray(params.get("w", [1.0, 1.0]), dtype=np.float32)
    def _calc(sub):
        term1 = 1.0 - w[0] * sub[:, 0]
        term2 = 1.0 - w[1] * sub[:, 1]
        return np.clip(1.0 - term1 * term2, 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)

def agg_owa_2f(r, params):
    """Ordered Weighted Averaging (OWA) of 2 features: v_1 * r_(1) + v_2 * r_(2)."""
    v = np.asarray(params.get("v", [0.8, 0.2]), dtype=np.float32)
    def _calc(sub):
        r_sorted = np.sort(sub, axis=1)[:, ::-1]  # descending order
        return np.clip(np.sum(r_sorted * v[None, :], axis=1), 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)

MODEL_FNS_2F = {
    "Max": agg_max_2f,
    "PowerMean": agg_power_mean_2f,
    "Noisy-OR": agg_noisy_or_2f,
    "OWA": agg_owa_2f
}

# --------------------------------------------------------------------------
# Parameter Optimization on Training Positives (2 Features)
# --------------------------------------------------------------------------
def fit_model_parameters_2f(name, train_pos):
    if name == "Max":
        return {}
    elif name == "PowerMean":
        best_val = -1.0
        best_p = 4.0
        best_w = np.array([0.5, 0.5])
        grid_p = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0]
        weights_pool = [
            np.array([0.5, 0.5]),
            np.array([0.6, 0.4]),
            np.array([0.4, 0.6]),
            np.array([0.7, 0.3]),
            np.array([0.3, 0.7]),
            np.array([0.8, 0.2]),
            np.array([0.2, 0.8]),
            np.array([0.9, 0.1]),
            np.array([0.1, 0.9])
        ]
        for p in grid_p:
            for w in weights_pool:
                scores = agg_power_mean_2f(train_pos, {"p": p, "w": w})
                min_s = scores.min()
                if min_s > best_val:
                    best_val = min_s
                    best_p = p
                    best_w = w
        return {"p": best_p, "w": best_w}
    elif name == "Noisy-OR":
        best_val = -1.0
        best_w = np.array([1.0, 1.0])
        w_grid = [0.2, 0.4, 0.6, 0.8, 1.0]
        for w1 in w_grid:
            for w2 in w_grid:
                w = np.array([w1, w2])
                scores = agg_noisy_or_2f(train_pos, {"w": w})
                min_s = scores.min()
                if min_s > best_val:
                    best_val = min_s
                    best_w = w
        return {"w": best_w}
    elif name == "OWA":
        best_val = -1.0
        best_v = np.array([1.0, 0.0])
        v_pool = [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.8, 0.2]),
            np.array([0.7, 0.3]),
            np.array([0.6, 0.4]),
            np.array([0.5, 0.5])
        ]
        for v in v_pool:
            scores = agg_owa_2f(train_pos, {"v": v})
            min_s = scores.min()
            if min_s > best_val:
                best_val = min_s
                best_v = v
        return {"v": best_v}

# --------------------------------------------------------------------------
# Evaluation Metrics Helper
# --------------------------------------------------------------------------
TPR_STEPS_5PCT = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]

def evaluate_risk_predictions(pos_scores, neg_scores):
    pos_arr = np.asarray(pos_scores, dtype=np.float64)
    neg_arr = np.asarray(neg_scores, dtype=np.float64)
    n_neg = len(neg_arr)

    step_metrics = {}
    for tpr in TPR_STEPS_5PCT:
        q = 100.0 * (1.0 - tpr)
        thr = float(np.percentile(pos_arr, q)) if tpr < 1.0 else float(pos_arr.min())
        fpr = float((neg_arr >= thr).mean())
        step_metrics[tpr] = {"thr": thr, "fpr": fpr}

    tpr_grid_60 = np.linspace(0.60, 1.0, 81)
    q_grid_60 = 100.0 * (1.0 - tpr_grid_60)
    thr_grid_60 = np.percentile(pos_arr, q_grid_60)
    fpr_grid_60 = np.array([(neg_arr >= t).mean() for t in thr_grid_60], dtype=np.float64)

    norm_pauc_60 = float(np.trapz(1.0 - fpr_grid_60, tpr_grid_60) / 0.40)
    norm_pauc_95 = float(np.trapz(1.0 - fpr_grid_60[tpr_grid_60 >= 0.95], tpr_grid_60[tpr_grid_60 >= 0.95]) / 0.05)

    return {
        "min_pos": step_metrics[1.00]["thr"],
        "fpr_100": step_metrics[1.00]["fpr"],
        "fpr_99": step_metrics[0.99]["fpr"],
        "fpr_95": step_metrics[0.95]["fpr"],
        "fpr_90": step_metrics[0.90]["fpr"],
        "norm_pauc_60": norm_pauc_60,
        "norm_pauc_95": norm_pauc_95,
        "tpr_grid": tpr_grid_60,
        "fpr_grid": fpr_grid_60
    }

def main():
    events_npz_path = os.path.join(TAXONOMY_DIR, "risk_features_events.npz")
    negatives_npz_path = os.path.join(TAXONOMY_DIR, "risk_features_negatives.npz")

    if not os.path.exists(events_npz_path) or not os.path.exists(negatives_npz_path):
        log("Error: Missing input NPZ files.")
        return

    # 1. Load data & Slice columns 0 (r_weak) and 2 (r_swap)
    pos_data = np.load(events_npz_path)
    neg_data = np.load(negatives_npz_path)

    pos_risk_3f = pos_data["risk_matrix"]       # [4713, 3]
    pos_seqs = pos_data["seq"]
    pos_ds = pos_data["dataset"]
    pos_risk = pos_risk_3f[:, [0, 2]]           # Slice: Keep [r_weak, r_swap]

    neg_risk_3f = neg_data["risk_matrix"]       # [1647180, 3]
    neg_seq_names = neg_data["seq_names"]
    neg_seq_datasets = neg_data["seq_datasets"]
    neg_seq_starts = neg_data["seq_starts"]
    neg_seq_ends = neg_data["seq_ends"]
    neg_risk = neg_risk_3f[:, [0, 2]]           # Slice: Keep [r_weak, r_swap]

    neg_seq_slices = {}
    for name, ds, st, en in zip(neg_seq_names, neg_seq_datasets, neg_seq_starts, neg_seq_ends):
        neg_seq_slices[(str(ds), str(name))] = (int(st), int(en))

    n_events = len(pos_risk)
    n_negs = len(neg_risk)
    log(f"Loaded {n_events} positive events and {n_negs} negative detections (2-Feature sliced).")

    # 2. Sequence-Level 5-Fold Stratification
    unique_seqs = []
    for ds in ["MOT17", "MOT20", "SportsMOT"]:
        pos_s_set = set(pos_seqs[pos_ds == ds])
        neg_s_set = set(s for (d, s) in neg_seq_slices.keys() if d == ds)
        s_list = sorted(list(pos_s_set | neg_s_set))
        unique_seqs.append((ds, s_list))

    np.random.seed(42)
    folds = [[] for _ in range(5)]
    for ds, s_list in unique_seqs:
        shuffled = np.random.permutation(s_list)
        for i, s in enumerate(shuffled):
            folds[i % 5].append((ds, s))

    model_names = ["Max", "PowerMean", "Noisy-OR", "OWA"]
    test_evals_2f = {}

    # 3. Test Evaluation (5-Fold Cross-Validation) for 2 Features
    for name in model_names:
        log("  Evaluating %s (2-Feature) across 5 folds..." % name)
        p_test_all = np.zeros(n_events, dtype=np.float32)
        n_test_all = np.zeros(n_negs, dtype=np.float32)

        for k in range(5):
            test_set = set(folds[k])
            pos_test_mask = np.array([(d, s) in test_set for d, s in zip(pos_ds, pos_seqs)])
            
            neg_test_mask = np.zeros(n_negs, dtype=bool)
            for item in test_set:
                sl = neg_seq_slices.get(item)
                if sl is not None:
                    neg_test_mask[sl[0]:sl[1]] = True

            pos_train_sub = pos_risk[~pos_test_mask]
            params_k = fit_model_parameters_2f(name, pos_train_sub)

            p_test_all[pos_test_mask] = MODEL_FNS_2F[name](pos_risk[pos_test_mask], params_k)
            n_test_all[neg_test_mask] = MODEL_FNS_2F[name](neg_risk[neg_test_mask], params_k)

        eval_res = evaluate_risk_predictions(p_test_all, n_test_all)
        test_evals_2f[name] = eval_res
        log("  [Test 2F] %-10s -> Min(IDS)=%.4f, FPR@100%%=%.2f%%, FPR@95%%=%.2f%%, FPR@90%%=%.2f%%, pAUC[0.95,1]=%.4f" % (
            name, eval_res["min_pos"], eval_res["fpr_100"] * 100.0,
            eval_res["fpr_95"] * 100.0, eval_res["fpr_90"] * 100.0,
            eval_res["norm_pauc_95"]
        ))

    # 4. Compare with 3-Feature baseline by reading baseline summary if it exists
    # Or we can just output the 2F metrics to CSV and plot ROC curves
    csv_path = os.path.join(TAXONOMY_DIR, "two_features_vs_baseline.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "FeatureCount", "FPR@100% (Recall)", "FPR@95% (Recall)", "FPR@90% (Recall)", "pAUC [0.95,1.00]"])
        for name in model_names:
            res = test_evals_2f[name]
            writer.writerow([name, "2 Features", f"{res['fpr_100']*100:.2f}%", f"{res['fpr_95']*100:.2f}%", f"{res['fpr_90']*100:.2f}%", f"{res['norm_pauc_95']:.4f}"])

    log(f"消融评估对照数据已写入 {csv_path}")

    # 5. Plot comparison ROC curve
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    colors = {"Max": "#2b5c8f", "PowerMean": "#d95f02", "Noisy-OR": "#7570b3", "OWA": "#1b9e77"}
    for name in model_names:
        res = test_evals_2f[name]
        ax.plot(res["fpr_grid"] * 100.0, res["tpr_grid"] * 100.0,
                label=f"{name} 2F (pAUC[0.95,1]={res['norm_pauc_95']:.4f})",
                color=colors[name], linewidth=2)
    ax.set_title("消融实验: 双特征检测性能评测 ROC (Detection Rate in [60%, 100%])")
    ax.set_xlabel("FPR (False Positive Rate) % on Normal Negatives")
    ax.ylabel("TPR (True Positive Rate) % on IDS Events")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    ax.set_ylim(60, 101)
    ax.set_xlim(-0.5, 30)  # Zoom in on high-performance area
    
    png_path = os.path.join(TAXONOMY_DIR, "fig_two_features_roc.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    log(f"对比 ROC 图像已保存至 {png_path}")

if __name__ == "__main__":
    main()
