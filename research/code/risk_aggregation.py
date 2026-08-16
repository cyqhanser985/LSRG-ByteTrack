# risk_aggregation.py
# ==============================================================================
# Monotonic Bounded Risk Aggregation Models & Extreme Separation Evaluation
#
# Part of LSRG-ByteTrack Research Workspace.
#
# This script implements and evaluates four [0, 1] monotonic bounded aggregation
# operators on calibrated causal risk components r = [r_weak, r_comp, r_swap]:
#
#   1. Max Baseline:     R_max(r) = max(r_weak, r_comp, r_swap)
#   2. Power Mean:       R_pmean(r; p, w) = ( \sum w_i r_i^p )^(1/p), p >= 1
#   3. Noisy-OR:         R_noisy_or(r; w) = 1 - \prod (1 - w_i r_i), w_i \in [0, 1]
#   4. OWA:              R_owa(r; v) = \sum v_i r_(i),  r_(1) >= r_(2) >= r_(3)
#
# Optimization Target:
#   \max_\theta \min_{x \in IDS} R_\theta(x)  (Maximizing worst-case IDS separation)
#
# Evaluation Protocols:
#   - Oracle: Global parameter optimization on all 114 sequences.
#   - Test:   5-Fold Cross-Validation strictly partitioned by sequence.
#
# Metrics Evaluated:
#   - FPR @ TPR = 100% (min IDS threshold)
#   - FPR @ TPR = 99%
#   - FPR @ TPR = 95%
#   - pAUC [0.95, 1.0] (Normalized partial Area Under ROC Curve)
#   - Class breakdown across S_c, S_r, S_h
#
# Outputs:
#   - research/taxonomy/risk_aggregation_summary.csv
#   - research/taxonomy/risk_aggregation_roc.png
#
# Pure stdlib + numpy + scipy + matplotlib(Agg); ASCII comments only.
# Run in-place with the bytetrack conda interpreter.
# ==============================================================================

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as A

ROOT = A._repo_root()
TAXONOMY_DIR = os.path.join(ROOT, "research", "taxonomy")
REPORTS_DIR = os.path.join(ROOT, "research", "reports")

_t0 = time.time()


def log(msg):
    print("[+%6.1fs] %s" % (time.time() - _t0, msg))


# --------------------------------------------------------------------------
# Aggregation Operators (Vectorized [N, 3] -> [N,])
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


def agg_max(r, params=None):
    """Max baseline: R = max(r_1, r_2, r_3)."""
    return _chunked_apply(lambda x: np.max(x, axis=1).astype(np.float32), r)


def agg_power_mean(r, params):
    """Generalized power mean: R = ( \sum w_i r_i^p )^(1/p)."""
    p = float(params.get("p", 4.0))
    w = np.asarray(params.get("w", [1/3., 1/3., 1/3.]), dtype=np.float32)
    def _calc(sub):
        weighted_sum = np.sum(w[None, :] * np.power(np.clip(sub, 0.0, 1.0), p), axis=1)
        return np.clip(np.power(weighted_sum, 1.0 / p), 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)


def agg_noisy_or(r, params):
    """Noisy-OR probabilistic model: R = 1 - \prod (1 - w_i r_i)."""
    w = np.asarray(params.get("w", [1.0, 1.0, 1.0]), dtype=np.float32)
    def _calc(sub):
        # 1 - (1 - w1*r1)(1 - w2*r2)(1 - w3*r3) computed column-by-column to save memory
        term1 = 1.0 - w[0] * sub[:, 0]
        term2 = 1.0 - w[1] * sub[:, 1]
        term3 = 1.0 - w[2] * sub[:, 2]
        prob_none = term1 * term2 * term3
        return np.clip(1.0 - prob_none, 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)


def agg_owa(r, params):
    """Ordered Weighted Averaging (OWA): R = \sum v_i r_(i)."""
    v = np.asarray(params.get("v", [0.8, 0.15, 0.05]), dtype=np.float32)
    def _calc(sub):
        r_sorted = np.sort(sub, axis=1)[:, ::-1]  # descending order
        return np.clip(np.sum(r_sorted * v[None, :], axis=1), 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)


MODEL_FNS = {
    "Max": agg_max,
    "PowerMean": agg_power_mean,
    "Noisy-OR": agg_noisy_or,
    "OWA": agg_owa
}


# --------------------------------------------------------------------------
# Parameter Optimization on Training Positives (IDS Events)
# --------------------------------------------------------------------------
def fit_model_parameters(name, train_pos):
    """Optimize parameters maximizing worst-case / extreme separation:
    \max_\theta \min_{x \in IDS} R_\theta(x).
    """
    if name == "Max":
        return {}

    elif name == "PowerMean":
        best_val = -1.0
        best_p = 4.0
        best_w = np.array([1/3., 1/3., 1/3.])
        grid_p = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]
        # Candidate weights (covering uniform, single-feature, and balanced combinations)
        weights_pool = [
            np.array([1/3., 1/3., 1/3.]),
            np.array([0.50, 0.25, 0.25]),
            np.array([0.25, 0.50, 0.25]),
            np.array([0.25, 0.25, 0.50]),
            np.array([0.60, 0.20, 0.20]),
            np.array([0.40, 0.40, 0.20]),
            np.array([0.40, 0.20, 0.40]),
            np.array([0.20, 0.40, 0.40]),
            np.array([0.70, 0.15, 0.15]),
            np.array([0.80, 0.10, 0.10]),
        ]
        for p in grid_p:
            for w in weights_pool:
                scores = agg_power_mean(train_pos, {"p": p, "w": w})
                min_s = scores.min()
                if min_s > best_val:
                    best_val = min_s
                    best_p = p
                    best_w = w
        return {"p": best_p, "w": best_w, "opt_min": best_val}

    elif name == "Noisy-OR":
        best_val = -1.0
        best_w = np.array([1.0, 1.0, 1.0])
        w_grid = [0.2, 0.4, 0.6, 0.8, 1.0]
        for w1 in w_grid:
            for w2 in w_grid:
                for w3 in w_grid:
                    w = np.array([w1, w2, w3])
                    scores = agg_noisy_or(train_pos, {"w": w})
                    min_s = scores.min()
                    if min_s > best_val:
                        best_val = min_s
                        best_w = w
        return {"w": best_w, "opt_min": best_val}

    elif name == "OWA":
        best_val = -1.0
        best_v = np.array([1.0, 0.0, 0.0])
        v_pool = [
            np.array([1.00, 0.00, 0.00]),
            np.array([0.90, 0.075, 0.025]),
            np.array([0.85, 0.1125, 0.0375]),
            np.array([0.80, 0.15, 0.05]),
            np.array([0.75, 0.1875, 0.0625]),
            np.array([0.70, 0.225, 0.075]),
            np.array([0.60, 0.30, 0.10]),
            np.array([0.50, 0.375, 0.125]),
            np.array([1/3., 1/3., 1/3.]),
        ]
        for v in v_pool:
            scores = agg_owa(train_pos, {"v": v})
            min_s = scores.min()
            if min_s > best_val:
                best_val = min_s
                best_v = v
        return {"v": best_v, "opt_min": best_val}


# --------------------------------------------------------------------------
# Evaluation Metrics (FPR @ TPR=60%..100% in 5% steps & pAUC[0.60, 1.0])
# --------------------------------------------------------------------------
TPR_STEPS_5PCT = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]


def evaluate_risk_predictions(pos_scores, neg_scores):
    """Compute high-recall metrics and partial AUC on risk predictions."""
    pos_arr = np.asarray(pos_scores, dtype=np.float64)
    neg_arr = np.asarray(neg_scores, dtype=np.float64)
    n_neg = len(neg_arr)

    # Thresholds and FPR at 5% TPR steps
    step_metrics = {}
    for tpr in TPR_STEPS_5PCT:
        q = 100.0 * (1.0 - tpr)
        thr = float(np.percentile(pos_arr, q)) if tpr < 1.0 else float(pos_arr.min())
        fpr = float((neg_arr >= thr).mean())
        step_metrics[tpr] = {"thr": thr, "fpr": fpr}

    # Dense TPR grid from 0.60 to 1.00 (81 points)
    tpr_grid_60 = np.linspace(0.60, 1.0, 81)
    q_grid_60 = 100.0 * (1.0 - tpr_grid_60)
    thr_grid_60 = np.percentile(pos_arr, q_grid_60)
    fpr_grid_60 = np.array([(neg_arr >= t).mean() for t in thr_grid_60], dtype=np.float64)

    # Normalized pAUC over [0.60, 1.00]
    norm_pauc_60 = float(np.trapz(1.0 - fpr_grid_60, tpr_grid_60) / 0.40)
    mean_fpr_pauc_60 = float(np.trapz(fpr_grid_60, tpr_grid_60) / 0.40)

    # Normalized pAUC over [0.95, 1.00]
    mask_95 = tpr_grid_60 >= 0.95
    tpr_sub = tpr_grid_60[mask_95]
    fpr_sub = fpr_grid_60[mask_95]
    norm_pauc_95 = float(np.trapz(1.0 - fpr_sub, tpr_sub) / (tpr_sub[-1] - tpr_sub[0]))
    mean_fpr_pauc_95 = float(np.trapz(fpr_sub, tpr_sub) / (tpr_sub[-1] - tpr_sub[0]))

    return {
        "min_pos": step_metrics[1.00]["thr"],
        "thr_99": step_metrics[0.99]["thr"],
        "thr_95": step_metrics[0.95]["thr"],
        "fpr_100": step_metrics[1.00]["fpr"],
        "fpr_99": step_metrics[0.99]["fpr"],
        "fpr_95": step_metrics[0.95]["fpr"],
        "step_metrics": step_metrics,
        "norm_pauc_60": norm_pauc_60,
        "mean_fpr_pauc_60": mean_fpr_pauc_60,
        "norm_pauc_95": norm_pauc_95,
        "mean_fpr_pauc_95": mean_fpr_pauc_95,
        "norm_pauc": norm_pauc_95,
        "mean_fpr_pauc": mean_fpr_pauc_95,
        "tpr_grid": tpr_grid_60,
        "fpr_grid": fpr_grid_60
    }


# --------------------------------------------------------------------------
# Plotting High-Recall Comparison Curves (TPR 60% to 100%)
# --------------------------------------------------------------------------
def plot_aggregation_curves(oracle_evals, test_evals, out_png_path):
    """Plot high-resolution ROC and tradeoff curves for TPR in [60%, 100%]."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    colors = {
        "Max": "#2b5c8f",
        "PowerMean": "#d95f02",
        "Noisy-OR": "#7570b3",
        "OWA": "#1b9e77"
    }
    styles = {
        "Max": "-",
        "PowerMean": "--",
        "Noisy-OR": "-.",
        "OWA": ":"
    }
    markers = {
        "Max": "o",
        "PowerMean": "s",
        "Noisy-OR": "^",
        "OWA": "d"
    }

    # Left Plot: Full High-Recall Curve (TPR in [60%, 100%] vs FPR)
    ax1 = axes[0]
    for name, res in test_evals.items():
        tpr = res["tpr_grid"] * 100.0
        fpr = res["fpr_grid"] * 100.0
        ax1.plot(
            fpr, tpr,
            label=f"{name} (pAUC[0.6,1]={res['norm_pauc_60']:.4f}, pAUC[0.95,1]={res['norm_pauc_95']:.4f})",
            color=colors[name],
            linestyle=styles[name],
            linewidth=2.4
        )

    ax1.set_title("Test (5-Fold CV) ROC Curves: Detection Rate $\in [60\%, 100\%]$", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("False Positive Rate (%) on Normal Negatives", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate / Detection Rate (%) on IDS Events", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#cccccc")
    ax1.set_ylim(58.0, 101.5)
    ax1.set_xlim(-1.0, 102.0)

    # Right Plot: FPR vs Detection Rate (TPR 60% -> 100% in 5% steps)
    ax2 = axes[1]
    steps = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]
    steps_pct = [t * 100.0 for t in steps]

    for name, res in test_evals.items():
        fpr_steps = [res["step_metrics"][t]["fpr"] * 100.0 for t in steps]
        ax2.plot(
            steps_pct, fpr_steps,
            label=f"{name} (FPR at TPR=95%: {res['step_metrics'][0.95]['fpr']*100:.1f}%)",
            color=colors[name],
            linestyle=styles[name],
            marker=markers[name],
            markersize=6,
            linewidth=2.2
        )

    ax2.set_title("Test (5-Fold CV) False Positive Rate vs Detection Rate (5% Steps)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Target Detection Rate / TPR (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("False Positive Rate / FPR (%) (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_xticks(steps_pct)
    ax2.set_xticklabels([f"{t*100:.0f}%" if t < 0.99 else ("99%" if t == 0.99 else "100%") for t in steps], fontsize=8.5)
    ax2.set_xlim(58.0, 102.0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#cccccc")
    ax2.set_ylim(-2.0, 102.0)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300)
    plt.close()
    log("Saved updated comparative TPR 60%%-100%% plot: %s" % out_png_path)


# --------------------------------------------------------------------------
# Main Execution Pipeline
# --------------------------------------------------------------------------
def main():
    log("Starting risk aggregation model evaluation pipeline...")

    # 1. Load Precomputed Data
    events_npz_path = os.path.join(TAXONOMY_DIR, "risk_features_events.npz")
    negatives_npz_path = os.path.join(TAXONOMY_DIR, "risk_features_negatives.npz")

    if not os.path.isfile(events_npz_path):
        sys.exit("Error: %s not found. Run risk_features.py first." % events_npz_path)
    if not os.path.isfile(negatives_npz_path):
        sys.exit("Error: %s not found. Run test_neg_cache.py or extract negatives first." % negatives_npz_path)

    pos_data = np.load(events_npz_path)
    neg_data = np.load(negatives_npz_path)

    pos_risk = pos_data["risk_matrix"]       # [4713, 3]
    pos_seqs = pos_data["seq"]
    pos_ds = pos_data["dataset"]
    pos_cls = pos_data["class_labels"]

    neg_risk = neg_data["risk_matrix"]       # [1647180, 3]
    neg_seq_names = neg_data["seq_names"]
    neg_seq_datasets = neg_data["seq_datasets"]
    neg_seq_starts = neg_data["seq_starts"]
    neg_seq_ends = neg_data["seq_ends"]

    neg_seq_slices = {}
    for name, ds, st, en in zip(neg_seq_names, neg_seq_datasets, neg_seq_starts, neg_seq_ends):
        neg_seq_slices[(str(ds), str(name))] = (int(st), int(en))

    n_events = len(pos_risk)
    n_negs = len(neg_risk)
    log("Loaded %d events and %d negative detections." % (n_events, n_negs))

    # 2. Sequence-Level 5-Fold Stratification
    unique_seqs = []
    for ds in ["MOT17", "MOT20", "SportsMOT"]:
        pos_s_set = set(pos_seqs[pos_ds == ds])
        neg_s_set = set(s for (d, s) in neg_seq_slices.keys() if d == ds)
        s_list = sorted(list(pos_s_set | neg_s_set))
        unique_seqs.append((ds, s_list))

    np.random.seed(42)  # Deterministic seed for reproducible CV split
    folds = [[] for _ in range(5)]
    for ds, s_list in unique_seqs:
        shuffled = np.random.permutation(s_list)
        for i, s in enumerate(shuffled):
            folds[i % 5].append((ds, s))

    log("5-Fold cross validation configured with sequence counts: %s" % str([len(f) for f in folds]))

    # 3. Model Names
    model_names = ["Max", "PowerMean", "Noisy-OR", "OWA"]

    # 4. Oracle (Global Full Dataset) Evaluation
    log("Running Oracle evaluation (global parameter optimization)...")
    oracle_evals = {}
    oracle_params = {}

    for name in model_names:
        params = fit_model_parameters(name, pos_risk)
        oracle_params[name] = params
        p_scores = MODEL_FNS[name](pos_risk, params)
        n_scores = MODEL_FNS[name](neg_risk, params)
        eval_res = evaluate_risk_predictions(p_scores, n_scores)
        oracle_evals[name] = eval_res
        log("  [Oracle] %-10s -> Min(IDS)=%.4f, FPR@100%%=%.2f%%, FPR@99%%=%.2f%%, FPR@95%%=%.2f%%, pAUC[0.6,1]=%.4f, pAUC[0.95,1]=%.4f" % (
            name, eval_res["min_pos"], eval_res["fpr_100"] * 100.0,
            eval_res["fpr_99"] * 100.0, eval_res["fpr_95"] * 100.0,
            eval_res["norm_pauc_60"], eval_res["norm_pauc_95"]
        ))

    # 5. Test (5-Fold Cross-Validation by Sequence) Evaluation
    log("Running Test evaluation (5-Fold cross validation partitioned by sequence)...")
    test_evals = {}
    class_breakdown = {cls: {} for cls in ["S_c", "S_r", "S_h"]}
    class_names = ["S_c", "S_r", "S_h"]

    for name in model_names:
        log("  Evaluating %s across 5 folds..." % name)
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
            params_k = fit_model_parameters(name, pos_train_sub)

            p_test_all[pos_test_mask] = MODEL_FNS[name](pos_risk[pos_test_mask], params_k)
            n_test_all[neg_test_mask] = MODEL_FNS[name](neg_risk[neg_test_mask], params_k)

        eval_res = evaluate_risk_predictions(p_test_all, n_test_all)
        test_evals[name] = eval_res
        log("  [Test]   %-10s -> Min(IDS)=%.4f, FPR@100%%=%.2f%%, FPR@99%%=%.2f%%, FPR@95%%=%.2f%%, pAUC[0.6,1]=%.4f, pAUC[0.95,1]=%.4f" % (
            name, eval_res["min_pos"], eval_res["fpr_100"] * 100.0,
            eval_res["fpr_99"] * 100.0, eval_res["fpr_95"] * 100.0,
            eval_res["norm_pauc_60"], eval_res["norm_pauc_95"]
        ))

        # Class breakdown for this model
        for cls in class_names:
            c_mask = np.array([c == cls for c in pos_cls])
            class_breakdown[cls][name] = evaluate_risk_predictions(p_test_all[c_mask], n_test_all)

    # --------------------------------------------------------------------------
    # Export Deliverables
    # --------------------------------------------------------------------------
    # 1. Summary CSV Table
    summary_csv_path = os.path.join(TAXONOMY_DIR, "risk_aggregation_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "protocol", "parameters", "min_ids_score",
            "fpr_at_tpr_100", "fpr_at_tpr_99", "fpr_at_tpr_95",
            "norm_pauc_60_100", "norm_pauc_95_100", "mean_fpr_pauc_95"
        ])
        for name in model_names:
            # Oracle
            o_res = oracle_evals[name]
            writer.writerow([
                name, "Oracle", str(oracle_params[name]),
                "%.6f" % o_res["min_pos"],
                "%.4f%%" % (o_res["fpr_100"] * 100.0),
                "%.4f%%" % (o_res["fpr_99"] * 100.0),
                "%.4f%%" % (o_res["fpr_95"] * 100.0),
                "%.6f" % o_res["norm_pauc_60"],
                "%.6f" % o_res["norm_pauc_95"],
                "%.4f%%" % (o_res["mean_fpr_pauc_95"] * 100.0)
            ])
            # Test
            t_res = test_evals[name]
            writer.writerow([
                name, "Test (5-Fold CV)", "CV-Fold-Optimized",
                "%.6f" % t_res["min_pos"],
                "%.4f%%" % (t_res["fpr_100"] * 100.0),
                "%.4f%%" % (t_res["fpr_99"] * 100.0),
                "%.4f%%" % (t_res["fpr_95"] * 100.0),
                "%.6f" % t_res["norm_pauc_60"],
                "%.6f" % t_res["norm_pauc_95"],
                "%.4f%%" % (t_res["mean_fpr_pauc_95"] * 100.0)
            ])
    log("Saved aggregation summary CSV: %s" % summary_csv_path)

    # 2. Detailed 5% Stepped TPR Grid CSV
    grid_csv_path = os.path.join(TAXONOMY_DIR, "risk_aggregation_tpr_grid.csv")
    with open(grid_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["tpr_target"]
        for name in model_names:
            header.extend([f"{name}_Oracle_FPR", f"{name}_Test_FPR"])
        writer.writerow(header)
        for tpr in TPR_STEPS_5PCT:
            tpr_label = f"{tpr*100:.1f}%" if tpr < 0.99 else ("99.0%" if tpr == 0.99 else "100.0%")
            row = [tpr_label]
            for name in model_names:
                o_fpr = oracle_evals[name]["step_metrics"][tpr]["fpr"] * 100.0
                t_fpr = test_evals[name]["step_metrics"][tpr]["fpr"] * 100.0
                row.extend(["%.4f%%" % o_fpr, "%.4f%%" % t_fpr])
            writer.writerow(row)
    log("Saved 5%% stepped TPR grid CSV: %s" % grid_csv_path)

    # 3. High-Recall ROC Curves PNG (TPR in 60%..100%)
    roc_png_path = os.path.join(TAXONOMY_DIR, "risk_aggregation_roc.png")
    plot_aggregation_curves(oracle_evals, test_evals, roc_png_path)

    # --------------------------------------------------------------------------
    # Formatted Console Table Output
    # --------------------------------------------------------------------------
    print("\n" + "=" * 94)
    print("DETAILED 5% STEPPED TPR BENCHMARK TABLE (TPR = 60% -> 100% vs FPR)")
    print("=" * 94)
    print("%-12s | %-16s | %-16s | %-16s | %-16s" % (
        "TPR Target", "Max (Test FPR)", "PowerMean (Test)", "Noisy-OR (Test)", "OWA (Test)"
    ))
    print("-" * 94)
    for tpr in TPR_STEPS_5PCT:
        tpr_label = f"{tpr*100:5.1f}%" if tpr < 0.99 else (" 99.0%" if tpr == 0.99 else "100.0%")
        m_fpr = test_evals["Max"]["step_metrics"][tpr]["fpr"] * 100.0
        p_fpr = test_evals["PowerMean"]["step_metrics"][tpr]["fpr"] * 100.0
        n_fpr = test_evals["Noisy-OR"]["step_metrics"][tpr]["fpr"] * 100.0
        o_fpr = test_evals["OWA"]["step_metrics"][tpr]["fpr"] * 100.0
        print("%-12s | %15.2f%% | %15.2f%% | %15.2f%% | %15.2f%%" % (
            tpr_label, m_fpr, p_fpr, n_fpr, o_fpr
        ))
    print("=" * 94)

    print("\nCLASS-WISE TEST (5-FOLD CV) 5% STEPPED TPR METRICS:")
    print("-" * 94)
    for cls in class_names:
        print(f"\n--- [{cls} Failure Class] ---")
        print("%-12s | %-16s | %-16s | %-16s | %-16s" % (
            "TPR Target", "Max (Test FPR)", "PowerMean (Test)", "Noisy-OR (Test)", "OWA (Test)"
        ))
        print("-" * 94)
        for tpr in [0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00]:
            tpr_label = f"{tpr*100:5.1f}%" if tpr < 0.99 else (" 99.0%" if tpr == 0.99 else "100.0%")
            m_fpr = class_breakdown[cls]["Max"]["step_metrics"][tpr]["fpr"] * 100.0
            p_fpr = class_breakdown[cls]["PowerMean"]["step_metrics"][tpr]["fpr"] * 100.0
            n_fpr = class_breakdown[cls]["Noisy-OR"]["step_metrics"][tpr]["fpr"] * 100.0
            o_fpr = class_breakdown[cls]["OWA"]["step_metrics"][tpr]["fpr"] * 100.0
            print("%-12s | %15.2f%% | %15.2f%% | %15.2f%% | %15.2f%%" % (
                tpr_label, m_fpr, p_fpr, n_fpr, o_fpr
            ))
    print("=" * 94 + "\n")


if __name__ == "__main__":
    main()
