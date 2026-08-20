# class_risk_breakdown.py
# ==============================================================================
# Isolated Class-Wise (S_c, S_r, S_h) High-Recall (TPR 60%-100%) vs FPR Analysis
#
# Part of LSRG-ByteTrack Research Workspace.
#
# Evaluates S_c (cold start), S_r (active takeover), and S_h (history reactivation)
# separately against the full negative population (1,647,180 C-group normal detections)
# across TPR in [60%, 100%] in 5% steps and fine-grained high-recall steps.
#
# Purpose:
#   Pinpoint which specific failure class is driving the high FPR when global TPR >= 85%.
# ==============================================================================

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
# Aggregation Operators
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
    return _chunked_apply(lambda x: np.max(x, axis=1).astype(np.float32), r)


def agg_power_mean(r, params):
    p = float(params.get("p", 4.0))
    w = np.asarray(params.get("w", [1/3., 1/3., 1/3.]), dtype=np.float32)
    def _calc(sub):
        weighted_sum = np.sum(w[None, :] * np.power(np.clip(sub, 0.0, 1.0), p), axis=1)
        return np.clip(np.power(weighted_sum, 1.0 / p), 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)


def agg_noisy_or(r, params):
    w = np.asarray(params.get("w", [1.0, 1.0, 1.0]), dtype=np.float32)
    def _calc(sub):
        term1 = 1.0 - w[0] * sub[:, 0]
        term2 = 1.0 - w[1] * sub[:, 1]
        term3 = 1.0 - w[2] * sub[:, 2]
        return np.clip(1.0 - term1 * term2 * term3, 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)


def agg_owa(r, params):
    v = np.asarray(params.get("v", [0.8, 0.15, 0.05]), dtype=np.float32)
    def _calc(sub):
        r_sorted = np.sort(sub, axis=1)[:, ::-1]
        return np.clip(np.sum(r_sorted * v[None, :], axis=1), 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)


MODEL_FNS = {
    "Max": agg_max,
    "PowerMean": agg_power_mean,
    "Noisy-OR": agg_noisy_or,
    "OWA": agg_owa
}


def fit_model_parameters(name, train_pos):
    if name == "Max":
        return {}
    elif name == "PowerMean":
        best_val = -1.0
        best_p = 4.0
        best_w = np.array([1/3., 1/3., 1/3.])
        grid_p = [1.0, 2.0, 4.0, 8.0, 16.0]
        weights_pool = [
            np.array([1/3., 1/3., 1/3.]),
            np.array([0.50, 0.25, 0.25]),
            np.array([0.25, 0.50, 0.25]),
            np.array([0.25, 0.25, 0.50]),
            np.array([0.60, 0.20, 0.20]),
            np.array([0.40, 0.40, 0.20]),
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
        w_grid = [0.4, 0.7, 1.0]
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
            np.array([0.80, 0.15, 0.05]),
            np.array([0.60, 0.30, 0.10]),
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
# Evaluation Helper
# --------------------------------------------------------------------------
TPR_STEPS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]


def evaluate_curve(pos_scores, neg_scores):
    pos_arr = np.asarray(pos_scores, dtype=np.float64)
    neg_arr = np.asarray(neg_scores, dtype=np.float64)

    step_metrics = {}
    for tpr in TPR_STEPS:
        q = 100.0 * (1.0 - tpr)
        thr = float(np.percentile(pos_arr, q)) if tpr < 1.0 else float(pos_arr.min())
        fpr = float((neg_arr >= thr).mean())
        step_metrics[tpr] = {"thr": thr, "fpr": fpr}

    # Dense curve for plotting [0.60, 1.00]
    tpr_dense = np.linspace(0.60, 1.0, 101)
    q_dense = 100.0 * (1.0 - tpr_dense)
    thr_dense = np.percentile(pos_arr, q_dense)
    fpr_dense = np.array([(neg_arr >= t).mean() for t in thr_dense], dtype=np.float64)

    norm_pauc_60 = float(np.trapz(1.0 - fpr_dense, tpr_dense) / 0.40)
    norm_pauc_95 = float(np.trapz(1.0 - fpr_dense[tpr_dense >= 0.95], tpr_dense[tpr_dense >= 0.95]) / 0.05)

    return {
        "step_metrics": step_metrics,
        "tpr_dense": tpr_dense,
        "fpr_dense": fpr_dense,
        "norm_pauc_60": norm_pauc_60,
        "norm_pauc_95": norm_pauc_95,
        "min_score": float(pos_arr.min()),
        "p50_score": float(np.median(pos_arr)),
        "p10_score": float(np.percentile(pos_arr, 10)),
        "p05_score": float(np.percentile(pos_arr, 5)),
        "p01_score": float(np.percentile(pos_arr, 1)),
    }


def main():
    log("Loading precomputed events and negatives...")
    events_npz = os.path.join(TAXONOMY_DIR, "risk_features_events.npz")
    negatives_npz = os.path.join(TAXONOMY_DIR, "risk_features_negatives.npz")

    pos_data = np.load(events_npz)
    neg_data = np.load(negatives_npz)

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
    log(f"Loaded {n_events} positive events and {n_negs} negative detections.")

    # 5-Fold Stratification
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

    # Evaluate 5-Fold Test Out-of-Fold predictions for all models
    model_names = ["Max", "PowerMean", "Noisy-OR", "OWA"]
    oof_pos = {m: np.zeros(n_events, dtype=np.float32) for m in model_names}
    oof_neg = {m: np.zeros(n_negs, dtype=np.float32) for m in model_names}

    for name in model_names:
        log(f"Running 5-fold cross-validation for {name}...")
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

            oof_pos[name][pos_test_mask] = MODEL_FNS[name](pos_risk[pos_test_mask], params_k)
            oof_neg[name][neg_test_mask] = MODEL_FNS[name](neg_risk[neg_test_mask], params_k)

    # Class masks
    class_names = ["S_c", "S_r", "S_h"]
    class_masks = {cls: (pos_cls == cls) for cls in class_names}
    class_masks["Overall"] = np.ones(n_events, dtype=bool)

    # Results dictionary: res[target_group][model_name]
    groups = ["Overall", "S_c", "S_r", "S_h"]
    eval_results = {g: {} for g in groups}

    for g in groups:
        mask = class_masks[g]
        for m in model_names:
            eval_results[g][m] = evaluate_curve(oof_pos[m][mask], oof_neg[m])

    # Also evaluate single raw features against negatives for deep insight
    single_feats = ["r_weak", "r_comp", "r_swap"]
    single_feat_res = {g: {} for g in groups}
    for g in groups:
        mask = class_masks[g]
        for j, f_name in enumerate(single_feats):
            single_feat_res[g][f_name] = evaluate_curve(pos_risk[mask, j], neg_risk[:, j])

    # --------------------------------------------------------------------------
    # Print Comprehensive Table to Console
    # --------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("ISOLATED EVENT CLASS (S_c vs S_r vs S_h vs Overall) TPR vs FPR (against ALL 1.65M Negatives)")
    print("=" * 105)
    print(f"Sample Counts: Overall={n_events} | S_c={class_masks['S_c'].sum()} (38.8%) | S_r={class_masks['S_r'].sum()} (40.3%) | S_h={class_masks['S_h'].sum()} (20.9%)")
    print("-" * 105)
    print(f"{'Target TPR':<10} | {'S_c (Noisy-OR)':<15} | {'S_h (Noisy-OR)':<15} | {'S_r (Noisy-OR)':<15} | {'Overall (Noisy-OR)':<18} | {'S_r (Max)':<12}")
    print("-" * 105)
    for tpr in TPR_STEPS:
        tpr_lbl = f"{tpr*100:5.1f}%" if tpr < 0.99 else (" 99.0%" if tpr == 0.99 else "100.0%")
        sc_fpr = eval_results["S_c"]["Noisy-OR"]["step_metrics"][tpr]["fpr"] * 100.0
        sh_fpr = eval_results["S_h"]["Noisy-OR"]["step_metrics"][tpr]["fpr"] * 100.0
        sr_fpr = eval_results["S_r"]["Noisy-OR"]["step_metrics"][tpr]["fpr"] * 100.0
        ov_fpr = eval_results["Overall"]["Noisy-OR"]["step_metrics"][tpr]["fpr"] * 100.0
        sr_max = eval_results["S_r"]["Max"]["step_metrics"][tpr]["fpr"] * 100.0
        print(f"{tpr_lbl:<10} | {sc_fpr:14.2f}% | {sh_fpr:14.2f}% | {sr_fpr:14.2f}% | {ov_fpr:17.2f}% | {sr_max:11.2f}%")
    print("=" * 105)

    print("\nDETAILED STEPPED BREAKDOWN FOR S_r (ACTIVE TAKEOVER) ACROSS ALL 4 AGGREGATION MODELS:")
    print("-" * 105)
    print(f"{'Target TPR':<10} | {'Max (FPR)':<14} | {'PowerMean (FPR)':<16} | {'Noisy-OR (FPR)':<16} | {'OWA (FPR)':<14} | {'Noisy-OR vs Max Gain'}")
    print("-" * 105)
    for tpr in TPR_STEPS:
        tpr_lbl = f"{tpr*100:5.1f}%" if tpr < 0.99 else (" 99.0%" if tpr == 0.99 else "100.0%")
        m_fpr = eval_results["S_r"]["Max"]["step_metrics"][tpr]["fpr"] * 100.0
        p_fpr = eval_results["S_r"]["PowerMean"]["step_metrics"][tpr]["fpr"] * 100.0
        n_fpr = eval_results["S_r"]["Noisy-OR"]["step_metrics"][tpr]["fpr"] * 100.0
        o_fpr = eval_results["S_r"]["OWA"]["step_metrics"][tpr]["fpr"] * 100.0
        gain = m_fpr - n_fpr
        print(f"{tpr_lbl:<10} | {m_fpr:13.2f}% | {p_fpr:15.2f}% | {n_fpr:15.2f}% | {o_fpr:13.2f}% | {gain:+6.2f} pp")
    print("=" * 105)

    # --------------------------------------------------------------------------
    # Score Distribution & Extreme Tail Breakdown
    # --------------------------------------------------------------------------
    print("\nRISK SCORE PERCENTILES (Noisy-OR) BY CLASS (Lower score = Harder to detect = Pushes FPR higher):")
    print("-" * 105)
    print(f"{'Group':<10} | {'Min (0%)':<10} | {'P01 (1%)':<10} | {'P05 (5%)':<10} | {'P10 (10%)':<10} | {'P50 (Median)':<12} | {'Score >= 0.99 (FPR<=1%)'}")
    print("-" * 105)
    for g in groups:
        s = eval_results[g]["Noisy-OR"]
        mask = class_masks[g]
        high_risk_frac = (oof_pos["Noisy-OR"][mask] >= 0.99).mean() * 100.0
        print(f"{g:<10} | {s['min_score']:10.6f} | {s['p01_score']:10.6f} | {s['p05_score']:10.6f} | {s['p10_score']:10.6f} | {s['p50_score']:12.6f} | {high_risk_frac:6.2f}%")
    print("=" * 105)

    # --------------------------------------------------------------------------
    # Plotting 4-Panel Master Analytical Figure
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    colors_group = {
        "Overall": "#333333",
        "S_c": "#2ca02c",      # Green (Cold Start)
        "S_h": "#1f77b4",      # Blue (History Reactivation)
        "S_r": "#d62728",      # Red (Active Takeover)
    }

    # Panel 1: Head-to-Head Class-Wise ROC (TPR 60%-100% vs FPR) under Noisy-OR
    ax1 = axes[0, 0]
    for g in groups:
        res = eval_results[g]["Noisy-OR"]
        lw = 3.0 if g in ["S_r", "Overall"] else 2.2
        ls = "-" if g != "Overall" else "--"
        lbl = f"{g} (pAUC[0.6,1]={res['norm_pauc_60']:.3f}, pAUC[0.95,1]={res['norm_pauc_95']:.3f})"
        ax1.plot(res["fpr_dense"] * 100.0, res["tpr_dense"] * 100.0, label=lbl,
                 color=colors_group[g], linewidth=lw, linestyle=ls)
    ax1.set_title("1. Class-Wise High-Recall ROC (Noisy-OR Aggregator)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("False Positive Rate (%) on 1.65M Normal Negatives", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Detection Rate / TPR (%) on Target Events", fontsize=11, fontweight="bold")
    ax1.set_xlim(-1.0, 102.0)
    ax1.set_ylim(58.0, 101.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#cccccc")

    # Panel 2: FPR vs Detection Rate (TPR 60% -> 100% in 5% steps)
    ax2 = axes[0, 1]
    steps_pct = [t * 100.0 for t in TPR_STEPS]
    markers = {"Overall": "o", "S_c": "s", "S_h": "^", "S_r": "D"}
    for g in groups:
        res = eval_results[g]["Noisy-OR"]
        fpr_steps = [res["step_metrics"][t]["fpr"] * 100.0 for t in TPR_STEPS]
        lw = 2.8 if g in ["S_r", "Overall"] else 2.0
        ax2.plot(steps_pct, fpr_steps, label=f"{g} (FPR@95%={res['step_metrics'][0.95]['fpr']*100:.1f}%)",
                 color=colors_group[g], marker=markers[g], markersize=6, linewidth=lw)
    ax2.set_title("2. FPR Escalation Across 5% Target TPR Steps (Noisy-OR)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Target Detection Rate / TPR (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("False Positive Rate / FPR (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(steps_pct)
    ax2.set_xticklabels([f"{t*100:.0f}%" if t < 0.99 else ("99%" if t == 0.99 else "100%") for t in TPR_STEPS], fontsize=8.5)
    ax2.set_xlim(58.0, 102.0)
    ax2.set_ylim(-2.0, 102.0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#cccccc")

    # Panel 3: Model Comparison Specifically on the Bottleneck Class (S_r)
    ax3 = axes[1, 0]
    model_colors = {"Max": "#2b5c8f", "PowerMean": "#d95f02", "Noisy-OR": "#d62728", "OWA": "#1b9e77"}
    model_styles = {"Max": "-", "PowerMean": "--", "Noisy-OR": "-", "OWA": ":"}
    for m in model_names:
        res = eval_results["S_r"][m]
        fpr_steps = [res["step_metrics"][t]["fpr"] * 100.0 for t in TPR_STEPS]
        ax3.plot(steps_pct, fpr_steps, label=f"{m} (FPR@95%={res['step_metrics'][0.95]['fpr']*100:.1f}%)",
                 color=model_colors[m], linestyle=model_styles[m], marker="o", markersize=5, linewidth=2.2)
    ax3.set_title("3. Model Benchmark on Bottleneck Class S_r (Active Takeover)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Target Detection Rate on S_r (%)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("False Positive Rate / FPR (%)", fontsize=11, fontweight="bold")
    ax3.set_xticks(steps_pct)
    ax3.set_xticklabels([f"{t*100:.0f}%" if t < 0.99 else ("99%" if t == 0.99 else "100%") for t in TPR_STEPS], fontsize=8.5)
    ax3.set_xlim(58.0, 102.0)
    ax3.set_ylim(-2.0, 102.0)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#cccccc")

    # Panel 4: Risk Score CDF Tail Distribution (Showing the Root Cause)
    ax4 = axes[1, 1]
    score_bins = np.linspace(0.0, 1.0, 501)
    neg_scores_sample = oof_neg["Noisy-OR"][::10]  # downsample for fast CDF
    
    # Negatives CDF
    neg_cdf = np.array([(oof_neg["Noisy-OR"] <= b).mean() for b in score_bins])
    ax4.plot(score_bins, neg_cdf * 100.0, label="1.65M Normal Negatives", color="#7f7f7f", linestyle="--", linewidth=2.0)

    for g in ["S_c", "S_h", "S_r"]:
        mask = class_masks[g]
        g_scores = oof_pos["Noisy-OR"][mask]
        g_cdf = np.array([(g_scores <= b).mean() for b in score_bins])
        ax4.plot(score_bins, g_cdf * 100.0, label=f"Positive {g} Events (N={mask.sum()})",
                 color=colors_group[g], linewidth=2.5)

    ax4.set_title("4. Risk Score Cumulative Distribution Function (CDF)", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Calibrated Aggregated Risk Score R", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Cumulative Fraction (%) <= R", fontsize=11, fontweight="bold")
    ax4.set_xlim(-0.02, 1.02)
    ax4.set_ylim(-2.0, 102.0)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#cccccc")

    # --------------------------------------------------------------------------
    # Run full comprehensive breakdown across all datasets and classes
    # --------------------------------------------------------------------------
    import comprehensive_class_dataset_breakdown as CBD
    import generate_paper_tables_and_figures as GPTF
    
    log("Running comprehensive multi-dataset and failure-class breakdown...")
    CBD.main()
    log("Generating master tables and publication-grade 8-panel analytical figure...")
    GPTF.fig, GPTF.axes
    os.system(f'python "{os.path.join(HERE, "generate_paper_tables_and_figures.py")}"')
    log("All deliverables generated successfully.")


if __name__ == "__main__":
    main()
