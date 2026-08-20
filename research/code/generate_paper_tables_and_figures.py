# generate_paper_tables_and_figures.py
# ==============================================================================
# Generates Publication-Grade Tables and 8-Panel Figures for Dataset & Class Risk Analysis
#
# Part of LSRG-ByteTrack Research Workspace.
# ==============================================================================

import json
import os
import sys
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

json_path = os.path.join(TAXONOMY_DIR, "class_dataset_breakdown_full.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

TPR_STEPS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]
DATASETS = ["MOT17", "MOT20", "SportsMOT", "Overall"]
CLASSES = ["Overall", "S_c", "S_r", "S_h"]
MODELS = ["Max", "PowerMean", "Noisy-OR", "OWA"]

# --------------------------------------------------------------------------
# 1. Generate Master CSV Tables
# --------------------------------------------------------------------------
# (a) Main Stepped TPR vs FPR Table across all Datasets x Classes (Noisy-OR Global)
tpr_master_rows = []
header = ["Dataset", "Class", "N_Events", "FPR@60%", "FPR@70%", "FPR@80%", "FPR@85%", "FPR@90%", "FPR@95%", "FPR@99%", "FPR@100%", "pAUC[0.6,1.0]", "pAUC[0.95,1.0]"]

csv_lines = [",".join(header)]

for ds in DATASETS:
    for cls in CLASSES:
        res = data[ds][cls]["Noisy-OR"]["global"]
        if res is None:
            continue
        sm = res["step_metrics"]
        n_pos = res["n_pos"]
        row = [
            ds, cls, str(n_pos),
            f"{sm['0.6']['fpr']*100:.2f}%",
            f"{sm['0.7']['fpr']*100:.2f}%",
            f"{sm['0.8']['fpr']*100:.2f}%",
            f"{sm['0.85']['fpr']*100:.2f}%",
            f"{sm['0.9']['fpr']*100:.2f}%",
            f"{sm['0.95']['fpr']*100:.2f}%",
            f"{sm['0.99']['fpr']*100:.2f}%",
            f"{sm['1.0']['fpr']*100:.2f}%",
            f"{res['norm_pauc_60']:.4f}",
            f"{res['norm_pauc_95']:.4f}"
        ]
        csv_lines.append(",".join(row))

out_csv = os.path.join(TAXONOMY_DIR, "class_dataset_breakdown_master_table.csv")
with open(out_csv, "w", encoding="utf-8") as f:
    f.write("\n".join(csv_lines))
print(f"Saved master CSV table: {out_csv}")

# (b) Score Distribution Percentiles Table
pct_header = ["Dataset", "Class", "N_Events", "Min", "P01", "P05", "P10", "P50(Median)", "P90", "P99", "Mean", "Std", "Score>=0.99(%)", "Score>=0.95(%)"]
pct_lines = [",".join(pct_header)]

for ds in DATASETS:
    for cls in CLASSES:
        res = data[ds][cls]["Noisy-OR"]["global"]
        if res is None:
            continue
        row = [
            ds, cls, str(res["n_pos"]),
            f"{res['min_score']:.6f}",
            f"{res['p01_score']:.6f}",
            f"{res['p05_score']:.6f}",
            f"{res['p10_score']:.6f}",
            f"{res['p50_score']:.6f}",
            f"{res['p90_score']:.6f}",
            f"{res['p99_score']:.6f}",
            f"{res['mean_score']:.6f}",
            f"{res['std_score']:.6f}",
            f"{res['frac_ge_099']*100:.2f}%",
            f"{res['frac_ge_095']*100:.2f}%"
        ]
        pct_lines.append(",".join(row))

out_pct_csv = os.path.join(TAXONOMY_DIR, "class_dataset_percentiles_table.csv")
with open(out_pct_csv, "w", encoding="utf-8") as f:
    f.write("\n".join(pct_lines))
print(f"Saved percentiles CSV table: {out_pct_csv}")

# (c) Intra-Dataset FPR Benchmark Table
intra_header = ["Dataset", "Class", "N_Events", "N_Neg_Intra", "Intra_FPR@60%", "Intra_FPR@80%", "Intra_FPR@90%", "Intra_FPR@95%", "Intra_FPR@99%", "Intra_pAUC[0.6,1.0]", "Intra_pAUC[0.95,1.0]"]
intra_lines = [",".join(intra_header)]

for ds in ["MOT17", "MOT20", "SportsMOT"]:
    for cls in CLASSES:
        res = data[ds][cls]["Noisy-OR"]["intra"]
        if res is None:
            continue
        sm = res["step_metrics"]
        row = [
            ds, cls, str(res["n_pos"]), str(res["n_neg"]),
            f"{sm['0.6']['fpr']*100:.2f}%",
            f"{sm['0.8']['fpr']*100:.2f}%",
            f"{sm['0.9']['fpr']*100:.2f}%",
            f"{sm['0.95']['fpr']*100:.2f}%",
            f"{sm['0.99']['fpr']*100:.2f}%",
            f"{res['norm_pauc_60']:.4f}",
            f"{res['norm_pauc_95']:.4f}"
        ]
        intra_lines.append(",".join(row))

out_intra_csv = os.path.join(TAXONOMY_DIR, "class_dataset_intra_benchmark_table.csv")
with open(out_intra_csv, "w", encoding="utf-8") as f:
    f.write("\n".join(intra_lines))
print(f"Saved intra benchmark CSV table: {out_intra_csv}")

# --------------------------------------------------------------------------
# 2. Generate Master 8-Panel High-Resolution Analytical Figure
# --------------------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(20, 26), dpi=300)
plt.subplots_adjust(hspace=0.28, wspace=0.22)

colors_cls = {
    "Overall": "#222222",
    "S_c": "#2ca02c",      # Emerald Green (Cold Start)
    "S_h": "#1f77b4",      # Steel Blue (History Reactivation)
    "S_r": "#d62728",      # Crimson Red (Active Takeover)
}

styles_cls = {
    "Overall": "--",
    "S_c": "-",
    "S_h": "-.",
    "S_r": "-"
}

# Panel 1: MOT17 High-Recall ROC Curves (Noisy-OR)
ax1 = axes[0, 0]
for cls in CLASSES:
    res = data["MOT17"][cls]["Noisy-OR"]["global"]
    tpr = np.array(res["tpr_dense"]) * 100.0
    fpr = np.array(res["fpr_dense"]) * 100.0
    lw = 3.0 if cls in ["S_r", "Overall"] else 2.2
    lbl = f"{cls} (pAUC[0.6,1]={res['norm_pauc_60']:.3f}, FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%)"
    ax1.plot(fpr, tpr, label=lbl, color=colors_cls[cls], linestyle=styles_cls[cls], linewidth=lw)
ax1.set_title("(a) MOT17-half: Class-Wise High-Recall ROC Curves", fontsize=12, fontweight="bold", pad=8)
ax1.set_xlabel("False Positive Rate (%) on 1.65M Negatives", fontsize=10, fontweight="bold")
ax1.set_ylabel("True Positive Rate / Recall (%)", fontsize=10, fontweight="bold")
ax1.set_xlim(-1, 101)
ax1.set_ylim(58, 101.5)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cccccc")

# Panel 2: MOT20 High-Recall ROC Curves (Noisy-OR)
ax2 = axes[0, 1]
for cls in CLASSES:
    res = data["MOT20"][cls]["Noisy-OR"]["global"]
    tpr = np.array(res["tpr_dense"]) * 100.0
    fpr = np.array(res["fpr_dense"]) * 100.0
    lw = 3.0 if cls in ["S_r", "Overall"] else 2.2
    lbl = f"{cls} (pAUC[0.6,1]={res['norm_pauc_60']:.3f}, FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%)"
    ax2.plot(fpr, tpr, label=lbl, color=colors_cls[cls], linestyle=styles_cls[cls], linewidth=lw)
ax2.set_title("(b) MOT20: Class-Wise High-Recall ROC Curves (Dense Crowd)", fontsize=12, fontweight="bold", pad=8)
ax2.set_xlabel("False Positive Rate (%) on 1.65M Negatives", fontsize=10, fontweight="bold")
ax2.set_ylabel("True Positive Rate / Recall (%)", fontsize=10, fontweight="bold")
ax2.set_xlim(-1, 101)
ax2.set_ylim(58, 101.5)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cccccc")

# Panel 3: SportsMOT High-Recall ROC Curves (Noisy-OR)
ax3 = axes[1, 0]
for cls in CLASSES:
    res = data["SportsMOT"][cls]["Noisy-OR"]["global"]
    tpr = np.array(res["tpr_dense"]) * 100.0
    fpr = np.array(res["fpr_dense"]) * 100.0
    lw = 3.0 if cls in ["S_r", "Overall"] else 2.2
    lbl = f"{cls} (pAUC[0.6,1]={res['norm_pauc_60']:.3f}, FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%)"
    ax3.plot(fpr, tpr, label=lbl, color=colors_cls[cls], linestyle=styles_cls[cls], linewidth=lw)
ax3.set_title("(c) SportsMOT: Class-Wise High-Recall ROC Curves (High Agility)", fontsize=12, fontweight="bold", pad=8)
ax3.set_xlabel("False Positive Rate (%) on 1.65M Negatives", fontsize=10, fontweight="bold")
ax3.set_ylabel("True Positive Rate / Recall (%)", fontsize=10, fontweight="bold")
ax3.set_xlim(-1, 101)
ax3.set_ylim(58, 101.5)
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cccccc")

# Panel 4: Overall Unified High-Recall ROC Curves (Noisy-OR)
ax4 = axes[1, 1]
for cls in CLASSES:
    res = data["Overall"][cls]["Noisy-OR"]["global"]
    tpr = np.array(res["tpr_dense"]) * 100.0
    fpr = np.array(res["fpr_dense"]) * 100.0
    lw = 3.0 if cls in ["S_r", "Overall"] else 2.2
    lbl = f"{cls} (pAUC[0.6,1]={res['norm_pauc_60']:.3f}, FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%)"
    ax4.plot(fpr, tpr, label=lbl, color=colors_cls[cls], linestyle=styles_cls[cls], linewidth=lw)
ax4.set_title("(d) Overall Benchmark: Class-Wise High-Recall ROC Curves", fontsize=12, fontweight="bold", pad=8)
ax4.set_xlabel("False Positive Rate (%) on 1.65M Negatives", fontsize=10, fontweight="bold")
ax4.set_ylabel("True Positive Rate / Recall (%)", fontsize=10, fontweight="bold")
ax4.set_xlim(-1, 101)
ax4.set_ylim(58, 101.5)
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cccccc")

# Panel 5: Cross-Dataset Comparison on the S_r Bottleneck Class
ax5 = axes[2, 0]
ds_colors = {"MOT17": "#e41a1c", "MOT20": "#377eb8", "SportsMOT": "#4daf4a", "Overall": "#984ea3"}
ds_markers = {"MOT17": "o", "MOT20": "s", "SportsMOT": "^", "Overall": "D"}
steps_pct = [t * 100.0 for t in TPR_STEPS]

for ds in DATASETS:
    res = data[ds]["S_r"]["Noisy-OR"]["global"]
    fpr_steps = [res["step_metrics"][str(t) if str(t) in res["step_metrics"] else f"{t:.2f}"]["fpr"] * 100.0 for t in TPR_STEPS]
    ax5.plot(steps_pct, fpr_steps, label=f"{ds} (FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%, pAUC={res['norm_pauc_60']:.3f})",
             color=ds_colors[ds], marker=ds_markers[ds], markersize=6, linewidth=2.4)
ax5.set_title("(e) Cross-Dataset FPR Escalation on Bottleneck S_r (Active Takeover)", fontsize=12, fontweight="bold", pad=8)
ax5.set_xlabel("Target Recall / TPR on S_r (%)", fontsize=10, fontweight="bold")
ax5.set_ylabel("False Positive Rate / FPR (%)", fontsize=10, fontweight="bold")
ax5.set_xticks(steps_pct)
ax5.set_xticklabels([f"{t*100:.0f}%" if t < 0.99 else ("99%" if t == 0.99 else "100%") for t in TPR_STEPS], fontsize=8)
ax5.set_xlim(58, 102)
ax5.set_ylim(-2, 102)
ax5.grid(True, linestyle="--", alpha=0.5)
ax5.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cccccc")

# Panel 6: Aggregation Model Benchmark on S_r (Overall Dataset)
ax6 = axes[2, 1]
model_colors = {"Max": "#2b5c8f", "PowerMean": "#d95f02", "Noisy-OR": "#d62728", "OWA": "#1b9e77"}
model_styles = {"Max": "-", "PowerMean": "--", "Noisy-OR": "-", "OWA": ":"}
for m in MODELS:
    res = data["Overall"]["S_r"][m]["global"]
    fpr_steps = [res["step_metrics"][str(t) if str(t) in res["step_metrics"] else f"{t:.2f}"]["fpr"] * 100.0 for t in TPR_STEPS]
    ax6.plot(steps_pct, fpr_steps, label=f"{m} (FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%, pAUC={res['norm_pauc_60']:.3f})",
             color=model_colors[m], linestyle=model_styles[m], marker="o", markersize=5, linewidth=2.2)
ax6.set_title("(f) Aggregation Operator Benchmark on S_r (Overall Dataset)", fontsize=12, fontweight="bold", pad=8)
ax6.set_xlabel("Target Recall / TPR on S_r (%)", fontsize=10, fontweight="bold")
ax6.set_ylabel("False Positive Rate / FPR (%)", fontsize=10, fontweight="bold")
ax6.set_xticks(steps_pct)
ax6.set_xticklabels([f"{t*100:.0f}%" if t < 0.99 else ("99%" if t == 0.99 else "100%") for t in TPR_STEPS], fontsize=8)
ax6.set_xlim(58, 102)
ax6.set_ylim(-2, 102)
ax6.grid(True, linestyle="--", alpha=0.5)
ax6.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cccccc")

# Panel 7: Single Feature Capacity Comparison on Overall Dataset
ax7 = axes[3, 0]
feat_colors = {"r_weak": "#ff7f0e", "r_comp": "#2ca02c", "r_swap": "#1f77b4"}
for fn in ["r_weak", "r_comp", "r_swap"]:
    for cls in ["S_c", "S_r"]:
        res = data["Overall"][cls]["single_feats"][fn]["global"]
        tpr = np.array(res["tpr_dense"]) * 100.0
        fpr = np.array(res["fpr_dense"]) * 100.0
        ls = "-" if cls == "S_r" else "--"
        lbl = f"{fn} on {cls} (pAUC={res['norm_pauc_60']:.3f})"
        ax7.plot(fpr, tpr, label=lbl, color=feat_colors[fn], linestyle=ls, linewidth=2.0)
ax7.set_title("(g) Single Causal Feature Discriminability on S_c vs S_r", fontsize=12, fontweight="bold", pad=8)
ax7.set_xlabel("False Positive Rate (%) on 1.65M Negatives", fontsize=10, fontweight="bold")
ax7.set_ylabel("True Positive Rate / Recall (%)", fontsize=10, fontweight="bold")
ax7.set_xlim(-1, 101)
ax7.set_ylim(58, 101.5)
ax7.grid(True, linestyle="--", alpha=0.5)
ax7.legend(loc="lower right", fontsize=8.0, frameon=True, facecolor="white", edgecolor="#cccccc")

# Panel 8: Risk Score CDF Tail Distribution
ax8 = axes[3, 1]
score_bins = np.linspace(0.0, 1.0, 501)
for ds in ["MOT17", "MOT20", "SportsMOT"]:
    for cls in ["S_c", "S_r"]:
        res = data[ds][cls]["Noisy-OR"]["global"]
        # CDF calculation from percentiles / dense points approximation or score array
        # Let's plot the stepped quantiles
        ax8.plot([res["min_score"], res["p01_score"], res["p05_score"], res["p10_score"], res["p50_score"], res["p90_score"], res["p99_score"], 1.0],
                 [0, 1, 5, 10, 50, 90, 99, 100],
                 label=f"{ds} - {cls} (Min={res['min_score']:.3f})",
                 marker="o", markersize=4, linewidth=1.8)
ax8.set_title("(h) Quantile Profile of Calibrated Risk Scores (Tail Anatomy)", fontsize=12, fontweight="bold", pad=8)
ax8.set_xlabel("Calibrated Risk Score R", fontsize=10, fontweight="bold")
ax8.set_ylabel("Cumulative Event Percentage (%) <= R", fontsize=10, fontweight="bold")
ax8.set_xlim(-0.02, 1.02)
ax8.set_ylim(-2, 102)
ax8.grid(True, linestyle="--", alpha=0.5)
ax8.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor="white", edgecolor="#cccccc")

plt.tight_layout()
out_fig_path = os.path.join(TAXONOMY_DIR, "class_risk_tpr_fpr_comparison.png")
plt.savefig(out_fig_path, dpi=300)
plt.close()
print(f"Saved master 8-panel analytical figure: {out_fig_path}")
