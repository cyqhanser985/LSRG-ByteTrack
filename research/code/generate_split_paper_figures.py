# generate_split_paper_figures.py
# ==============================================================================
# Generates Modular, Standalone Publication-Grade Figures (Decoupled & Split)
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

json_path = os.path.join(TAXONOMY_DIR, "class_dataset_breakdown_full.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

TPR_STEPS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]
DATASETS = ["MOT17", "MOT20", "SportsMOT", "Overall"]
CLASSES = ["Overall", "S_c", "S_r", "S_h"]
MODELS = ["Max", "PowerMean", "Noisy-OR", "OWA"]

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

# --------------------------------------------------------------------------
# 1. Standalone Single-Dataset ROC Figures
# --------------------------------------------------------------------------
def plot_dataset_roc(ds_name, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    for cls in CLASSES:
        res = data[ds_name][cls]["Noisy-OR"]["global"]
        tpr = np.array(res["tpr_dense"]) * 100.0
        fpr = np.array(res["fpr_dense"]) * 100.0
        lw = 3.0 if cls in ["S_r", "Overall"] else 2.2
        lbl = f"{cls} (pAUC[0.6,1]={res['norm_pauc_60']:.3f}, FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%)"
        ax.plot(fpr, tpr, label=lbl, color=colors_cls[cls], linestyle=styles_cls[cls], linewidth=lw)
    
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (%) on 1.65M Negatives", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate / Recall (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(-1, 101)
    ax.set_ylim(58, 101.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    out_path = os.path.join(TAXONOMY_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

plot_dataset_roc("MOT17", "MOT17-half: Class-Wise High-Recall ROC Curves", "fig_roc_mot17.png")
plot_dataset_roc("MOT20", "MOT20: Class-Wise High-Recall ROC Curves (Dense Crowd)", "fig_roc_mot20.png")
plot_dataset_roc("SportsMOT", "SportsMOT: Class-Wise High-Recall ROC Curves (High Agility)", "fig_roc_sportsmot.png")
plot_dataset_roc("Overall", "Overall Unified Benchmark: Class-Wise High-Recall ROC Curves", "fig_roc_overall.png")

# --------------------------------------------------------------------------
# 2. Combined 4-Panel Dataset ROC Grid (Clean 2x2)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)
plt.subplots_adjust(hspace=0.25, wspace=0.20)
ds_grid = [
    ("MOT17", "(a) MOT17-half (Urban Pedestrians, Camera Motion)", axes[0, 0]),
    ("MOT20", "(b) MOT20 (Extremely Dense Crowd)", axes[0, 1]),
    ("SportsMOT", "(c) SportsMOT (High-Speed & Non-linear Agility)", axes[1, 0]),
    ("Overall", "(d) Overall Unified Benchmark (All 114 Sequences)", axes[1, 1])
]

for ds_name, title, ax in ds_grid:
    for cls in CLASSES:
        res = data[ds_name][cls]["Noisy-OR"]["global"]
        tpr = np.array(res["tpr_dense"]) * 100.0
        fpr = np.array(res["fpr_dense"]) * 100.0
        lw = 2.8 if cls in ["S_r", "Overall"] else 2.0
        lbl = f"{cls} (pAUC={res['norm_pauc_60']:.3f}, FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%)"
        ax.plot(fpr, tpr, label=lbl, color=colors_cls[cls], linestyle=styles_cls[cls], linewidth=lw)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("False Positive Rate (%) on 1.65M Negatives", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate / Recall (%)", fontsize=10, fontweight="bold")
    ax.set_xlim(-1, 101)
    ax.set_ylim(58, 101.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc")

plt.tight_layout()
out_grid_path = os.path.join(TAXONOMY_DIR, "fig_dataset_roc_2x2_grid.png")
plt.savefig(out_grid_path, dpi=300)
plt.close()
print(f"Saved: {out_grid_path}")

# --------------------------------------------------------------------------
# 3. Cross-Dataset S_r Bottleneck Escalation (Standalone)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
ds_colors = {"MOT17": "#e41a1c", "MOT20": "#377eb8", "SportsMOT": "#4daf4a", "Overall": "#984ea3"}
ds_markers = {"MOT17": "o", "MOT20": "s", "SportsMOT": "^", "Overall": "D"}
steps_pct = [t * 100.0 for t in TPR_STEPS]

for ds in DATASETS:
    res = data[ds]["S_r"]["Noisy-OR"]["global"]
    fpr_steps = [res["step_metrics"][str(t) if str(t) in res["step_metrics"] else f"{t:.2f}"]["fpr"] * 100.0 for t in TPR_STEPS]
    ax.plot(steps_pct, fpr_steps, label=f"{ds} (FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%, pAUC={res['norm_pauc_60']:.3f})",
             color=ds_colors[ds], marker=ds_markers[ds], markersize=6.5, linewidth=2.5)

ax.set_title("Cross-Dataset FPR Escalation on Bottleneck S_r (Active Takeover)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Target Recall / TPR on S_r (%)", fontsize=11, fontweight="bold")
ax.set_ylabel("False Positive Rate / FPR (%) (Lower is Better)", fontsize=11, fontweight="bold")
ax.set_xticks(steps_pct)
ax.set_xticklabels([f"{t*100:.0f}%" if t < 0.99 else ("99%" if t == 0.99 else "100%") for t in TPR_STEPS], fontsize=9)
ax.set_xlim(58, 102)
ax.set_ylim(-2, 102)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")
plt.tight_layout()
out_sr_path = os.path.join(TAXONOMY_DIR, "fig_sr_cross_dataset_escalation.png")
plt.savefig(out_sr_path, dpi=300)
plt.close()
print(f"Saved: {out_sr_path}")

# --------------------------------------------------------------------------
# 4. Aggregation Operator Benchmark on S_r (Standalone)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
model_colors = {"Max": "#2b5c8f", "PowerMean": "#d95f02", "Noisy-OR": "#d62728", "OWA": "#1b9e77"}
model_styles = {"Max": "-", "PowerMean": "--", "Noisy-OR": "-", "OWA": ":"}
for m in MODELS:
    res = data["Overall"]["S_r"][m]["global"]
    fpr_steps = [res["step_metrics"][str(t) if str(t) in res["step_metrics"] else f"{t:.2f}"]["fpr"] * 100.0 for t in TPR_STEPS]
    ax.plot(steps_pct, fpr_steps, label=f"{m} (FPR@95%={res['step_metrics']['0.95']['fpr']*100:.1f}%, pAUC={res['norm_pauc_60']:.3f})",
             color=model_colors[m], linestyle=model_styles[m], marker="o", markersize=6, linewidth=2.4)

ax.set_title("Aggregation Operator Benchmark on S_r (Overall Dataset)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Target Recall / TPR on S_r (%)", fontsize=11, fontweight="bold")
ax.set_ylabel("False Positive Rate / FPR (%) (Lower is Better)", fontsize=11, fontweight="bold")
ax.set_xticks(steps_pct)
ax.set_xticklabels([f"{t*100:.0f}%" if t < 0.99 else ("99%" if t == 0.99 else "100%") for t in TPR_STEPS], fontsize=9)
ax.set_xlim(58, 102)
ax.set_ylim(-2, 102)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")
plt.tight_layout()
out_model_path = os.path.join(TAXONOMY_DIR, "fig_sr_model_benchmark.png")
plt.savefig(out_model_path, dpi=300)
plt.close()
print(f"Saved: {out_model_path}")

# --------------------------------------------------------------------------
# 5. Single Feature Discriminability: S_c vs S_r (Standalone)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
feat_colors = {"r_weak": "#ff7f0e", "r_comp": "#2ca02c", "r_swap": "#1f77b4"}
for fn in ["r_weak", "r_comp", "r_swap"]:
    for cls in ["S_c", "S_r"]:
        res = data["Overall"][cls]["single_feats"][fn]["global"]
        tpr = np.array(res["tpr_dense"]) * 100.0
        fpr = np.array(res["fpr_dense"]) * 100.0
        ls = "-" if cls == "S_r" else "--"
        lbl = f"{fn} on {cls} (pAUC={res['norm_pauc_60']:.3f})"
        ax.plot(fpr, tpr, label=lbl, color=feat_colors[fn], linestyle=ls, linewidth=2.2)

ax.set_title("Single Causal Feature Discriminability on S_c vs S_r", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate (%) on 1.65M Negatives", fontsize=11, fontweight="bold")
ax.set_ylabel("True Positive Rate / Recall (%)", fontsize=11, fontweight="bold")
ax.set_xlim(-1, 101)
ax.set_ylim(58, 101.5)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#cccccc")
plt.tight_layout()
out_feat_path = os.path.join(TAXONOMY_DIR, "fig_single_feature_discriminability.png")
plt.savefig(out_feat_path, dpi=300)
plt.close()
print(f"Saved: {out_feat_path}")

# --------------------------------------------------------------------------
# 6. Quantile Profile & Tail Anatomy (Standalone)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
for ds in ["MOT17", "MOT20", "SportsMOT"]:
    for cls in ["S_c", "S_r"]:
        res = data[ds][cls]["Noisy-OR"]["global"]
        ls = "-" if cls == "S_r" else "--"
        ax.plot([res["min_score"], res["p01_score"], res["p05_score"], res["p10_score"], res["p50_score"], res["p90_score"], res["p99_score"], 1.0],
                 [0, 1, 5, 10, 50, 90, 99, 100],
                 label=f"{ds} - {cls} (Min={res['min_score']:.3f})",
                 marker="o", markersize=5, linewidth=2.0, linestyle=ls)

ax.set_title("Quantile Profile of Calibrated Risk Scores (Tail Anatomy)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Calibrated Risk Score R", fontsize=11, fontweight="bold")
ax.set_ylabel("Cumulative Event Percentage (%) <= R", fontsize=11, fontweight="bold")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-2, 102)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#cccccc")
plt.tight_layout()
out_quant_path = os.path.join(TAXONOMY_DIR, "fig_risk_score_tail_quantiles.png")
plt.savefig(out_quant_path, dpi=300)
plt.close()
print(f"Saved: {out_quant_path}")
