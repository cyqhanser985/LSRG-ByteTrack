# -*- coding: utf-8 -*-
# Plot reproduction results for MOT17 / MOT20 / SportsMOT V001 evaluations.
# All source is pure ASCII; Chinese labels use unicode escapes.
import os
import sys
import glob
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/charts"
os.makedirs(OUT, exist_ok=True)

# Chinese labels (unicode escapes)
T_OVERALL = "\u4e09\u6570\u636e\u96c6\u603b\u4f53\u6307\u6807\u5bf9\u6bd4"   # 三数据集总体指标对比
T_MOT17 = "MOT17 \u5404\u573a\u666f\u8ddf\u8e2a\u6307\u6807"                 # MOT17 各场景跟踪指标
T_MOT20 = "MOT20 \u5404\u5e8f\u5217\u8ddf\u8e2a\u6307\u6807"                 # MOT20 各序列跟踪指标
T_SPORTS = "SportsMOT \u5404\u5e8f\u5217 MOTA"                               # SportsMOT 各序列 MOTA
T_ERR = "\u8bef\u5dee\u7edf\u8ba1\u5bf9\u6bd4 (FP / FN / IDs)"               # 误差统计对比
L_DATASET = "\u6570\u636e\u96c6"                                            # 数据集
L_SCENE = "\u573a\u666f"                                                    # 场景
L_SEQ = "\u5e8f\u5217"                                                      # 序列
L_VALUE = "\u6307\u6807\u503c (%)"                                          # 指标值(%)
L_OVERALL = "OVERALL"


def load_summary(ds, res_dir):
    gtfiles = sorted(str(p) for p in (Path("datasets") / ds).glob("V*/gt/gt.txt"))
    tsfiles = [f for f in glob.glob(os.path.join(res_dir, "*.txt"))
               if not os.path.basename(f).startswith("eval")]
    gt = OrderedDict((Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)) for f in gtfiles)
    ts = OrderedDict((os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1)) for f in tsfiles)
    names = [k for k in ts if k in gt]
    accs = [mm.utils.compare_to_groundtruth(gt[k], ts[k], "iou", distth=0.5) for k in names]
    mh = mm.metrics.create()
    metrics = ["idf1", "mota", "recall", "precision", "mostly_tracked",
               "mostly_lost", "num_false_positives", "num_misses",
               "num_switches", "num_objects", "num_unique_objects"]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    return names, summary


def scene_name(v, ds):
    mp = os.path.join("datasets", ds, "mapping.txt")
    if os.path.exists(mp):
        for line in open(mp, encoding="utf-8", errors="ignore"):
            p = line.split()
            if len(p) == 2 and p[0] == v:
                return p[1].split("/")[-1]
    return v


DATA = [
    ("MOT17", "YOLOX_outputs/mot17_v001_full/track_results"),
    ("MOT20", "YOLOX_outputs/mot20_v001_full/track_results"),
    ("SportsMOT", "YOLOX_outputs/sportsmot_v001_full/track_results"),
]

summaries = {}
for ds, res in DATA:
    names, summary = load_summary(ds, res)
    summaries[ds] = (names, summary)
    print(ds, "sequences:", len(names), "overall MOTA: %.1f%%  IDF1: %.1f%%" % (
        summary.loc[L_OVERALL, "mota"] * 100, summary.loc[L_OVERALL, "idf1"] * 100))


# ---- Fig 1: overall comparison across 3 datasets ---------------------------
fig, ax = plt.subplots(figsize=(9, 5.2))
metrics = ["mota", "idf1", "recall", "precision"]
labels = ["MOTA", "IDF1", "Rcll", "Prcn"]
colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
x = np.arange(len(DATA))
width = 0.2
for i, m in enumerate(metrics):
    vals = [summaries[d][1].loc[L_OVERALL, m] * 100 for d, _ in DATA]
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=labels[i], color=colors[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, "%.1f" % v, ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([d for d, _ in DATA], fontsize=11)
ax.set_ylabel(L_VALUE)
ax.set_ylim(0, 110)
ax.set_title(T_OVERALL, fontsize=13, fontweight="bold")
ax.legend(loc="lower right", ncol=4, fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_overall_comparison.png"), dpi=150)
plt.close(fig)

# ---- Fig 2: MOT17 per-scene ------------------------------------------------
names, summary = summaries["MOT17"]
scenes = {}
for v in names:
    s = scene_name(v, "MOT17")
    base = s.split("-")[0] + "-" + s.split("-")[1] if s.startswith("MOT17") else s
    scenes.setdefault(base, []).append(v)
scene_list = sorted(scenes.keys())
mota = [summary.loc[scenes[s], "mota"].mean() * 100 for s in scene_list]
idf1 = [summary.loc[scenes[s], "idf1"].mean() * 100 for s in scene_list]
x = np.arange(len(scene_list))
fig, ax = plt.subplots(figsize=(9, 5.2))
b1 = ax.bar(x - 0.19, mota, 0.38, label="MOTA", color="#d62728")
b2 = ax.bar(x + 0.19, idf1, 0.38, label="IDF1", color="#1f77b4")
for b, v in zip(list(b1) + list(b2), mota + idf1):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, "%.1f" % v, ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(scene_list, fontsize=10)
ax.set_ylabel(L_VALUE)
ax.set_ylim(0, 105)
ax.set_title(T_MOT17, fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_mot17_scenes.png"), dpi=150)
plt.close(fig)

# ---- Fig 3: MOT20 per-sequence --------------------------------------------
names, summary = summaries["MOT20"]
mota = [summary.loc[v, "mota"] * 100 for v in names]
idf1 = [summary.loc[v, "idf1"] * 100 for v in names]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - 0.19, mota, 0.38, label="MOTA", color="#d62728")
b2 = ax.bar(x + 0.19, idf1, 0.38, label="IDF1", color="#1f77b4")
for b, v in zip(list(b1) + list(b2), mota + idf1):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, "%.1f" % v, ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([v + " (" + scene_name(v, "MOT20") + ")" for v in names], fontsize=9)
ax.set_ylabel(L_VALUE)
ax.set_ylim(0, 105)
ax.set_title(T_MOT20, fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_mot20_sequences.png"), dpi=150)
plt.close(fig)

# ---- Fig 4: SportsMOT per-sequence MOTA ------------------------------------
names, summary = summaries["SportsMOT"]
mota = [summary.loc[v, "mota"] * 100 for v in names]
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(range(1, len(names) + 1), mota, marker="o", ms=3, lw=1, color="#1f77b4")
ax.axhline(summary.loc[L_OVERALL, "mota"] * 100, color="#d62728", ls="--", lw=1.2,
           label="OVERALL %.1f%%" % (summary.loc[L_OVERALL, "mota"] * 100))
ax.set_xlabel(L_SEQ)
ax.set_ylabel("MOTA (%)")
ax.set_ylim(85, 101)
ax.set_xticks(range(1, len(names) + 1, 10))
ax.set_title(T_SPORTS, fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_sportsmot_sequences.png"), dpi=150)
plt.close(fig)

# ---- Fig 5: error summary (FP / FN / IDs) ---------------------------------
fig, ax = plt.subplots(figsize=(9, 5.2))
err_metrics = ["num_false_positives", "num_misses", "num_switches"]
err_labels = ["FP", "FN", "IDs"]
x = np.arange(len(DATA))
width = 0.25
for i, m in enumerate(err_metrics):
    vals = [summaries[d][1].loc[L_OVERALL, m] for d, _ in DATA]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=err_labels[i],
                  color=["#1f77b4", "#d62728", "#ff7f0e"][i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.03, "%d" % v, ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([d for d, _ in DATA], fontsize=11)
ax.set_yscale("log")
ax.set_ylabel("\u6570\u91cf (log)")  # 数量(log)
ax.set_title(T_ERR, fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_error_summary.png"), dpi=150)
plt.close(fig)

print("charts saved to", os.path.abspath(OUT))
