# -*- coding: utf-8 -*-
"""Compute the step-8 metric distribution tables / NA stats / gap
distributions from the CURRENT event tables (research/data), for the HTML
report package and switch_metrics_report.md.  Outputs JSON.

Usage:
  python tools/_report_stats.py > _report_stats.json
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "research", "data")
DATASETS = ("MOT17", "MOT20", "SportsMOT")
METRICS = ("IoU_last", "dx_last", "dy_last", "dist_last", "IoU_prev",
           "IoU_next", "IoU_swap", "dist_swap", "area_ratio", "gap")


def load_metrics(ds):
    rows = []
    with open(os.path.join(DATA, "%s_events_metrics.csv" % ds),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def stat_block(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "N": int(a.size),
        "mean": float(a.mean()), "std": float(a.std()),
        "min": float(a.min()),
        "P25": float(np.percentile(a, 25)), "P50": float(np.percentile(a, 50)),
        "P75": float(np.percentile(a, 75)), "P90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def metric_stats(rows, typ):
    """Per-metric stats + missing counts for one type subset."""
    sub = [r for r in rows if r["type"] == typ]
    n = len(sub)
    stats, missing = {}, {}
    for m in METRICS:
        vals = [float(r[m]) for r in sub if r[m].strip()]
        stats[m] = stat_block(vals) if len(vals) >= 2 else None
        missing[m] = n - len(vals)
    # gap distribution over events with a last frame
    gaps = [int(float(r["gap"])) for r in sub if r["gap"].strip()]
    gap_dist = {"1": 0, "2": 0, "3-5": 0, ">5": 0}
    for g in gaps:
        if g == 1:
            gap_dist["1"] += 1
        elif g == 2:
            gap_dist["2"] += 1
        elif g <= 5:
            gap_dist["3-5"] += 1
        else:
            gap_dist[">5"] += 1
    return {"n": n, "stats": stats, "missing": missing,
            "gap_n": len(gaps), "gap_dist": gap_dist}


def main():
    out = {}
    for ds in DATASETS:
        rows = load_metrics(ds)
        n_sw = sum(1 for r in rows if r["type"] == "switch")
        n_re = sum(1 for r in rows if r["type"] == "reuse")
        out[ds] = {"n_switch": n_sw, "n_reuse": n_re,
                   "switch": metric_stats(rows, "switch"),
                   "reuse": metric_stats(rows, "reuse")}
    # combined
    all_rows = []
    for ds in DATASETS:
        all_rows += load_metrics(ds)
    out["ALL"] = {"n_switch": sum(out[d]["n_switch"] for d in DATASETS),
                  "n_reuse": sum(out[d]["n_reuse"] for d in DATASETS),
                  "switch": metric_stats(all_rows, "switch"),
                  "reuse": metric_stats(all_rows, "reuse")}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
