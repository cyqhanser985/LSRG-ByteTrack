# -*- coding: utf-8 -*-
"""Recompute per-dataset MOTA / IDF1 / IDs (num_switches) on the CURRENT
(cleaned) gt.txt vs the frozen v001_full track_results, mirroring
tools/plot_results.py.  Prints a JSON line per dataset.
"""
import glob
import json
import os
from collections import OrderedDict
from pathlib import Path

import motmetrics as mm

ROOT = Path(__file__).resolve().parents[1]
TRACK_DIRS = {"MOT17": "mot17_v001_full", "MOT20": "mot20_v001_full",
              "SportsMOT": "sportsmot_v001_full"}
METRICS = ["idf1", "mota", "recall", "precision", "mostly_tracked",
           "mostly_lost", "num_false_positives", "num_misses",
           "num_switches", "num_objects", "num_unique_objects"]


def eval_ds(ds):
    gtfiles = sorted(str(p) for p in (ROOT / "datasets" / ds).glob("V*/gt/gt.txt"))
    tsdir = ROOT / "YOLOX_outputs" / TRACK_DIRS[ds] / "track_results"
    tsfiles = [f for f in glob.glob(os.path.join(tsdir, "*.txt"))
               if not os.path.basename(f).startswith("eval")]
    gt = OrderedDict((Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D",
                     min_confidence=1)) for f in gtfiles)
    ts = OrderedDict((os.path.splitext(Path(f).parts[-1])[0],
                      mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1))
                     for f in tsfiles)
    names = [k for k in ts if k in gt]
    accs = [mm.utils.compare_to_groundtruth(gt[k], ts[k], "iou", distth=0.5)
            for k in names]
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=METRICS,
                              generate_overall=True)
    row = summary.loc["OVERALL"]
    print(json.dumps({"ds": ds, "n_seq": len(names),
                      "idf1": float(row["idf1"]), "mota": float(row["mota"]),
                      "recall": float(row["recall"]),
                      "precision": float(row["precision"]),
                      "mt": int(row["mostly_tracked"]),
                      "ml": int(row["mostly_lost"]),
                      "fp": int(row["num_false_positives"]),
                      "fn": int(row["num_misses"]),
                      "ids": int(row["num_switches"]),
                      "objects": int(row["num_objects"]),
                      "unique_objects": int(row["num_unique_objects"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    import sys
    for d in (sys.argv[1:] or ["MOT17", "MOT20", "SportsMOT"]):
        eval_ds(d)
