# -*- coding: utf-8 -*-
import glob, json, os, sys
from collections import OrderedDict
from pathlib import Path
import motmetrics as mm
ROOT = Path(r"E:\科研\ByteTrack")
TRACK_DIRS = {"MOT17": "mot17_v001_full", "MOT20": "mot20_v001_full", "SportsMOT": "sportsmot_v001_full"}
METRICS = ["idf1", "mota", "recall", "precision", "mostly_tracked", "mostly_lost",
           "num_false_positives", "num_misses", "num_switches", "num_objects",
           "num_unique_objects", "num_fragmentations", "motp"]
for ds in (sys.argv[1:] or ["MOT17", "MOT20", "SportsMOT"]):
    gtfiles = sorted(str(p) for p in (ROOT / "datasets" / ds).glob("V*/gt/gt.txt"))
    tsdir = ROOT / "YOLOX_outputs" / TRACK_DIRS[ds] / "track_results"
    tsfiles = [f for f in glob.glob(os.path.join(tsdir, "*.txt")) if not os.path.basename(f).startswith("eval")]
    gt = OrderedDict((Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)) for f in gtfiles)
    ts = OrderedDict((os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1)) for f in tsfiles)
    names = [k for k in ts if k in gt]
    accs = [mm.utils.compare_to_groundtruth(gt[k], ts[k], "iou", distth=0.5) for k in names]
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=METRICS, generate_overall=True)
    rows = []
    for name in names + ["OVERALL"]:
        r = summary.loc[name]
        rows.append({"seq": name, "mota": float(r["mota"]), "idf1": float(r["idf1"]),
                     "recall": float(r["recall"]), "precision": float(r["precision"]),
                     "gt": int(r["num_unique_objects"]), "mt": int(r["mostly_tracked"]),
                     "ml": int(r["mostly_lost"]), "fp": int(r["num_false_positives"]),
                     "fn": int(r["num_misses"]), "ids": int(r["num_switches"]),
                     "fm": int(r["num_fragmentations"]), "motp": float(r["motp"])})
    print(json.dumps({"ds": ds, "rows": rows}, ensure_ascii=False))
