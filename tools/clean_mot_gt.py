# -*- coding: utf-8 -*-
"""Rebuild MOT17 / MOT20 gt.txt files from the per-dataset eval.json,
keeping only conf==1 rows (active pedestrians).

Background: the previous gt.txt conversion flattened the official conf
column to 1 and erased the class column, so ignore regions (conf=0) and
static/distractor targets were loaded as active pedestrians by every
consumer (MOT evaluation, the bad-case diagnostic chain).  The eval.json
files still carry the conf column; this script rewrites each
datasets/{DS}/V*/gt/gt.txt from it, keeping exactly the rows with
conf == 1.  The 10-column layout (frame,tid,x,y,w,h,1,-1,-1,-1) and the
frame ranges are preserved, so no consumer needs changes.

Verified impact:
  MOT17   293,259 -> 162,126 rows (131,133 ignore-region rows removed)
  MOT20  1,336,920 -> 1,134,614 rows (202,306 ignore-region rows removed)
  SportsMOT is untouched (its gt.txt is already clean).

Usage (from the repo root):
    python tools/clean_mot_gt.py --datasets mot17,mot20
    python tools/clean_mot_gt.py            # both, same as above
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS_MAP = {"mot17": "MOT17", "mot20": "MOT20"}


def clean_dataset(ds):
    data_dir = ROOT / "datasets" / ds
    eval_json = data_dir / "annotations" / "eval.json"
    assert eval_json.exists(), "eval.json not found: %s" % eval_json
    data = json.load(open(eval_json, encoding="utf-8"))
    img_of = {i["id"]: i for i in data["images"]}

    # annotations grouped by sequence, conf==1 only
    anns_by_seq = defaultdict(list)
    for a in data["annotations"]:
        if a.get("conf", 0) != 1:
            continue
        img = img_of[a["image_id"]]
        seq = img["file_name"].split("/")[0]
        anns_by_seq[seq].append((img["frame_id"], a["track_id"], tuple(a["bbox"])))

    seqs = sorted(anns_by_seq)
    print("== %s: %d sequences with conf==1 annotations" % (ds, len(seqs)))
    tot_before = tot_after = 0
    for seq in seqs:
        gt_path = data_dir / seq / "gt" / "gt.txt"
        before = 0
        if gt_path.exists():
            before = sum(1 for _ in open(gt_path, encoding="utf-8-sig"))
        anns = sorted(set(anns_by_seq[seq]))
        with open(gt_path, "w", encoding="utf-8") as f:
            for frame, tid, (x, y, w, h) in anns:
                f.write("%d,%d,%.1f,%.1f,%.1f,%.1f,1,-1,-1,-1\n"
                        % (frame, tid, x, y, w, h))
        after = len(anns)
        tot_before += before
        tot_after += after
        print("  %s: %8d -> %8d rows" % (seq, before, after))
    print("  TOTAL: %d -> %d rows (ignore regions removed: %d)"
          % (tot_before, tot_after, tot_before - tot_after))


def main():
    ap = argparse.ArgumentParser(description="clean MOT17/MOT20 gt.txt")
    ap.add_argument("--datasets", default="mot17,mot20",
                    help="comma list of mot17|mot20")
    args = ap.parse_args()
    for d in args.datasets.split(","):
        ds = DS_MAP[d.strip().lower()]
        clean_dataset(ds)


if __name__ == "__main__":
    main()
