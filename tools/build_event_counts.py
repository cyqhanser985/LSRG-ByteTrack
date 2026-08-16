# -*- coding: utf-8 -*-
"""Rebuild research/taxonomy/event_counts_by_sequence.csv from the current
frozen event tables (research/data/*_events_metrics.csv).

This file is the SANITY anchor of research/code/analysis.py (per-class counts
per sequence).  It must be regenerated whenever the event tables change:
  python tools/build_event_counts.py
The S_c / S_r / S_h classification is the single source of truth from
analysis.py (classify() on the na_flag column):
  no_last_seen            -> S_c (cold start, never output before F)
  no_prev absent          -> S_r (active takeover, output at F-1)
  otherwise               -> S_h (history reactivation)
"""
import csv
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "research", "data")
OUT = os.path.join(ROOT, "research", "taxonomy")
DATASETS = ("MOT17", "MOT20", "SportsMOT")


def flags(na_flag):
    s = (na_flag or "").strip()
    return set(x for x in s.split("|") if x) if s else set()


def classify(fl):
    if "no_last_seen" in fl:
        return "S_c"
    if "no_prev" not in fl:
        return "S_r"
    return "S_h"


def main():
    rows = []
    for ds in DATASETS:
        path = os.path.join(DATA, "%s_events_metrics.csv" % ds)
        per_seq = defaultdict(Counter)
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                seq = r["seq"]
                if r["type"].strip() == "switch":
                    per_seq[seq][classify(flags(r["na_flag"]))] += 1
                    per_seq[seq]["n_switch"] += 1
                else:
                    per_seq[seq]["n_reuse"] += 1
        for seq in sorted(per_seq):
            c = per_seq[seq]
            rows.append((ds, seq, c["S_c"], c["S_r"], c["S_h"],
                         c["n_switch"], c["n_reuse"]))
        tot = sum(c["n_switch"] for c in per_seq.values())
        print("%s: %d sequences, %d switch events" % (ds, len(per_seq), tot))
    out_path = os.path.join(OUT, "event_counts_by_sequence.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seq", "n_S_c", "n_S_r", "n_S_h",
                    "n_switch", "n_reuse"])
        w.writerows(rows)
    print("wrote %s (%d rows)" % (out_path, len(rows)))


if __name__ == "__main__":
    main()
