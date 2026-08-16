# -*- coding: utf-8 -*-
"""Rebuild the frozen IDS event sets from the CURRENT gt.txt files and the
canonical track_results, via motmetrics temporal matching.

This is the single source of truth for {MOT17,MOT20,SportsMOT}_events.csv /
_events_metrics.csv / _events_summary.csv after the 2026-08-15 GT cleaning
(conf==1 only).  The V1/V2 generator scripts were deleted; this script
re-implements the exact documented semantics (see
reports/ByteTrack_ID分析报告/docs/switch_metrics_report.md, step 8 metric
definitions, and COE.md section 3):

  * per sequence, feed EVERY frame of the union of GT/track frames to a
    motmetrics MOTAccumulator (empty frames included, max_iou=0.5), like the
    original "empty frames counted" protocol;
  * switch event = a motmetrics SWITCH row (F, gt g, tracker B): the GT's
    assignment changed vs its last MATCH-row assignment.  old_hid A = the
    tracker of g's last MATCH row before F ("unknown" if g never had one);
  * reuse event = a motmetrics MATCH row (F, gt g, tracker B) where B's last
    MATCH row was on a different gt g_old (MATCH rows only -- SWITCH rows are
    deliberately blind to, so |S n Reuse| = 0);
  * metrics (from the track txt only, per the step-8 report):
      F_l    = last frame < F where the event tracker tid outputs a box
      gap    = F - F_l
      IoU_last / dx_last / dy_last / dist_last = tid box continuity F_l -> F
      IoU_prev / IoU_next = IoU(tid@F-1, tid@F) / IoU(tid@F, tid@F+1)
      IoU_swap / dist_swap (switch only) = old tracker A box@F vs tid box@F
      area_ratio = area(tid@F) / area(tid@F_l)
      na_flag    = no_last_seen | no_prev | no_next | no_old, joined with '|'
                   (no_last_seen: tid never output before F; no_old: A has no
                    box at F in the track txt).

Outputs (overwrite, canonical location research/data/):
  {DS}_events.csv          seq,frame,type,gt_id_old,gt_id_new,track_id,note
  {DS}_events_metrics.csv  19-column event + physics metrics table
  {DS}_events_summary.csv  per-sequence counts vs motmetrics SWITCH/TRANSFER

Usage (from the repo root, with the bytetrack conda interpreter):
  python tools/build_ids_events.py                 # MOT17 + MOT20 + SportsMOT
  python tools/build_ids_events.py --datasets mot17
"""
import argparse
import csv
import os
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd
import motmetrics as mm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "research", "data")
TRACK_DIRS = {"MOT17": "mot17_v001_full", "MOT20": "mot20_v001_full",
              "SportsMOT": "sportsmot_v001_full"}
MAX_IOU = 0.5          # motmetrics iou_matrix candidate cutoff (official)
EVENT_TYPES = ("MATCH", "SWITCH", "TRANSFER")   # resolved assignment rows


def load_frames(txt_path):
    """{frame: {tid: (x, y, w, h)}} for one sequence file."""
    frames = defaultdict(dict)
    with open(txt_path, encoding="utf-8-sig") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            try:
                fr = int(float(p[0]))
                tid = int(float(p[1]))
            except ValueError:
                continue
            frames[fr][tid] = (float(p[2]), float(p[3]),
                               float(p[4]), float(p[5]))
    return frames


def run_accumulator(gt_frames, tr_frames):
    """motmetrics MOTAccumulator over the union of frames (empty frames
    counted); returns the resolved events dataframe."""
    frames = sorted(set(gt_frames) | set(tr_frames))
    acc = mm.MOTAccumulator(auto_id=False)
    for fr in frames:
        gb = gt_frames.get(fr) or {}
        tb = tr_frames.get(fr) or {}
        gids, tids = sorted(gb), sorted(tb)
        if gids and tids:
            garr = np.array([[gb[g][0], gb[g][1], gb[g][2], gb[g][3]]
                             for g in gids])
            tarr = np.array([[tb[t][0], tb[t][1], tb[t][2], tb[t][3]]
                             for t in tids])
            dist = mm.distances.iou_matrix(garr, tarr, max_iou=MAX_IOU)
        else:
            dist = np.empty((len(gids), len(tids)))
        acc.update(gids, tids, dist, frameid=fr)
    return acc.events


def extract_events(events, seq):
    """Iterate the resolved event stream once, tracking per-GT and per-tracker
    last MATCH rows, and return (switches, reuses) as plain dicts."""
    gt_last_match = {}    # gid -> (frame, tracker)
    tr_last_match = {}    # tid -> (frame, gid)
    switches, reuses = [], []
    n_sw_metric = 0
    n_transfer = 0
    for (fid, _eid), row in events.iterrows():
        typ = row["Type"]
        oid, hid = row["OId"], row["HId"]
        oid_na, hid_na = pd.isna(oid), pd.isna(hid)
        if typ == "SWITCH" and not oid_na and not hid_na:
            oid, hid = int(oid), int(hid)
            n_sw_metric += 1
            old = gt_last_match.get(oid)
            switches.append({
                "seq": seq, "frame": int(fid), "type": "switch",
                "gt_id_old": oid, "gt_id_new": oid, "track_id": hid,
                "old_hid": old[1] if old else None,
                "note": "tracker %s -> %d on gt %d"
                        % (str(old[1]) if old else "unknown", hid, oid),
            })
        elif typ == "TRANSFER":
            n_transfer += 1
        elif typ == "MATCH" and not oid_na and not hid_na:
            oid, hid = int(oid), int(hid)
            prev = tr_last_match.get(hid)
            if prev is not None and prev[1] != oid:
                reuses.append({
                    "seq": seq, "frame": int(fid), "type": "reuse",
                    "gt_id_old": prev[1], "gt_id_new": oid,
                    "track_id": hid, "old_hid": 0,
                    "note": "tracker %d matched gt %d (last frame %d) "
                            "now gt %d" % (hid, prev[1], prev[0], oid),
                })
            gt_last_match[oid] = (int(fid), hid)
            tr_last_match[hid] = (int(fid), oid)
    return switches, reuses, n_sw_metric, n_transfer


def box_iou(b1, b2):
    ax1, ay1, ax2, ay2 = b1[0], b1[1], b1[0] + b1[2], b1[1] + b1[3]
    bx1, by1, bx2, by2 = b2[0], b2[1], b2[0] + b2[2], b2[1] + b2[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def center(b):
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def compute_metrics(ev, tr_frames):
    """Per-event physics metrics from the track txt (step-8 definitions)."""
    F = ev["frame"]
    tid = ev["track_id"]
    box_f = tr_frames[F][tid]
    flags = []

    # last appearance before F
    frames_of = sorted(f for f in tr_frames if tid in tr_frames[f])
    idx = bisect_left(frames_of, F)
    last_seen = frames_of[idx - 1] if idx > 0 else None
    if last_seen is None:
        flags.append("no_last_seen")
    else:
        b_l = tr_frames[last_seen][tid]
        ev["last_frame"] = last_seen
        ev["gap"] = F - last_seen
        ev["IoU_last"] = box_iou(b_l, box_f)
        c0, c1 = center(b_l), center(box_f)
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        ev["dx_last"] = dx
        ev["dy_last"] = dy
        ev["dist_last"] = float(np.hypot(dx, dy))
        ev["area_ratio"] = (box_f[2] * box_f[3]) / (b_l[2] * b_l[3])

    # adjacent-frame continuity of the event tracker itself
    if F - 1 in tr_frames and tid in tr_frames[F - 1]:
        ev["IoU_prev"] = box_iou(tr_frames[F - 1][tid], box_f)
    else:
        flags.append("no_prev")
    if F + 1 in tr_frames and tid in tr_frames[F + 1]:
        ev["IoU_next"] = box_iou(box_f, tr_frames[F + 1][tid])
    else:
        flags.append("no_next")

    # switch only: old tracker A spatial relation at F
    if ev["type"] == "switch" and ev["old_hid"] is not None:
        a = ev["old_hid"]
        a_box = tr_frames.get(F, {}).get(a)
        if a_box is not None:
            ev["IoU_swap"] = box_iou(a_box, box_f)
            ca, cb = center(a_box), center(box_f)
            ev["dist_swap"] = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
        else:
            flags.append("no_old")
    ev["na_flag"] = "|".join(flags)


def fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return "%.8f" % v
    return str(v)


def write_dataset(ds):
    tr_dir = os.path.join(ROOT, "YOLOX_outputs", TRACK_DIRS[ds],
                          "track_results")
    ds_root = os.path.join(ROOT, "datasets", ds)
    seqs = sorted(d for d in os.listdir(ds_root)
                  if os.path.isdir(os.path.join(ds_root, d, "gt")))
    ev_rows, met_rows, sum_rows = [], [], []
    n_switch_tot = n_reuse_tot = n_sw_tot = n_tr_tot = 0
    for seq in seqs:
        gt_path = os.path.join(ds_root, seq, "gt", "gt.txt")
        tr_path = os.path.join(tr_dir, "%s.txt" % seq)
        if not os.path.exists(gt_path) or not os.path.exists(tr_path):
            print("  skip %s: missing gt/track file" % seq)
            continue
        gt_frames = load_frames(gt_path)
        tr_frames = load_frames(tr_path)
        events = run_accumulator(gt_frames, tr_frames)
        switches, reuses, n_sw, n_tr = extract_events(events, seq)

        # stable order: (frame, switch<reuse, stream order)
        all_ev = [(e, i, 0) for i, e in enumerate(switches)]
        all_ev += [(e, i, 1) for i, e in enumerate(reuses)]
        all_ev.sort(key=lambda t: (t[0]["frame"], t[2], t[1]))

        sw = re = 0
        for ev, _i, _r in all_ev:
            ev_rows.append("%s,%d,%s,%d,%d,%d,%s" % (
                ev["seq"], ev["frame"], ev["type"],
                ev["gt_id_old"], ev["gt_id_new"], ev["track_id"],
                ev["note"]))
            compute_metrics(ev, tr_frames)
            met_rows.append("%s,%d,%s,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
                ev["seq"], ev["frame"], ev["type"],
                ev["gt_id_old"], ev["gt_id_new"], ev["track_id"],
                fmt(ev["old_hid"]), fmt(ev.get("last_frame")),
                fmt(ev.get("gap")), fmt(ev.get("IoU_last")),
                fmt(ev.get("dx_last")), fmt(ev.get("dy_last")),
                fmt(ev.get("dist_last")), fmt(ev.get("IoU_prev")),
                fmt(ev.get("IoU_next")), fmt(ev.get("IoU_swap")),
                fmt(ev.get("dist_swap")), fmt(ev.get("area_ratio")),
                ev.get("na_flag", "")))
            if ev["type"] == "switch":
                sw += 1
            else:
                re += 1
        status = "OK" if sw == n_sw else "FAIL"
        sum_rows.append("%s,%d,%d,%d,%d,%s" % (seq, sw, re, n_sw, n_tr, status))
        n_switch_tot += sw
        n_reuse_tot += re
        n_sw_tot += n_sw
        n_tr_tot += n_tr
        print("  %s: switch=%d reuse=%d (motmetrics SWITCH=%d TRANSFER=%d) %s"
              % (seq, sw, re, n_sw, n_tr, status))

    for name, rows, header in (
            ("%s_events.csv" % ds, ev_rows,
             "seq,frame,type,gt_id_old,gt_id_new,track_id,note"),
            ("%s_events_metrics.csv" % ds, met_rows,
             "seq,frame,type,gt_id_old,gt_id_new,track_id,old_hid,last_frame,"
             "gap,IoU_last,dx_last,dy_last,dist_last,IoU_prev,IoU_next,"
             "IoU_swap,dist_swap,area_ratio,na_flag"),
            ("%s_events_summary.csv" % ds, sum_rows,
             "seq,switch_count,reuse_count,num_switches_metric,"
             "transfer_rows,status")):
        path = os.path.join(DATA, name)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(header + "\n")
            for r in rows:
                f.write(r + "\n")
        print("wrote %s (%d rows)" % (path, len(rows)))
    print("== %s total: switch=%d reuse=%d (SWITCH rows=%d TRANSFER rows=%d)"
          % (ds, n_switch_tot, n_reuse_tot, n_sw_tot, n_tr_tot))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="mot17,mot20,sportsmot")
    args = ap.parse_args()
    ds_map = {"mot17": "MOT17", "mot20": "MOT20", "sportsmot": "SportsMOT"}
    for d in args.datasets.split(","):
        ds = ds_map[d.strip().lower()]
        print("== %s" % ds)
        write_dataset(ds)


if __name__ == "__main__":
    main()
