"""DEN online/offline alignment analysis (alert-mode gate logs vs frozen events).

Compares the online DEN gate triggers (logged by BYTETracker alert mode) with
the offline V8 validation expectations:

  1. Trigger rate per event class (S_r / S_c / S_h) at the anchor point and
     across the full eps0 x gamma grid (post-hoc, from logged features).
  2. False-positive proxy on normal frames vs the offline FPR<=1% budget.
  3. Feature alignment: online top1/top2/margin/N vs offline per-event
     features from taxonomy/gate_feasibility_events.csv.
  4. Snapshot consistency: online n_snapshot vs frozen track_results rows.

Usage:
  python research/code/den_online_eval.py \\
      --ds mot17 --logdir YOLOX_outputs/mot17_den_alert_full/den_gate_log \\
      --offline "research/taxonomy/gate_feasibility_events.csv" \\
      --frozen-track YOLOX_outputs/mot17_v001_full/track_results --tag full

Outputs: taxonomy/den_online_{ds}_{tag}.csv + console summary.
Self-contained (read-only CSVs); does not import the frozen analysis.py.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

TOP1_TH = 0.2
TOP2_TH = 0.2
EPS0_GRID = [0.15, 0.20]
GAMMA_GRID = [1.0, 1.25, 1.5, 2.0]
ANCHOR = (0.20, 1.25)


def load_offline_events(path, ds):
    """gate_feasibility_events.csv -> list of event dicts (filtered by ds)."""
    events = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("dataset", "").strip().lower() != ds.lower():
                continue
            events.append(row)
    return events


def load_logs(logdir):
    """Load per-video gate logs. Returns (rows_by_video, frames_by_video,
    summaries). rows: list of dicts; frames: list of dicts."""
    rows_by_video = {}
    frames_by_video = {}
    summaries = {}
    if not os.path.isdir(logdir):
        return rows_by_video, frames_by_video, summaries
    for fn in os.listdir(logdir):
        if fn.endswith("_frames.csv"):
            video = fn[: -len("_frames.csv")]
            frames = []
            int_cols = ["frame", "eligible", "n_det", "n_candidate",
                        "n_triggered", "n_pool", "n_snapshot"]
            with open(os.path.join(logdir, fn), "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    for c in int_cols:
                        if c in row:
                            row[c] = int(row[c])
                    frames.append(row)
            frames_by_video[video] = frames
        elif fn.endswith("_summary.json"):
            with open(os.path.join(logdir, fn), "r", encoding="utf-8-sig") as f:
                summaries[fn[: -len("_summary.json")]] = json.load(f)
        elif fn.endswith(".csv"):
            video = fn[: -len(".csv")]
            rows = []
            with open(os.path.join(logdir, fn), "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    row["frame"] = int(row["frame"])
                    row["top1_tid"] = int(row["top1_tid"])
                    row["top1"] = float(row["top1"])
                    row["top2"] = float(row["top2"])
                    row["margin"] = float(row["margin"])
                    row["n_neighbor"] = int(row["n_neighbor"])
                    row["triggered"] = int(row["triggered"])
                    for c in ["det_x1", "det_y1", "det_x2", "det_y2"]:
                        if c in row:
                            row[c] = float(row[c])
                    rows.append(row)
            rows_by_video[video] = rows
    return rows_by_video, frames_by_video, summaries


def load_frozen_frame_counts(frozen_track_dir):
    """{video: {frame: row_count}} from frozen track_results files."""
    counts = {}
    if not os.path.isdir(frozen_track_dir):
        return counts
    for fn in os.listdir(frozen_track_dir):
        if not fn.endswith(".txt"):
            continue
        video = fn[: -len(".txt")]
        per_frame = {}
        with open(os.path.join(frozen_track_dir, fn), "r") as f:
            for line in f:
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                per_frame[int(parts[0])] = per_frame.get(int(parts[0]), 0) + 1
        counts[video] = per_frame
    return counts


def load_frozen_boxes(frozen_track_dir):
    """{video: {(frame, tid): (x1, y1, x2, y2)}} for box-level matching."""
    boxes = {}
    if not os.path.isdir(frozen_track_dir):
        return boxes
    for fn in os.listdir(frozen_track_dir):
        if not fn.endswith(".txt"):
            continue
        video = fn[: -len(".txt")]
        per_seq = {}
        with open(os.path.join(frozen_track_dir, fn), "r") as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 7:
                    continue
                fr, tid = int(p[0]), int(p[1])
                x1, y1, w, h = (float(v) for v in p[2:6])
                per_seq[(fr, tid)] = (x1, y1, x1 + w, y1 + h)
        boxes[video] = per_seq
    return boxes


def bbox_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def trig_from_features(top1, top2, margin, n_neighbor, eps0, gamma):
    """Offline trigger formula: top1>=0.2 & top2>=0.2 & margin < eps0*gamma(N)."""
    eps = eps0 * (gamma if n_neighbor > 0 else 1.0)
    return top1 >= TOP1_TH and top2 >= TOP2_TH and margin < eps


def main():
    ap = argparse.ArgumentParser(description="DEN online/offline alignment")
    ap.add_argument("--ds", required=True, choices=["mot17", "mot20", "sportsmot"])
    ap.add_argument("--logdir", required=True, help="path to den_gate_log/")
    ap.add_argument("--offline", required=True, help="taxonomy/gate_feasibility_events.csv")
    ap.add_argument("--frozen-track", required=True,
                    help="YOLOX_outputs/{ds}_v001_full/track_results "
                         "(event boxes; the frozen baseline run)")
    ap.add_argument("--same-track", required=True,
                    help="track_results of THIS alert run (snapshot consistency)")
    ap.add_argument("--tag", default="alert", help="output tag")
    args = ap.parse_args()

    rows_by_video, frames_by_video, summaries = load_logs(args.logdir)
    all_rows = [r for rows in rows_by_video.values() for r in rows]
    all_frames = [fr for frs in frames_by_video.values() for fr in frs]
    if not all_rows:
        print("ERROR: no gate log rows found in {}".format(args.logdir))
        sys.exit(1)
    has_box_cols = "det_x1" in all_rows[0]
    covered_videos = set(rows_by_video.keys())

    events = load_offline_events(args.offline, args.ds)
    events = [e for e in events if int(e.get("no_box", 0)) == 0
              and e["seq"] in covered_videos]

    # event frames over covered videos
    event_frames = set()
    for e in events:
        event_frames.add((e["seq"], int(e["frame"])))

    frozen_boxes = load_frozen_boxes(args.frozen_track)
    same_counts = load_frozen_frame_counts(args.same_track)

    print("=== {} | {} covered videos, {} eligible frames, {} candidate rows, "
          "det boxes logged: {} ==="
          .format(args.ds, len(covered_videos), len(all_frames), len(all_rows),
                  has_box_cols))

    # ---- FPR proxy: non-event eligible frames, anchor point ----
    norm_n_det = 0
    norm_n_trig = 0
    norm_n_cand = 0
    for video, frs in frames_by_video.items():
        for fr in frs:
            if not fr["eligible"]:
                continue
            if (video, fr["frame"]) in event_frames:
                continue
            norm_n_det += fr["n_det"]
            norm_n_cand += fr["n_candidate"]
            norm_n_trig += fr["n_triggered"]
    fpr_det = float(norm_n_trig) / norm_n_det if norm_n_det else float("nan")
    fpr_cand = float(norm_n_trig) / norm_n_cand if norm_n_cand else float("nan")

    # ---- per-event hit levels at anchor ----
    rows_at = {}
    for video, rows in rows_by_video.items():
        for r in rows:
            rows_at.setdefault((video, r["frame"]), []).append(r)

    def event_hit_levels(ev, raf, anchor):
        """frame-level (any triggered row) / box-level (triggered det box
        IoU>0.5 with the event track's frozen F box) / id-level (that row's
        top1_tid == frozen tid -- informational: track ids are NOT comparable
        across runs, so this is expected to be near zero)."""
        tid = int(ev["track_id"])
        fb = frozen_boxes.get(ev["seq"], {}).get((int(ev["frame"]), tid))
        frame_hit = box_hit = id_hit = 0
        for r in raf:
            t = trig_from_features(r["top1"], r["top2"], r["margin"],
                                   r["n_neighbor"], anchor[0], anchor[1])
            if not t:
                continue
            frame_hit = 1
            if has_box_cols and fb is not None:
                det = (r["det_x1"], r["det_y1"], r["det_x2"], r["det_y2"])
                if bbox_iou(fb, det) > 0.5:
                    box_hit = 1
                    if r["top1_tid"] == tid:
                        id_hit = 1
        return frame_hit, box_hit, id_hit

    tpr = {}
    for cls in ["S_r", "S_c", "S_h"]:
        cls_ev = [e for e in events if e.get("class") == cls]
        n = len(cls_ev)
        hits = {"frame": 0, "box": 0, "id": 0, "absent": 0}
        for e in cls_ev:
            raf = rows_at.get((e["seq"], int(e["frame"])), None)
            if raf is None:
                hits["absent"] += 1
                continue
            fh, bh, ih = event_hit_levels(e, raf, ANCHOR)
            hits["frame"] += fh
            hits["box"] += bh
            hits["id"] += ih
        denom = n - hits["absent"]
        tpr[cls] = {
            "n": n, "n_absent": hits["absent"],
            "frame": (float(hits["frame"]) / denom) if denom else float("nan"),
            "box": (float(hits["box"]) / denom) if denom else float("nan"),
            "id": (float(hits["id"]) / denom) if denom else float("nan"),
        }

    # ---- grid sweep (post-hoc) ----
    grid_rows = []
    for eps0 in EPS0_GRID:
        for gamma in GAMMA_GRID:
            g_n_trig = 0
            for video, frs in frames_by_video.items():
                for fr in frs:
                    if not fr["eligible"] or (video, fr["frame"]) in event_frames:
                        continue
                    for r in rows_at.get((video, fr["frame"]), []):
                        if trig_from_features(r["top1"], r["top2"], r["margin"],
                                              r["n_neighbor"], eps0, gamma):
                            g_n_trig += 1
            g_fpr = float(g_n_trig) / norm_n_det if norm_n_det else float("nan")
            g_tpr = {"frame": {}, "box": {}}
            for cls in ["S_r", "S_c", "S_h"]:
                hit_f = hit_b = 0
                denom = 0
                for e in events:
                    if e.get("class") != cls:
                        continue
                    raf = rows_at.get((e["seq"], int(e["frame"])), None)
                    if raf is None:
                        continue
                    denom += 1
                    fb = None
                    if has_box_cols:
                        fb = frozen_boxes.get(e["seq"], {}).get(
                            (int(e["frame"]), int(e["track_id"])))
                    f_hit = False
                    for r in raf:
                        if not trig_from_features(r["top1"], r["top2"], r["margin"],
                                                  r["n_neighbor"], eps0, gamma):
                            continue
                        f_hit = True
                        if fb is not None:
                            det = (r["det_x1"], r["det_y1"], r["det_x2"], r["det_y2"])
                            if bbox_iou(fb, det) > 0.5:
                                hit_b += 1
                                break
                    hit_f += int(f_hit)
                g_tpr["frame"][cls] = float(hit_f) / denom if denom else float("nan")
                g_tpr["box"][cls] = float(hit_b) / denom if denom else float("nan")
            grid_rows.append({
                "eps0": eps0, "gamma": gamma, "fpr_det": g_fpr,
                "tpr_Sr_frame": g_tpr["frame"]["S_r"],
                "tpr_Sr_box": g_tpr["box"]["S_r"],
                "tpr_Sc_frame": g_tpr["frame"]["S_c"],
                "tpr_Sc_box": g_tpr["box"]["S_c"],
                "tpr_Sh_frame": g_tpr["frame"]["S_h"],
                "tpr_Sh_box": g_tpr["box"]["S_h"],
            })

    # ---- feature alignment: S_r events, online row matched by BOX ----
    diffs = {"top1": [], "top2": [], "margin": [], "n": []}
    n_align = 0
    for e in events:
        if e.get("class") != "S_r":
            continue
        raf = rows_at.get((e["seq"], int(e["frame"])), None)
        if raf is None or not has_box_cols:
            continue
        tid = int(e["track_id"])
        fb = frozen_boxes.get(e["seq"], {}).get((int(e["frame"]), tid))
        if fb is None:
            continue
        best = None
        best_iou = 0.0
        for r in raf:
            det = (r["det_x1"], r["det_y1"], r["det_x2"], r["det_y2"])
            iou = bbox_iou(fb, det)
            if iou > best_iou:
                best_iou = iou
                best = r
        if best is None or best_iou < 0.5:
            continue
        diffs["top1"].append(best["top1"] - float(e["top1"]))
        diffs["top2"].append(best["top2"] - float(e["top2"]))
        diffs["margin"].append(best["margin"] - float(e["margin"]))
        if e.get("n_neighbor") is not None and str(e["n_neighbor"]).strip():
            diffs["n"].append(best["n_neighbor"] - int(float(e["n_neighbor"])))
        n_align += 1

    # ---- snapshot consistency vs THIS run's own track_results ----
    n_snap_checked = 0
    n_snap_mismatch = 0
    for video, frs in frames_by_video.items():
        fc = same_counts.get(video, {})
        for fr in frs:
            if not fr["eligible"]:
                continue
            prev_frame = fr["frame"] - 1
            if prev_frame in fc:
                n_snap_checked += 1
                if fc[prev_frame] != fr["n_snapshot"]:
                    n_snap_mismatch += 1

    # ---- timing ----
    gate_ms = [s["mean_gate_ms"] for s in summaries.values()
               if s.get("mean_gate_ms") is not None]
    snap_ms = [s["mean_snap_ms"] for s in summaries.values()
               if s.get("mean_snap_ms") is not None]

    # ================= output =================
    print("\n--- Trigger rates (anchor eps0={} gamma={}) ---".format(*ANCHOR))
    print("FPR proxy (normal frames): trigger_rate_det = {:.3%}, "
          "trigger_rate_candidate = {:.3%} (offline budget <=1%)"
          .format(fpr_det, fpr_cand))
    for cls in ["S_r", "S_c", "S_h"]:
        t = tpr[cls]
        print("{}: n={} (no log rows on event frame: {}) | frame-level={:.1%} "
              "box-level={:.1%} id-level={:.1%} (informational)"
              .format(cls, t["n"], t["n_absent"], t["frame"], t["box"], t["id"]))

    print("\n--- Grid sweep (post-hoc from logged features; frame-level / box-level) ---")
    hdr = "{:>5} {:>6} | {:>8} | {:>10} {:>10} | {:>8} {:>8}".format(
        "eps0", "gamma", "fpr_det", "tpr_Sr_f/b", "tpr_Sc_f/b", "tpr_Sh_f", "tpr_Sh_b")
    print(hdr)
    for g in grid_rows:
        print("{:>5.2f} {:>6.2f} | {:>8.3%} | {:>5.1%}/{:<4.1%} | {:>5.1%}/{:<4.1%} | {:>8.1%} {:>8.1%}".format(
            g["eps0"], g["gamma"], g["fpr_det"],
            g["tpr_Sr_frame"], g["tpr_Sr_box"],
            g["tpr_Sc_frame"], g["tpr_Sc_box"],
            g["tpr_Sh_frame"], g["tpr_Sh_box"]))

    print("\n--- Feature alignment (S_r, online row matched to the event track's "
          "frozen box by IoU>0.5, n={}) ---".format(n_align))
    for k, key in [("top1", "top1"), ("top2", "top2"), ("margin", "margin"),
                   ("N (top1 track vs own track)", "n")]:
        d = np.asarray(diffs[key])
        if len(d) == 0:
            print("  {}: no samples".format(k))
        else:
            print("  {}: median {:.4f} | P90 abs {:.4f} | mean abs {:.4f} (n={})"
                  .format(k, np.median(d), np.percentile(np.abs(d), 90),
                          np.mean(np.abs(d)), len(d)))

    print("\n--- Snapshot consistency ---")
    print("checked {} frames, mismatched {} ({:.2%})".format(
        n_snap_checked, n_snap_mismatch,
        float(n_snap_mismatch) / n_snap_checked if n_snap_checked else float("nan")))

    print("\n--- Timing ---")
    if gate_ms:
        print("mean gate block {:.3f} ms/frame (videos: {}), snapshot {:.3f} ms"
              .format(float(np.mean(gate_ms)), len(gate_ms),
                      float(np.mean(snap_ms)) if snap_ms else float("nan")))
    else:
        print("no summary timing available")

    # ================= write CSV =================
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taxonomy")
    out_dir = os.path.abspath(out_dir)
    out_path = os.path.join(out_dir, "den_online_{}_{}.csv".format(args.ds, args.tag))
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ds", "tag", "eps0", "gamma", "fpr_det",
                    "tpr_Sr_frame", "tpr_Sr_box", "tpr_Sc_frame", "tpr_Sc_box",
                    "tpr_Sh_frame", "tpr_Sh_box",
                    "fpr_anchor", "tpr_Sr_box_anchor",
                    "n_events_Sr", "n_events_Sc", "n_events_Sh"])
        for g in grid_rows:
            w.writerow([args.ds, args.tag, g["eps0"], g["gamma"], g["fpr_det"],
                        g["tpr_Sr_frame"], g["tpr_Sr_box"],
                        g["tpr_Sc_frame"], g["tpr_Sc_box"],
                        g["tpr_Sh_frame"], g["tpr_Sh_box"],
                        fpr_det, tpr["S_r"]["box"],
                        tpr["S_r"]["n"], tpr["S_c"]["n"], tpr["S_h"]["n"]])
    print("\nwrote {}".format(out_path))


if __name__ == "__main__":
    main()
