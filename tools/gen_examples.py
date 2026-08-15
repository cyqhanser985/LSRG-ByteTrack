#!/usr/bin/env python3
"""Pick typical ID-switch / ID-reuse examples and render annotated PNGs.

Usage:
    python tools/gen_examples.py -expn mot17_v001_full -ds MOT17
    python tools/gen_examples.py -expn sportsmot_v001_full -ds SportsMOT

For each dataset the script scans the frozen events CSV (canonical source:
research/data/{dataset}_events.csv) and scores candidate events:

  * switch: the OLD tracker must still be visible inside the +-15 frame
    window (so the viewer can see its color), its box at the event frame
    must not fully overlap the gt box (IoU < 0.5), and the NEW tracker
    must be present at the event frame.
  * reuse: the tracker must be present both before and after the event, and
    the old/new gt boxes must be spatially separated (IoU < 0.4).

The top-scoring 2 candidates per type with frame numbers at least 50 apart
are rendered: for each frame in [event-15, event+15] one PNG with all track
boxes (same colors as the videos, see draw_tracks_video.py) plus an ASCII
caption bar. The involved tracker id gets a thicker border.

Outputs go to YOLOX_outputs/analysis/examples/. Source is pure ASCII.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import draw_tracks_video as dtv

REPO_ROOT = Path(__file__).resolve().parents[1]

WIN = 15
MIN_FRAME_GAP = 50
OUT = REPO_ROOT / "YOLOX_outputs" / "analysis" / "examples"


def box_iou(a, b):
    """IoU of two (x, y, w, h) boxes; 0 if either is None."""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (aa + bb - inter)


def box_at(boxes, frame, tid):
    """Return the (x,y,w,h) box of tid at frame, or None."""
    for b in boxes.get(frame, []):
        if b[4] == tid:
            return (b[0], b[1], b[2], b[3])
    return None


def score_switch(boxes, frame, oid, hid, old_hid):
    """Return a quality score or -1 if the candidate is not usable."""
    lo, hi = frame - WIN, frame + WIN
    if min(boxes) > lo or max(boxes) < hi:
        return -1.0
    if old_hid is None:
        return -1.0
    old_vis = 0
    for f in range(lo, hi + 1):
        if box_at(boxes, f, old_hid) is not None:
            old_vis += 1
    if old_vis < 5:
        return -1.0
    new_box = box_at(boxes, frame, hid)
    old_box = box_at(boxes, frame, old_hid)
    if new_box is None:
        return -1.0
    # the old tracker's box at the event frame must not sit on the new match
    overlap = box_iou(old_box, new_box)
    if overlap > 0.5:
        return -1.0
    return old_vis + (1.0 - overlap) + \
        (0.2 if box_at(boxes, frame - 1, hid) is not None else 0.0)


def score_reuse(boxes, frame, hid, oid_old, oid_new, note_old_last):
    """Return a quality score or -1 if the candidate is not usable."""
    lo, hi = frame - WIN, frame + WIN
    if min(boxes) > lo or max(boxes) < hi:
        return -1.0
    # the tracker must have boxes around the event on both sides
    before = box_at(boxes, lo, hid)
    after = box_at(boxes, hi, hid)
    if before is None or after is None:
        return -1.0
    old_box = box_at(boxes, frame - 1, hid)
    new_box = box_at(boxes, frame, hid)
    if old_box is None or new_box is None:
        return -1.0
    # spatially separated targets read best in a still image
    sep = 1.0 - box_iou(old_box, new_box)
    return sep + 0.2 * (1.0 - box_iou(before, after))


def pick_examples(events, boxes, scorer, max_per_type):
    """Pick up to max_per_type examples with frame numbers >= gap apart."""
    scored = []
    for ev in events:
        s = scorer(boxes, ev)
        if s is None or s < 0:
            continue
        scored.append((s, ev))
    scored.sort(key=lambda t: -t[0])
    picked = []
    for s, ev in scored:
        if all(abs(ev[1] - p[1][1]) >= MIN_FRAME_GAP for p in picked):
            picked.append((s, ev))
            if len(picked) >= max_per_type:
                break
    return picked


def render_window(seq_root, seq, boxes, frame, caption, highlight_id, out_prefix):
    """Render frames [frame-15, frame+15] as PNGs with track boxes + labels."""
    colors = dtv.build_id_colors(boxes)
    img_dir = os.path.join(seq_root, seq, "img1")
    h, w = 1080, 1920
    for f in range(frame - WIN, frame + WIN + 1):
        img = dtv.read_image(os.path.join(img_dir, "%06d.jpg" % f))
        if img is None:
            continue
        h, w = img.shape[:2]
        line_w = max(1, int(round(h / 540.0)))
        font_scale = max(0.5, round(w / 1920.0 * 0.9, 2))
        for b in boxes.get(f, []):
            lw = line_w + 2 if b[4] == highlight_id else line_w
            x1, y1 = int(round(b[0])), int(round(b[1]))
            x2, y2 = int(round(b[0] + b[2])), int(round(b[1] + b[3]))
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[b[4]], lw)
            label = str(b[4])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, 2)
            ly = y1 - 6
            if ly - th < 0:
                ly = y2 + th + 4
            cv2.rectangle(img, (x1, ly - th - 2), (x1 + tw + 4, ly),
                          (0, 0, 0), -1)
            cv2.putText(img, label, (x1 + 2, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[b[4]], 2)
        cv2.rectangle(img, (0, 0), (w - 1, 34), (0, 0, 0), -1)
        cv2.putText(img, "frame %d | %s" % (f, caption),
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out = "%s_f%06d.png" % (out_prefix, f)
        cv2.imencode(".png", img)[1].tofile(out)
    return h, w


def main():
    ap = argparse.ArgumentParser(
        description="Pick ID-switch/reuse examples and render PNG windows.")
    ap.add_argument("-expn", required=True,
                    help="experiment name under YOLOX_outputs")
    ap.add_argument("-ds", required=True,
                    help="dataset dir name under datasets/, e.g. MOT17")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    seq_root = REPO_ROOT / "datasets" / args.ds
    events_csv = REPO_ROOT / "research" / "data" / ("%s_events.csv" % args.ds)

    switch_evs = []
    reuse_evs = []
    for line in open(events_csv, encoding="utf-8").read().splitlines()[1:]:
        p = line.split(",")
        seq = p[0]
        frame = int(p[1])
        typ = p[2]
        gt_old = int(p[3]) if p[3] != "" else 0
        gt_new = int(p[4]) if p[4] != "" else 0
        tid = int(p[5])
        note = ",".join(p[6:]).strip()
        if typ == "switch":
            old_hid = None
            nw = note.split()
            if len(nw) > 1 and nw[1] != "unknown":
                old_hid = int(nw[1])
            switch_evs.append((seq, frame, gt_old, tid, old_hid, note))
        elif typ == "reuse":
            # note: "tracker %d matched gt %d (last frame %d) now gt %d"
            import re
            m = re.findall(r"\d+", note)
            old_last = int(m[2]) if len(m) >= 3 else frame
            reuse_evs.append((seq, frame, tid, gt_old, gt_new, old_last, note))

    # group events by sequence
    from collections import defaultdict
    sw_by_seq = defaultdict(list)
    ru_by_seq = defaultdict(list)
    for ev in switch_evs:
        sw_by_seq[ev[0]].append(ev)
    for ev in reuse_evs:
        ru_by_seq[ev[0]].append(ev)

    all_seqs = sorted(set(list(sw_by_seq.keys()) + list(ru_by_seq.keys())))

    # pick a small set of sequences that contain both kinds of events when
    # possible, scoring candidates per sequence
    sw_pick = []
    ru_pick = []
    for seq in all_seqs:
        txt = REPO_ROOT / "YOLOX_outputs" / args.expn / "track_results" / ("%s.txt" % seq)
        if not txt.exists():
            continue
        boxes = dtv.parse_track(txt)
        if not boxes:
            continue
        sw_cands = sw_by_seq.get(seq, [])
        ru_cands = ru_by_seq.get(seq, [])
        if sw_cands:
            sw_pick.extend(pick_examples(
                sw_cands, boxes,
                lambda b, ev: score_switch(b, ev[1], ev[2], ev[3], ev[4]), 2))
        if ru_cands:
            ru_pick.extend(pick_examples(
                ru_cands, boxes,
                lambda b, ev: score_reuse(b, ev[1], ev[2], ev[3], ev[4],
                                          ev[5]), 2))

    # sort all picked and take 2 switch + 2 reuse across the dataset,
    # re-applying the frame gap across types and sequences
    def dedup(picked, n):
        out = []
        for s, ev in sorted(picked, key=lambda t: -t[0]):
            if any(abs(ev[1] - q[1][1]) < MIN_FRAME_GAP for q in out):
                continue
            out.append((s, ev))
            if len(out) >= n:
                break
        return out

    sw_final = dedup(sw_pick, 2)
    ru_final = dedup(ru_pick, 2)

    print("picked switch examples:")
    for s, ev in sw_final:
        seq, frame, oid, hid, old_hid, note = ev
        print("  %s f%d gt%d tracker %s->%d (score %.2f)"
              % (seq, frame, oid, old_hid, hid, s))
    print("picked reuse examples:")
    for s, ev in ru_final:
        seq, frame, tid, gt_old, gt_new, old_last, note = ev
        print("  %s f%d tracker %d gt %d->%d (last %d) (score %.2f)"
              % (seq, frame, tid, gt_old, gt_new, old_last, s))

    rendered = []
    for s, ev in sw_final:
        seq, frame, oid, hid, old_hid, note = ev
        txt = REPO_ROOT / "YOLOX_outputs" / args.expn / "track_results" / ("%s.txt" % seq)
        boxes = dtv.parse_track(txt)
        prefix = os.path.join(OUT, "%s_switch_%s_f%06d" % (args.ds, seq, frame))
        caption = "%s f%d SWITCH gt%d tracker%s->%d" % (
            seq, frame, oid,
            "unknown" if old_hid is None else str(old_hid), hid)
        render_window(seq_root, seq, boxes, frame,
                      caption, hid, prefix)
        rendered.append((seq, frame, "switch", hid, caption))
    for s, ev in ru_final:
        seq, frame, tid, gt_old, gt_new, old_last, note = ev
        txt = REPO_ROOT / "YOLOX_outputs" / args.expn / "track_results" / ("%s.txt" % seq)
        boxes = dtv.parse_track(txt)
        prefix = os.path.join(OUT, "%s_reuse_%s_f%06d" % (args.ds, seq, frame))
        caption = "%s f%d REUSE tracker%d gt%d->%d" % (
            seq, frame, tid, gt_old, gt_new)
        render_window(seq_root, seq, boxes, frame,
                      caption, tid, prefix)
        rendered.append((seq, frame, "reuse", tid, caption))

    with open(os.path.join(OUT, "picked_examples_%s.txt" % args.ds),
              "w", encoding="utf-8") as f:
        for (seq, frame, typ, tid, caption) in rendered:
            f.write("%s %d %s %d %s\n" % (seq, frame, typ, tid, caption))
    print("PNG windows written to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
