#!/usr/bin/env python3
"""Draw full-trajectory plots of trackers involved in ID-switch events.

Usage:
    python tools/draw_switch_trajectory.py            # all 6 examples
    python tools/draw_switch_trajectory.py -ds MOT17  # one dataset only

For each selected event (looked up in the events csvs by seq/frame) the
involved tracker's whole trajectory (first..last appearance) is drawn on a
pure-white canvas of the original frame size (read once from the first
frame of the sequence):

  * polyline through box centers of every appearance, drawn thick black
    underlay + thin colored line
  * frames < event frame F keep the tracker's golden-angle color
    (dtv.build_id_colors, consistent with videos/canvas), frames >= F use
    pure red BGR (0, 0, 255)
  * one small dot per frame, downsampled to MAX_DOTS when the track is
    longer (the polyline stays complete)
  * switch frame F gets a black ring plus an "f<F>" text label
  * first/last frames get black/white dual-color square markers
  * switch examples additionally draw the OLD tracker's whole trajectory
    as thin gray line + gray dots (no labels); reuse examples draw nothing
    extra (same tracker id, no second id involved)
  * light gray rectangle along the canvas border as a frame of reference
  * ASCII title bar at the top: "{seq} {type} f{frame} {old}->{new}"
    (cv2.putText cannot render CJK; the Chinese explanation lives in
    switch_metrics_report.md)

Outputs:
    YOLOX_outputs/analysis/trajectory/{DS}_{type}_{seq}_f%06d.png

Source is pure ASCII (English comments only).
"""

import argparse
import os

import cv2
import numpy as np

import draw_tracks_video as dtv

DS_EXPN = {"MOT17": "mot17_v001_full",
           "MOT20": "mot20_v001_full",
           "SportsMOT": "sportsmot_v001_full"}
ANALYSIS = os.path.join("YOLOX_outputs", "analysis")
OUT = os.path.join(ANALYSIS, "trajectory")

RED = (0, 0, 255)
GRAY = (130, 130, 130)
LIGHT_GRAY = (210, 210, 210)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

BAR_H = 40          # title bar height
MAX_DOTS = 500      # dot downsampling threshold
DOT_R = 2           # dot radius
DOT_OL = 1          # dot black outline width
LINE_W = 4          # colored polyline width
LINE_BLACK_W = 6    # black underlay width
OLD_W = 2           # old tracker gray polyline width
RING_R = 12         # switch-frame ring radius
RING_W = 4          # switch-frame ring width

# Picked with a frame-count + path-length scoring pass (see
# switch_metrics_report.md); every pick has >= 30 frames and a sizeable
# displacement on both sides of the event frame so the plot reads well.
EXAMPLES = [
    ("MOT17", "V001", 452, "switch", 30),
    ("MOT17", "V019", 431, "reuse", 996),
    ("MOT20", "V002", 2246, "switch", 539),
    ("MOT20", "V004", 2052, "reuse", 3445),
    ("SportsMOT", "V019", 281, "switch", 949),
    ("SportsMOT", "V046", 546, "reuse", 2288),
]


def box_at(boxes, frame, tid):
    """Return the (x, y, w, h) box of tid at frame, or None."""
    for b in boxes.get(frame, []):
        if b[4] == tid:
            return (b[0], b[1], b[2], b[3])
    return None


def track_centers(boxes, tid):
    """Sorted [(frame, (cx, cy)), ...] of every appearance of tid."""
    pts = []
    for f in sorted(boxes.keys()):
        b = box_at(boxes, f, tid)
        if b is not None:
            pts.append((f, (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)))
    return pts


def load_event(ds, seq, frame, tid_want):
    """Look up (type, gt_old, gt_new, tid, old_hid) in the events csv."""
    path = os.path.join(ANALYSIS, "%s_events.csv" % ds)
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        p = line.split(",")
        if p[0] != seq or int(p[1]) != frame or int(p[5]) != tid_want:
            continue
        typ = p[2]
        gt_old = int(p[3]) if p[3] else 0
        gt_new = int(p[4]) if p[4] else 0
        tid = int(p[5])
        note = ",".join(p[6:])
        old_hid = None
        if typ == "switch":
            nw = note.split()
            if len(nw) > 1 and nw[1] != "unknown":
                old_hid = int(nw[1])
        return typ, gt_old, gt_new, tid, old_hid
    return None


def polyline(img, pts, color, thick):
    """Draw an open polyline through pts (list of (x, y) floats)."""
    if len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], False, color, thick, cv2.LINE_AA)


def dot(img, x, y, color, r=DOT_R, outline=DOT_OL):
    """Small filled dot with a black outline for white-canvas contrast."""
    if outline > 0:
        cv2.circle(img, (int(round(x)), int(round(y))), r + outline, BLACK, -1)
    cv2.circle(img, (int(round(x)), int(round(y))), r, color, -1)


def square_marker(img, x, y, outer, inner, half=9):
    """Black/white dual-color square marker around (x, y)."""
    xi, yi = int(round(x)), int(round(y))
    cv2.rectangle(img, (xi - half, yi - half), (xi + half, yi + half),
                  outer, -1)
    cv2.rectangle(img, (xi - half // 2, yi - half // 2),
                  (xi + half // 2, yi + half // 2), inner, -1)


def title_bar(img, w, text):
    """Black bar across the top with white ASCII text (inset corners)."""
    cv2.rectangle(img, (2, 2), (w - 3, BAR_H - 1), BLACK, -1)
    cv2.putText(img, text, (10, BAR_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)


def frame_label(img, w, h, cx, cy, text, centers):
    """Text label near (cx, cy) with white backing, placed in whitespace.

    Tries four offsets and picks the first candidate whose box does not
    overlap any track center (plus margin) and stays inside the canvas, so
    the label never covers the trajectory.
    """
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 6
    cands = [(36, -36), (-36 - tw - pad, -36), (36, 36),
             (-36 - tw - pad, 36)]
    for dx, dy in cands:
        x0 = int(round(cx + dx))
        y0 = int(round(cy + dy))
        if (x0 < 2 or x0 + tw + pad > w - 2 or
                y0 - th - 2 < BAR_H or y0 + pad > h - 2):
            continue
        box = (x0 - 8, y0 - th - pad - 4, x0 + tw + pad + 8, y0 + 8)
        if any(box[0] <= px <= box[2] and box[1] <= py <= box[3]
               for px, py in centers):
            continue
        cv2.rectangle(img, (x0, y0), (x0 + tw + pad, y0 - th - pad + 4),
                      WHITE, -1)
        cv2.rectangle(img, (x0, y0), (x0 + tw + pad, y0 - th - pad + 4),
                      BLACK, 1)
        cv2.putText(img, text, (x0 + pad // 2, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
        return
    # fallback: draw plain text below the title bar (may touch the track)
    cv2.putText(img, text, (int(round(cx)), BAR_H + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)


def dot_sampling(n):
    """Indices of frames that get a dot (downsampled past MAX_DOTS)."""
    if n <= MAX_DOTS:
        return list(range(n))
    step = int(np.ceil(n / float(MAX_DOTS)))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def draw_example(ds, seq, F, typ, gt_old, gt_new, tid, old_hid):
    """Render one trajectory plot; returns (h, w) or None on failure."""
    expn = DS_EXPN[ds]
    boxes = dtv.parse_track(os.path.join(
        "YOLOX_outputs", expn, "track_results", seq + ".txt"))
    if not boxes:
        print("  [skip] %s: no track boxes" % seq)
        return None

    first = min(boxes)
    img_dir = os.path.join("datasets", ds, seq, "img1")
    probe = dtv.read_image(os.path.join(img_dir, "%06d.jpg" % first))
    if probe is None:
        print("  [skip] %s: cannot read first frame" % seq)
        return None
    h, w = probe.shape[:2]
    canvas = np.full((h, w, 3), 255, np.uint8)

    # border frame of reference (inset so the four corners stay white)
    cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), LIGHT_GRAY, 2)

    colors = dtv.build_id_colors(boxes)
    color_tid = colors[tid]

    pts = track_centers(boxes, tid)
    if len(pts) < 2:
        print("  [skip] %s: tracker %d too short (%d frames)"
              % (seq, tid, len(pts)))
        return None
    before = [(x, y) for f, (x, y) in pts if f < F]
    after = [(x, y) for f, (x, y) in pts if f >= F]

    # old tracker gray trajectory first (under the main track)
    if typ == "switch" and old_hid is not None:
        opts = track_centers(boxes, old_hid)
        if len(opts) >= 2:
            polyline(canvas, [(x, y) for _, (x, y) in opts], GRAY, OLD_W)
            for _, (x, y) in opts:
                dot(canvas, x, y, GRAY, r=2, outline=0)

    # start / end markers first (under the track, so dense dots near the
    # start stay visible)
    all_pts = [(x, y) for _, (x, y) in pts]
    square_marker(canvas, all_pts[0][0], all_pts[0][1], BLACK, WHITE)
    square_marker(canvas, all_pts[-1][0], all_pts[-1][1], WHITE, BLACK)

    # colored polylines: thick black underlay then thin color
    if len(before) >= 2:
        polyline(canvas, before, BLACK, LINE_BLACK_W)
        polyline(canvas, before, color_tid, LINE_W)
    if len(after) >= 2:
        polyline(canvas, after, BLACK, LINE_BLACK_W)
        polyline(canvas, after, RED, LINE_W)

    # dots
    for i in dot_sampling(len(pts)):
        x, y = all_pts[i]
        f_i = pts[i][0]
        col = color_tid if f_i < F else RED
        dot(canvas, x, y, col)

    # switch-frame ring + label (label placed in whitespace, away from the
    # track centers)
    for f_i, (x, y) in pts:
        if f_i == F:
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(canvas, (xi, yi), RING_R, BLACK, RING_W)
            frame_label(canvas, w, h, x, y, "f%d" % F, all_pts)
            break

    # title bar
    if typ == "switch":
        old_txt = "unknown" if old_hid is None else str(old_hid)
        caption = "%s switch f%d %s->%d" % (seq, F, old_txt, tid)
    else:
        caption = "%s reuse f%d %d->%d" % (seq, F, gt_old, gt_new)
    title_bar(canvas, w, caption)

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "%s_%s_%s_f%06d.png" % (ds, typ, seq, F))
    cv2.imencode(".png", canvas)[1].tofile(out)
    print("%s -> %s (%dx%d, %d frames, dots %d)"
          % (caption, out, w, h, len(pts), len(dot_sampling(len(pts)))))
    return h, w


def main():
    ap = argparse.ArgumentParser(
        description="Draw ID-switch trajectory plots on white canvases.")
    ap.add_argument("-ds", default=None,
                    help="dataset filter, e.g. MOT17 (default: all)")
    args = ap.parse_args()

    n_ok = 0
    for ds, seq, F, typ, tid_want in EXAMPLES:
        if args.ds and ds != args.ds:
            continue
        ev = load_event(ds, seq, F, tid_want)
        if ev is None:
            print("  [skip] %s %s f%d: event not found in csv"
                  % (ds, seq, F))
            continue
        got_typ, gt_old, gt_new, tid, old_hid = ev
        if got_typ != typ:
            print("  [warn] %s %s f%d: csv type %s, expected %s"
                  % (ds, seq, F, got_typ, typ))
        if draw_example(ds, seq, F, typ, gt_old, gt_new, tid, old_hid):
            n_ok += 1
    print("done: %d plots" % n_ok)


if __name__ == "__main__":
    main()
