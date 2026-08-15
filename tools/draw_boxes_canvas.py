#!/usr/bin/env python3
"""Draw MOT-format tracking results onto a pure white canvas (no background).

Usage:
    python tools/draw_boxes_canvas.py -expn mot17_v001_full -ds MOT17 --sequences V013
    python tools/draw_boxes_canvas.py -expn sportsmot_v001_full -ds SportsMOT --sequences V019
    python tools/draw_boxes_canvas.py -expn mot17_v001_full -ds MOT17 --sequences V013 --frames 330-360

Details:
- Every frame is a pure white image of the sequence's original WxH (size
  probed once via read_image on frame 1), so box coordinates stay exact
  while the background disappears - only detection boxes are visible.
- Reuses draw_tracks_video.parse_track / build_id_colors / draw_boxes /
  read_image, so id colors (golden-angle hue) and the black label-bar style
  are identical to the background-track videos.
- Outputs:
    full range:  YOLOX_outputs/analysis/canvas/{seq}/f%06d.png
                 YOLOX_outputs/analysis/canvas_videos/{seq}.mp4 (mp4v)
    --frames:    YOLOX_outputs/analysis/canvas/examples/{seq}/f%06d.png
                 (event-window mode, PNGs only)
- --no-label draws boxes only (no id number / black label bar).
- --outline draws a black outline under each colored box for readability on
  the white background (see canvas_report.md for the contrast rationale).
- fps is 30 by default; the sequences carry no fps metadata.
- Chinese paths are handled via cv2.imencode + tofile and np.fromfile.
Source is pure ASCII (English comments only).
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

import draw_tracks_video as dtv

CANVAS_DIR = os.path.join("YOLOX_outputs", "analysis", "canvas")
VIDEO_DIR = os.path.join("YOLOX_outputs", "analysis", "canvas_videos")
EXAMPLE_DIR = os.path.join(CANVAS_DIR, "examples")


def draw_boxes_only(img, boxes, colors, line_w):
    """Draw boxes without id labels (used for --no-label)."""
    for (x, y, w, h, tid) in boxes:
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[tid], line_w)


def draw_outlines(img, boxes, line_w):
    """Draw a black outline under each colored box (readability on white)."""
    for (x, y, w, h, tid) in boxes:
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), line_w + 2)


def probe_size(seq_root, seq):
    """Return (h, w) of the sequence images, reading one frame only."""
    img_dir = os.path.join(seq_root, seq, "img1")
    img = dtv.read_image(os.path.join(img_dir, "000001.jpg"))
    if img is None:
        img = dtv.read_image(os.path.join(img_dir, "000000.jpg"))
    if img is None:
        return None
    return img.shape[:2]


def read_canvas_png(path):
    """Read a rendered canvas PNG (Chinese-path safe)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def render_sequence(args, seq_root, seq, boxes):
    """Render one sequence (or a frame window) onto white canvases."""
    frames = sorted(boxes.keys())
    f_lo, f_hi = frames[0], frames[-1]
    is_window = False
    if args.frames:
        lo, hi = args.frames
        if hi < f_lo or lo > f_hi:
            print("  [skip] %s: window %d-%d outside track range %d-%d"
                  % (seq, lo, hi, f_lo, f_hi))
            return False
        f_lo, f_hi = max(f_lo, lo), min(f_hi, hi)
        out_dir = os.path.join(EXAMPLE_DIR, seq)
        is_window = True
    else:
        out_dir = os.path.join(CANVAS_DIR, seq)
    os.makedirs(out_dir, exist_ok=True)

    size = probe_size(seq_root, seq)
    if size is None:
        print("  [error] %s: cannot read any frame image" % seq)
        return False
    h, w = size
    colors = dtv.build_id_colors(boxes)
    line_w = max(1, int(round(h / 540.0)))
    font_scale = max(0.5, round(w / 1920.0 * 0.9, 2))

    n_png = 0
    for frame in range(f_lo, f_hi + 1):
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        fb = boxes.get(frame, [])
        if args.outline:
            draw_outlines(img, fb, line_w)
        if args.no_label:
            draw_boxes_only(img, fb, colors, line_w)
        else:
            dtv.draw_boxes(img, fb, colors, line_w, font_scale)
        out = os.path.join(out_dir, "f%06d.png" % frame)
        cv2.imencode(".png", img)[1].tofile(out)
        n_png += 1

    if is_window:
        print("  %s window %d-%d: %d PNGs -> %s"
              % (seq, f_lo, f_hi, n_png, out_dir))
        return True

    os.makedirs(VIDEO_DIR, exist_ok=True)
    vpath = os.path.join(VIDEO_DIR, seq + ".mp4")
    writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    if not writer.isOpened():
        print("  [error] %s: cannot open video writer %s" % (seq, vpath))
        return False
    # encode from the PNGs so mp4 and PNG frames are pixel-identical
    for frame in range(f_lo, f_hi + 1):
        img = read_canvas_png(os.path.join(out_dir, "f%06d.png" % frame))
        writer.write(img)
    writer.release()
    print("  %s frames %d-%d: %d PNGs + %s"
          % (seq, f_lo, f_hi, n_png, vpath))
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Render MOT tracking results on white canvases.")
    ap.add_argument("-expn", required=True,
                    help="experiment name under YOLOX_outputs")
    ap.add_argument("-ds", required=True,
                    help="dataset dir name under datasets/, e.g. MOT17")
    ap.add_argument("--sequences", default=None,
                    help="comma-separated sequence filter, e.g. V001,V002")
    ap.add_argument("--frames", default=None,
                    help="frame window lo-hi, e.g. 330-360 (event-window mode)")
    ap.add_argument("--no-label", action="store_true",
                    help="draw boxes only, skip id labels")
    ap.add_argument("--outline", action="store_true",
                    help="draw a black outline under each colored box")
    ap.add_argument("--fps", type=int, default=dtv.DEFAULT_FPS,
                    help="output frame rate (default 30)")
    args = ap.parse_args()

    if args.frames:
        try:
            lo_s, hi_s = args.frames.split("-")
            args.frames = (int(lo_s), int(hi_s))
        except ValueError:
            print("--frames must look like 330-360")
            sys.exit(1)

    res_dir = os.path.join("YOLOX_outputs", args.expn, "track_results")
    seq_root = os.path.join("datasets", args.ds)
    txts = sorted(glob.glob(os.path.join(res_dir, "*.txt")))
    if args.sequences:
        wanted = set(args.sequences.split(","))
        txts = [t for t in txts
                if os.path.splitext(os.path.basename(t))[0] in wanted]
    if not txts:
        print("no track files found under", res_dir)
        sys.exit(1)

    n_ok = 0
    for txt in txts:
        seq = os.path.splitext(os.path.basename(txt))[0]
        boxes = dtv.parse_track(txt)
        if not boxes:
            print("[skip] %s: empty track file" % seq)
            continue
        if render_sequence(args, seq_root, seq, boxes):
            n_ok += 1
        sys.stdout.flush()

    print("done: %d sequences rendered" % n_ok)


if __name__ == "__main__":
    main()
