#!/usr/bin/env python3
"""Render MOT-format tracking results onto original frames and encode mp4.

Usage:
    python tools/draw_tracks_video.py -expn mot17_v001_full -ds MOT17
    python tools/draw_tracks_video.py -expn mot17_v001_full -ds MOT17 --sequences V001
    python tools/draw_tracks_video.py -expn sportsmot_v001_full -ds SportsMOT

Details:
- Frames rendered: the contiguous range [min_frame, max_frame] present in the
  track txt (e.g. MOT17 val_half runs 301..600).
- Colors: per sequence, hue(id) = (id * 137.5) mod 180 (golden-angle step).
  The hue wheel is covered uniformly while numerically-adjacent ids stay
  >= 42.5 degrees apart in hue, so the same id keeps one stable color and
  nearby ids remain separable. Labels show the numeric id only
  (cv2.putText cannot render CJK).
- fps: 30 by default. The sequences carry no fps metadata; the report states
  this assumption. Resolution is the original frame size.
- Chinese paths are handled via cv2.imdecode(np.fromfile(...)).
Source is pure ASCII (English comments only).
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

DEFAULT_FPS = 30


def parse_track(txt_path):
    """Parse a MOT-format txt into {frame: [(x, y, w, h, tid), ...]}."""
    boxes = {}
    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            try:
                frame = int(float(p[0]))
                tid = int(float(p[1]))
                x = float(p[2])
                y = float(p[3])
                w = float(p[4])
                h = float(p[5])
            except ValueError:
                continue
            boxes.setdefault(frame, []).append((x, y, w, h, tid))
    return boxes


def build_id_colors(boxes):
    """Map every track id to a stable color (golden-angle hue stepping)."""
    ids = sorted({b[4] for fr in boxes.values() for b in fr})
    colors = {}
    for tid in ids:
        hue = (tid * 137.5) % 180.0
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors[tid] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return colors


def read_image(path):
    """Read an image with Chinese-path support; None on failure."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def draw_boxes(img, boxes, colors, line_w, font_scale):
    """Draw boxes and numeric id labels onto img in place."""
    for (x, y, w, h, tid) in boxes:
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        color = colors[tid]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_w)
        label = str(tid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 2)
        ly = y1 - 6
        if ly - th < 0:
            ly = y2 + th + 4
        cv2.rectangle(img, (x1, ly - th - 2), (x1 + tw + 4, ly), (0, 0, 0), -1)
        cv2.putText(img, label, (x1 + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)


def write_video(img_dir, boxes, out_path, fps):
    """Encode one sequence video; returns (written, missing) or None on error."""
    frames = sorted(boxes.keys())
    first = frames[0]
    last = frames[-1]

    probe = read_image(os.path.join(img_dir, "%06d.jpg" % first))
    if probe is None:
        # probe the very last frame for size instead
        probe = read_image(os.path.join(img_dir, "%06d.jpg" % last))
    if probe is None:
        print("  [error] cannot read any image of", img_dir)
        return None
    h, w = probe.shape[:2]

    colors = build_id_colors(boxes)
    line_w = max(1, int(round(h / 540.0)))
    font_scale = max(0.5, round(w / 1920.0 * 0.9, 2))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (w, h))
    if not writer.isOpened():
        print("  [error] cannot open video writer:", out_path)
        return None

    missing = 0
    written = 0
    for frame in range(first, last + 1):
        img = read_image(os.path.join(img_dir, "%06d.jpg" % frame))
        if img is None:
            missing += 1
            img = np.zeros((h, w, 3), dtype=np.uint8)
        draw_boxes(img, boxes.get(frame, []), colors, line_w, font_scale)
        writer.write(img)
        written += 1
    writer.release()
    return written, missing


def main():
    ap = argparse.ArgumentParser(
        description="Render MOT-format tracking results into mp4 videos.")
    ap.add_argument("-expn", required=True,
                    help="experiment name under YOLOX_outputs")
    ap.add_argument("-ds", required=True,
                    help="dataset dir name under datasets/, e.g. MOT17")
    ap.add_argument("--sequences", default=None,
                    help="comma-separated sequence filter, e.g. V001,V002")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS,
                    help="output frame rate (default 30)")
    args = ap.parse_args()

    res_dir = os.path.join("YOLOX_outputs", args.expn, "track_results")
    out_dir = os.path.join("YOLOX_outputs", args.expn, "track_videos")
    os.makedirs(out_dir, exist_ok=True)
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
    n_fail = 0
    for txt in txts:
        seq = os.path.splitext(os.path.basename(txt))[0]
        img_dir = os.path.join(seq_root, seq, "img1")
        if not os.path.isdir(img_dir):
            print("[skip] %s: img dir missing %s" % (seq, img_dir))
            n_fail += 1
            continue
        boxes = parse_track(txt)
        n_frames = len(boxes)
        n_boxes = sum(len(v) for v in boxes.values())
        print("== %s: %d frames, %d boxes" % (seq, n_frames, n_boxes))
        out_path = os.path.join(out_dir, seq + ".mp4")
        res = write_video(img_dir, boxes, out_path, args.fps)
        if res is None:
            n_fail += 1
        else:
            written, missing = res
            n_ok += 1
            if written != n_frames or missing > 0:
                print("  [warn] %s: written=%d txt_frames=%d missing_imgs=%d"
                      % (seq, written, n_frames, missing))
        sys.stdout.flush()

    print("done: %d videos ok, %d failed" % (n_ok, n_fail))


if __name__ == "__main__":
    main()
