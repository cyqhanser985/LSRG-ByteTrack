# -*- coding: utf-8 -*-
"""Merge MOT-format frame sequences (img1/*.jpg) into a single mp4 video."""
import os
import sys
import cv2
import numpy as np


def imread_unicode(path):
    """cv2.imread fails on non-ASCII (Chinese) paths on Windows."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def merge_sequence(seq_dir, out_path, fps=30.0):
    img_dir = os.path.join(seq_dir, "img1")
    frames = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    frames.sort(key=lambda f: int(os.path.splitext(f)[0]))
    if not frames:
        raise RuntimeError(f"no frames in {img_dir}")

    first = imread_unicode(os.path.join(img_dir, frames[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    assert writer.isOpened(), f"cannot open writer for {out_path}"

    for i, f in enumerate(frames, 1):
        img = imread_unicode(os.path.join(img_dir, f))
        if img is None:
            print(f"  [warn] skip unreadable frame {f}")
            continue
        writer.write(img)
        if i % 200 == 0:
            print(f"  {i}/{len(frames)} frames")
    writer.release()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"OK {os.path.basename(out_path)}: {len(frames)} frames, {w}x{h} @ {fps}fps, {size_mb:.1f} MB")


if __name__ == "__main__":
    seq_dir, out_path = sys.argv[1], sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    merge_sequence(seq_dir, out_path, fps)
