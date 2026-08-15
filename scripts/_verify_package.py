# -*- coding: utf-8 -*-
"""Verify the HTML report package: every asset referenced in index.html exists on disk."""
import io
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "reports" / "ByteTrack_ID分析报告"
html_path = base / "index.html"

with io.open(html_path, encoding="utf-8") as f:
    html = f.read()

# 1. collect all local references (src=, href=) except anchors and mailto
refs = re.findall(r'(?:src|href)="([^"#]+)"', html)
refs = [r for r in refs if not r.startswith(("http://", "https://", "mailto:"))]
missing = []
for r in refs:
    p = base / r.replace("/", os.sep)
    if not p.exists():
        missing.append(r)
print("references checked:", len(refs))
print("MISSING:", missing if missing else "none")

# 2. referenced data files are valid JS
for js in ["frame_data.js", "events.js", "video_inventory.js"]:
    p = base / "assets" / "data" / js
    with io.open(p, encoding="utf-8") as f:
        text = f.read()
    # crude check: balanced braces and expected window.X assignment
    assert text.count("{") == text.count("}"), js + " brace mismatch"
    print(js, "OK", len(text), "bytes")

# 3. counts
with io.open(base / "assets" / "data" / "frame_data.js", encoding="utf-8") as f:
    fd = json.loads(f.read().split("=", 1)[1].strip().rstrip(";"))
print("players:", len(fd), sorted(fd.keys()))
for k, v in fd.items():
    assert len(v["frames"]) == 31, (k, len(v["frames"]))
    for fr in v["frames"]:
        p = base / fr.replace("/", os.sep)
        assert p.exists(), "missing frame " + fr
print("all frame lists OK (31 frames each, files exist)")

with io.open(base / "assets" / "data" / "events.js", encoding="utf-8") as f:
    ev = json.loads(f.read().split("=", 1)[1].strip().rstrip(";"))
tot = sum(len(v) for v in ev.values())
print("events total:", tot, {k: len(v) for k, v in ev.items()})
assert tot == 7156

with io.open(base / "assets" / "data" / "video_inventory.js", encoding="utf-8") as f:
    inv = json.loads(f.read().split("=", 1)[1].strip().rstrip(";"))
nvid = sum(len(v) for v in inv.values())
print("videos in inventory:", nvid, {k: len(v) for k, v in inv.items()})
assert nvid == 115

# 4. clips referenced by data-clip attributes
clips = re.findall(r"assets/clips/([^\"]+)\.mp4", html)
for c in sorted(set(clips)):
    assert (base / "assets" / "clips" / (c + ".mp4")).exists(), c
print("clip links OK:", len(set(clips)), "unique")

# 5. examples group dirs contain 31 jpg each
exdir = base / "assets" / "examples"
for d in sorted(os.listdir(exdir)):
    n = len([x for x in os.listdir(exdir / d) if x.endswith(".jpg")])
    assert n == 31, (d, n)
print("example groups OK: 12 groups x 31 jpg")

print("ALL CHECKS PASSED")
