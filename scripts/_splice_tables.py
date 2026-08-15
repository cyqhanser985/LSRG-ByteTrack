# -*- coding: utf-8 -*-
"""Splice the generated metric_tables.html fragment into index.html at the METRIC_TABLES placeholder."""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "reports" / "ByteTrack_ID分析报告"
html_path = base / "index.html"
frag_path = base / "assets" / "data" / "metric_tables.html"

with io.open(html_path, encoding="utf-8") as f:
    html = f.read()
with io.open(frag_path, encoding="utf-8") as f:
    frag = f.read()

marker = "<!-- METRIC_TABLES -->"
assert marker in html, "placeholder missing"
html = html.replace(marker, frag)

with io.open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print("spliced:", len(frag), "bytes of tables ->", html_path)
