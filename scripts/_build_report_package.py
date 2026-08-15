# -*- coding: utf-8 -*-
"""Build the self-contained HTML report package for the ID-switch analysis.

Steps:
  1. Create OUT directory skeleton under the project root.
  2. Convert the 372 example PNGs (12 groups x 31 frames) to JPEG (q88) into assets/examples/<group>/.
  3. Copy the 62 canvas event-window PNGs (V013/V019) into assets/canvas/.
  4. Assemble 14 event video clips (12 examples + 2 canvas windows) via OpenCV at 30 fps.
  5. Generate assets/data/frame_data.js (frame paths + event frame numbers for the HTML frame player).
  6. Convert switch_metrics/_tables.md -> assets/data/metric_tables.html (HTML fragment embedded by index.html).
  7. Convert the 3 events CSVs -> assets/data/events.js (searchable tables).
  8. Generate assets/data/video_inventory.js (all 115 track videos with sizes).
  9. Copy: 7 full-sequence videos, 2 canvas videos, 6 trajectory PNGs, 9 CSV files, reports to docs/.

Run with the project base python (PIL) + bytetrack env python (cv2) or any python with cv2+PIL.
"""
import csv
import glob
import json
import os
import shutil
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SRC)  # scripts/ 位于项目根下一层
ANALYSIS = os.path.join(PROJECT, 'YOLOX_outputs', 'analysis')
OUT = os.path.join(PROJECT, 'reports', 'ByteTrack_ID分析报告')

FPS = 30
JPEG_QUALITY = 88

EXAMPLE_GROUPS = [
    'MOT17_switch_V013_f000345', 'MOT17_switch_V013_f000542',
    'MOT17_reuse_V019_f000431', 'MOT17_reuse_V013_f000564',
    'MOT20_switch_V001_f000314', 'MOT20_switch_V002_f001495',
    'MOT20_reuse_V004_f002052', 'MOT20_reuse_V004_f002172',
    'SportsMOT_switch_V019_f000281', 'SportsMOT_switch_V040_f000709',
    'SportsMOT_reuse_V087_f000332', 'SportsMOT_reuse_V052_f000017',
]
CANVAS_WINDOWS = [('V013', 330, 360), ('V019', 266, 296)]

FULL_VIDEOS = [  # (src_dir, src_name, dst_name)
    ('mot17_v001_full', 'V001.mp4', 'MOT17_V001.mp4'),
    ('mot17_v001_full', 'V013.mp4', 'MOT17_V013.mp4'),
    ('mot17_v001_full', 'V019.mp4', 'MOT17_V019.mp4'),
    ('mot20_v001_full', 'V001.mp4', 'MOT20_V001.mp4'),
    ('sportsmot_v001_full', 'V040.mp4', 'SportsMOT_V040.mp4'),
    ('sportsmot_v001_full', 'V052.mp4', 'SportsMOT_V052.mp4'),
    ('sportsmot_v001_full', 'V087.mp4', 'SportsMOT_V087.mp4'),
]

TRAJECTORY_FILES = [
    'MOT17_switch_V001_f000452.png', 'MOT17_reuse_V019_f000431.png',
    'MOT20_switch_V002_f002246.png', 'MOT20_reuse_V004_f002052.png',
    'SportsMOT_switch_V019_f000281.png', 'SportsMOT_reuse_V046_f000546.png',
]

CSV_FILES = [  # (subdir, name)
    ('', 'MOT17_events.csv'), ('', 'MOT20_events.csv'), ('', 'SportsMOT_events.csv'),
    ('', 'MOT17_events_summary.csv'), ('', 'MOT20_events_summary.csv'), ('', 'SportsMOT_events_summary.csv'),
    ('switch_metrics', 'MOT17_events_metrics.csv'), ('switch_metrics', 'MOT20_events_metrics.csv'),
    ('switch_metrics', 'SportsMOT_events_metrics.csv'),
]

DOC_FILES = [
    'id_switch_report.md', 'id_switch_report.pdf', 'canvas_report.md',
    'switch_metrics_report.md', 'reproduction_results.md',
]


def mkdirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def convert_examples():
    """Step 2: examples PNG -> JPEG (grouped)."""
    src_dir = os.path.join(ANALYSIS, 'examples')
    out_root = os.path.join(OUT, 'assets', 'examples')
    mkdirs(out_root)
    total = 0
    for group in EXAMPLE_GROUPS:
        d = os.path.join(out_root, group)
        mkdirs(d)
        files = sorted(glob.glob(os.path.join(src_dir, group + '_f*.png')))
        assert len(files) == 31, (group, len(files))
        for f in files:
            base = 'f%06d.jpg' % int(os.path.basename(f)[-10:-4])
            out = os.path.join(d, base)
            if not os.path.exists(out):
                try:
                    from PIL import Image
                    Image.open(f).convert('RGB').save(
                        out, 'JPEG', quality=JPEG_QUALITY, optimize=True)
                except ImportError:
                    import cv2
                    cv2.imwrite(out, cv2.imread(f), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            total += 1
    print('examples jpeg:', total)


def copy_canvas_pngs():
    """Step 3: canvas event-window PNGs."""
    src_dir = os.path.join(ANALYSIS, 'canvas', 'examples')
    out_root = os.path.join(OUT, 'assets', 'canvas')
    mkdirs(out_root)
    total = 0
    for seq, lo, hi in CANVAS_WINDOWS:
        d = os.path.join(out_root, seq)
        mkdirs(d)
        for f in sorted(glob.glob(os.path.join(src_dir, seq, 'f*.png'))):
            num = int(os.path.basename(f)[1:7])
            if lo <= num <= hi:
                shutil.copy2(f, os.path.join(d, os.path.basename(f)))
                total += 1
    print('canvas png:', total)


def imread_unicode(path):
    """cv2.imread cannot handle non-ASCII paths on Windows."""
    import cv2
    import numpy as np
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def make_clips():
    """Step 4: assemble 14 mp4 clips at 30 fps from the original PNGs."""
    import cv2
    out_dir = os.path.join(OUT, 'assets', 'clips')
    mkdirs(out_dir)
    src_ex = os.path.join(ANALYSIS, 'examples')
    src_canvas = os.path.join(ANALYSIS, 'canvas', 'examples')
    jobs = []
    for group in EXAMPLE_GROUPS:
        frames = sorted(glob.glob(os.path.join(src_ex, group + '_f*.png')))
        assert len(frames) == 31
        jobs.append((os.path.join(out_dir, group + '.mp4'), frames))
    for seq, lo, hi in CANVAS_WINDOWS:
        frames = sorted(
            f for f in glob.glob(os.path.join(src_canvas, seq, 'f*.png'))
            if lo <= int(os.path.basename(f)[1:7]) <= hi)
        assert len(frames) == 31
        jobs.append((os.path.join(out_dir, 'canvas_%s_f%04d-%04d.mp4' % (seq, lo, hi)), frames))
    for out_path, frames in jobs:
        first = imread_unicode(frames[0])
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
        for f in frames:
            vw.write(imread_unicode(f))
        vw.release()
        print('clip:', os.path.basename(out_path), w, 'x', h, len(frames), 'frames')


def make_frame_data():
    """Step 5: frame paths + event numbers for the HTML player."""
    data = {}
    src_ex = os.path.join(ANALYSIS, 'examples')
    for group in EXAMPLE_GROUPS:
        files = sorted(glob.glob(os.path.join(src_ex, group + '_f*.png')))
        nums = [int(os.path.basename(f)[-10:-4]) for f in files]
        event = int(group[-6:])
        rel = ['assets/examples/%s/f%06d.jpg' % (group, n) for n in nums]
        data[group] = {'event': event, 'frames': rel, 'nums': nums}
    src_canvas = os.path.join(ANALYSIS, 'canvas', 'examples')
    for seq, lo, hi in CANVAS_WINDOWS:
        files = sorted(f for f in glob.glob(os.path.join(src_canvas, seq, 'f*.png'))
                       if lo <= int(os.path.basename(f)[1:7]) <= hi)
        nums = [int(os.path.basename(f)[1:7]) for f in files]
        event = nums.index(int(lo + (hi - lo) / 2))  # window middle
        event = nums[int(len(nums) / 2)]
        rel = ['assets/canvas/%s/f%06d.png' % (seq, n) for n in nums]
        data['canvas_%s' % seq] = {'event': event, 'frames': rel, 'nums': nums}
    with open(os.path.join(OUT, 'assets', 'data', 'frame_data.js'), 'w', encoding='utf-8') as f:
        f.write('window.FRAME_DATA = %s;\n' % json.dumps(data, ensure_ascii=False))
    print('frame_data keys:', list(data.keys()))


def md_table_to_html(lines, start):
    """Convert a markdown pipe table into an HTML table."""
    i = start
    header = [c.strip() for c in lines[i].strip().strip('|').split('|')]
    i += 2  # skip header + separator
    rows = []
    while i < len(lines) and lines[i].strip().startswith('|'):
        rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
        i += 1
    h = '<thead><tr>' + ''.join('<th>%s</th>' % c for c in header) + '</tr></thead>'
    body = '<tbody>' + ''.join(
        '<tr>' + ''.join('<td>%s</td>' % c for c in r) + '</tr>' for r in rows) + '</tbody>'
    return '<div class="tblwrap"><table>%s%s</table></div>' % (h, body), i


def make_metric_tables():
    """Step 6: _tables.md -> HTML fragment."""
    src = os.path.join(ANALYSIS, 'switch_metrics', '_tables.md')
    with open(src, encoding='utf-8') as f:
        lines = f.read().splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('#### '):
            out.append('<h5>%s</h5>' % line[5:].strip())
        elif line.startswith('## '):
            out.append('<h4>%s</h4>' % line[3:].strip())
        elif line.strip().startswith('|') and i + 1 < len(lines) and \
                set(lines[i + 1].strip().replace('|', '').replace(':', '').replace('-', '').replace(' ', '')) == set():
            tbl, i = md_table_to_html(lines, i)
            out.append(tbl)
            continue
        elif line.strip().startswith('gap = F'):
            out.append('<p class="note">%s</p>' % line.strip())
        i += 1
    with open(os.path.join(OUT, 'assets', 'data', 'metric_tables.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print('metric_tables.html:', len('\n'.join(out)), 'bytes')


def make_events_js():
    """Step 7: events CSVs -> JS arrays."""
    rows = {}
    for name in ['MOT17', 'MOT20', 'SportsMOT']:
        with open(os.path.join(ANALYSIS, name + '_events.csv'), encoding='utf-8') as f:
            rd = list(csv.reader(f))
        header, data = rd[0], rd[1:]
        rows[name] = [{'seq': r[0], 'frame': int(r[1]), 'type': r[2],
                       'gt_id_old': r[3], 'gt_id_new': r[4], 'track_id': r[5],
                       'note': r[6] if len(r) > 6 else ''} for r in data]
    with open(os.path.join(OUT, 'assets', 'data', 'events.js'), 'w', encoding='utf-8') as f:
        f.write('window.EVENTS = %s;\n' % json.dumps(rows, ensure_ascii=False))
    print('events rows:', {k: len(v) for k, v in rows.items()})


def make_video_inventory():
    """Step 8: all 115 track videos with sizes."""
    inv = {}
    for ds, sub in [('MOT17', 'mot17_v001_full'), ('MOT20', 'mot20_v001_full'),
                    ('SportsMOT', 'sportsmot_v001_full')]:
        d = os.path.join(PROJECT, 'YOLOX_outputs', sub, 'track_videos')
        entries = []
        for name in sorted(os.listdir(d)):
            if name.endswith('.mp4'):
                sz = os.path.getsize(os.path.join(d, name))
                entries.append({'seq': name[:-4], 'bytes': sz})
        inv[ds] = entries
    with open(os.path.join(OUT, 'assets', 'data', 'video_inventory.js'), 'w', encoding='utf-8') as f:
        f.write('window.VIDEO_INVENTORY = %s;\n' % json.dumps(inv, ensure_ascii=False))
    print('video inventory:', {k: len(v) for k, v in inv.items()})


def copy_media():
    """Step 9: videos, trajectory PNGs, CSVs, docs."""
    vid_dir = os.path.join(OUT, 'assets', 'videos')
    mkdirs(vid_dir)
    for sub, src_name, dst_name in FULL_VIDEOS:
        s = os.path.join(PROJECT, 'YOLOX_outputs', sub, 'track_videos', src_name)
        shutil.copy2(s, os.path.join(vid_dir, dst_name))
        print('video:', dst_name, '%.1f MB' % (os.path.getsize(s) / 1e6))
    cv_dir = os.path.join(OUT, 'assets', 'canvas_videos')
    mkdirs(cv_dir)
    for name in ['V013.mp4', 'V019.mp4']:
        s = os.path.join(ANALYSIS, 'canvas_videos', name)
        shutil.copy2(s, os.path.join(cv_dir, name))
    traj_dir = os.path.join(OUT, 'assets', 'trajectory')
    mkdirs(traj_dir)
    for name in TRAJECTORY_FILES:
        shutil.copy2(os.path.join(ANALYSIS, 'trajectory', name), os.path.join(traj_dir, name))
    tbl_dir = os.path.join(OUT, 'assets', 'tables')
    mkdirs(tbl_dir)
    for sub, name in CSV_FILES:
        shutil.copy2(os.path.join(ANALYSIS, sub, name), os.path.join(tbl_dir, name))
    shutil.copy2(os.path.join(ANALYSIS, 'switch_metrics', '_tables.md'),
                 os.path.join(tbl_dir, '_tables.md'))
    docs = os.path.join(OUT, 'docs')
    mkdirs(docs)
    for name in DOC_FILES:
        s = os.path.join(ANALYSIS, name)
        if not os.path.exists(s):
            s = os.path.join(PROJECT, name)
        shutil.copy2(s, os.path.join(docs, name))
    print('media copied')


def main():
    mkdirs(os.path.join(OUT, 'assets', 'data'))
    convert_examples()
    copy_canvas_pngs()
    make_clips()
    make_frame_data()
    make_metric_tables()
    make_events_js()
    make_video_inventory()
    copy_media()
    print('DONE ->', OUT)


if __name__ == '__main__':
    main()
