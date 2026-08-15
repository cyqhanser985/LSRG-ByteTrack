# -*- coding: utf-8 -*-
"""Convert COCO JSON to per-sequence MOT format gt.txt files."""

import json
from pathlib import Path
from collections import defaultdict

import os
BASE = Path(os.getcwd()) / 'datasets'


def load_mapping(mapping_file):
    """Load V_number -> original_name mapping."""
    name_to_v = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                name_to_v[parts[1]] = parts[0]
    return name_to_v


def coco_to_gt(dataset_dir, json_files, parse_seq_from_path=True, mapping=None):
    """
    Convert COCO JSON annotations to per-sequence gt.txt files.
    
    parse_seq_from_path: if True, extract seq name from file_name path (e.g. V001/img1/000001.jpg -> V001)
                        if False, use video_id with mapping
    """
    for json_file in json_files:
        if not json_file.exists():
            print(f"  Not found: {json_file}")
            continue

        print(f"  Processing: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build image lookup: image_id -> {frame_id, file_name, video_id}
        image_map = {}
        for img in data['images']:
            image_map[img['id']] = img

        # Group annotations by sequence
        seq_anns = defaultdict(list)
        for ann in data['annotations']:
            img = image_map.get(ann['image_id'])
            if not img:
                continue

            fn = img['file_name']
            raw_seq = fn.split('/')[0]
            if parse_seq_from_path:
                # BFT: path already has Vxxx -> use directly
                seq_name = raw_seq
            else:
                # MOT17: path has MOT17-02-DPM -> map to Vxxx
                seq_name = mapping.get(raw_seq, raw_seq)

            seq_anns[seq_name].append({
                'frame': img.get('frame_id', 1),
                'track_id': ann.get('track_id', -1),
                'bbox': ann.get('bbox', [0, 0, 0, 0]),
                'conf': ann.get('conf', -1),
            })

        # Write gt.txt for each sequence
        for seq_name, anns in seq_anns.items():
            gt_dir = dataset_dir / seq_name / 'gt'
            gt_dir.mkdir(parents=True, exist_ok=True)
            gt_file = gt_dir / 'gt.txt'

            with open(gt_file, 'w', encoding='utf-8') as f:
                for ann in sorted(anns, key=lambda x: x['frame']):
                    x, y, w, h = ann['bbox']
                    line = f"{ann['frame']},{ann['track_id']},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{ann['conf']},-1,-1\n"
                    f.write(line)

            print(f"    {seq_name}/gt/gt.txt: {len(anns)} annotations")


def process_bft():
    print("=" * 60)
    print("BFT: converting COCO JSON -> per-sequence gt.txt")
    print("=" * 60)

    dataset_dir = BASE / 'BFT'
    anno_dir = dataset_dir / 'annotations_coco'

    json_files = sorted(anno_dir.glob('*.json'))
    coco_to_gt(dataset_dir, json_files, parse_seq_from_path=True)

    print("BFT done!\n")


def process_mot17():
    print("=" * 60)
    print("MOT17: converting COCO JSON -> per-sequence gt.txt")
    print("=" * 60)

    dataset_dir = BASE / 'MOT17'
    anno_dir = dataset_dir / 'annotations'

    # Load mapping: original name -> V number
    mapping = load_mapping(dataset_dir / 'mapping.txt')
    if not mapping:
        print("  No mapping found, skipping")
        return

    json_files = sorted(anno_dir.glob('*.json'))
    coco_to_gt(dataset_dir, json_files, parse_seq_from_path=False, mapping=mapping)

    # Update COCO JSON paths to V format
    update_mot17_coco_paths(dataset_dir, mapping)

    print("MOT17 done!\n")


def update_mot17_coco_paths(dataset_dir, mapping):
    """Update COCO JSON file_name from MOT17-02-DPM/img1/... to V001/img1/..."""
    anno_dir = dataset_dir / 'annotations'

    for json_file in sorted(anno_dir.glob('*.json')):
        print(f"  Updating paths in: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        updated = 0
        for img in data.get('images', []):
            fn = img.get('file_name', '')
            if '/' in fn:
                seq_name = fn.split('/')[0]
                if seq_name in mapping:
                    rest = fn[len(seq_name):]
                    img['file_name'] = mapping[seq_name] + rest
                    updated += 1

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"    Updated {updated} paths")


if __name__ == '__main__':
    process_bft()
    process_mot17()
    print("All done!")
