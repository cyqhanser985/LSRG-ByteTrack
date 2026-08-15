# -*- coding: utf-8 -*-
"""
Dataset restructuring script
Convert MOT17 / MOT20 / BFT / SportsMOT to standard format:
    V001/
      img1/    <- video frames (000001.jpg format)
      gt/      <- ground truth files
"""

import os
import shutil
import json
import tarfile
from pathlib import Path
from collections import OrderedDict

# Use cwd relative path to avoid encoding issues
BASE = Path(os.getcwd()) / 'datasets'


def get_seq_dirs(src_dir, exclude_prefixes=('._', '.'), exclude_names=None):
    if exclude_names is None:
        exclude_names = []
    dirs = []
    for d in sorted(src_dir.iterdir()):
        if d.is_dir() and not any(d.name.startswith(p) for p in exclude_prefixes) and d.name not in exclude_names:
            dirs.append(d)
    return dirs


def save_mapping(mapping_file, mapping_dict):
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("# Mapping: V_number -> original_name\n")
        f.write("# Format: V_number\toriginal_name\n")
        for v_name, orig_name in mapping_dict.items():
            f.write(f"{v_name}\t{orig_name}\n")
    print(f"  Mapping saved: {mapping_file}")


def restructure_sequences(src_dir, target_dir, mapping, has_img1=False, copy_gt_from=None):
    target_dir.mkdir(parents=True, exist_ok=True)

    for v_name, orig_name in mapping.items():
        src_seq = src_dir / orig_name
        dst_seq = target_dir / v_name

        if dst_seq.exists():
            print(f"  Skip existing: {dst_seq}")
            continue

        dst_img1 = dst_seq / 'img1'
        dst_gt = dst_seq / 'gt'
        dst_img1.mkdir(parents=True, exist_ok=True)
        dst_gt.mkdir(parents=True, exist_ok=True)

        if has_img1:
            src_img1 = src_seq / 'img1'
            if src_img1.exists():
                for f in src_img1.iterdir():
                    if f.is_file():
                        shutil.move(str(f), str(dst_img1 / f.name))
        else:
            for f in src_seq.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(dst_img1 / f.name))

        for f in src_seq.iterdir():
            if f.is_file():
                shutil.move(str(f), str(dst_seq / f.name))

        if copy_gt_from:
            gt_src = copy_gt_from / orig_name / 'gt'
            if gt_src.exists():
                for f in gt_src.iterdir():
                    shutil.copy2(str(f), str(dst_gt / f.name))

    leftovers = [d for d in src_dir.iterdir() if d.is_dir() and not any(d.name.startswith(p) for p in ('._', '.'))]
    for d in leftovers:
        try:
            remaining = list(d.rglob('*'))
            if not remaining:
                d.rmdir()
        except OSError:
            pass


def restructure_bft():
    print("\n" + "=" * 60)
    print("1. Restructuring BFT dataset")
    print("=" * 60)

    src_base = BASE / 'BFT' / 'BFT'
    target_base = BASE / 'BFT'

    exclude = ['annotations_coco', 'annotations_mot']
    all_seqs = get_seq_dirs(src_base, exclude_names=exclude)

    print(f"  Found {len(all_seqs)} sequences")

    mapping = OrderedDict()
    for i, d in enumerate(all_seqs):
        v_name = f"V{i + 1:03d}"
        mapping[v_name] = d.name

    save_mapping(target_base / 'mapping.txt', mapping)

    restructure_sequences(src_base, target_base, mapping, has_img1=False)

    for anno_dir in exclude:
        src_anno = src_base / anno_dir
        dst_anno = target_base / anno_dir
        if src_anno.exists() and not dst_anno.exists():
            shutil.move(str(src_anno), str(dst_anno))
            print(f"  Moved annotations: {anno_dir}")

    if src_base.exists():
        try:
            remaining = list(src_base.rglob('*'))
            if not remaining or all(f.name.startswith('.') for f in remaining):
                shutil.rmtree(str(src_base))
                print(f"  Cleaned empty dir: {src_base}")
        except OSError:
            pass

    update_coco_paths(target_base)

    print("  BFT done!")


def update_coco_paths(target_base):
    mapping_file = target_base / 'mapping.txt'
    if not mapping_file.exists():
        return

    orig_to_v = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                orig_to_v[parts[1]] = parts[0]

    anno_dir = target_base / 'annotations_coco'
    if not anno_dir.exists():
        print("  No annotations_coco found, skipping COCO JSON update")
        return

    for json_file in anno_dir.glob('*.json'):
        print(f"  Updating COCO JSON: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        updated = 0
        for img in data.get('images', []):
            fn = img.get('file_name', '')
            if '/' in fn:
                seq_name, frame_name = fn.split('/', 1)
                if seq_name in orig_to_v:
                    img['file_name'] = f"{orig_to_v[seq_name]}/img1/{frame_name}"
                    updated += 1

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"    Updated {updated} paths")


def restructure_mot17():
    print("\n" + "=" * 60)
    print("2. Restructuring MOT17 dataset")
    print("=" * 60)

    src_base = BASE / 'MOT17' / 'MOT17' / 'train'
    if not src_base.exists():
        print(f"  Source not found: {src_base}, skipping")
        return

    target_base = BASE / 'MOT17'

    all_seqs = get_seq_dirs(src_base)
    print(f"  Found {len(all_seqs)} sequences")

    mapping = OrderedDict()
    for i, d in enumerate(all_seqs):
        v_name = f"V{i + 1:03d}"
        mapping[v_name] = d.name

    save_mapping(target_base / 'mapping.txt', mapping)

    restructure_sequences(src_base, target_base, mapping, has_img1=True)

    nested = BASE / 'MOT17' / 'MOT17'
    if nested.exists():
        try:
            remaining = list(nested.rglob('*'))
            if not remaining or all(f.name.startswith('.') for f in remaining):
                shutil.rmtree(str(nested))
                print(f"  Cleaned empty dir: {nested}")
        except OSError:
            pass

    print("  MOT17 done!")


def restructure_mot20():
    print("\n" + "=" * 60)
    print("3. Restructuring MOT20 dataset")
    print("=" * 60)

    src_base = BASE / 'MOT20' / 'MOT20' / 'train'
    if not src_base.exists():
        print(f"  Source not found: {src_base}, skipping")
        return

    target_base = BASE / 'MOT20'
    gt_base = BASE / 'MOT20' / 'MOT20Labels' / 'train'

    all_seqs = get_seq_dirs(src_base)
    print(f"  Found {len(all_seqs)} sequences")

    mapping = OrderedDict()
    for i, d in enumerate(all_seqs):
        v_name = f"V{i + 1:03d}"
        mapping[v_name] = d.name

    save_mapping(target_base / 'mapping.txt', mapping)

    restructure_sequences(src_base, target_base, mapping, has_img1=True, copy_gt_from=gt_base)

    for nested_name in ['MOT20', 'MOT20Labels']:
        nested = BASE / 'MOT20' / nested_name
        if nested.exists():
            try:
                remaining = list(nested.rglob('*'))
                if not remaining or all(f.name.startswith('.') for f in remaining):
                    shutil.rmtree(str(nested))
                    print(f"  Cleaned empty dir: {nested}")
            except OSError:
                pass

    print("  MOT20 done!")


def restructure_sportsmot():
    print("\n" + "=" * 60)
    print("4. Restructuring SportsMOT dataset")
    print("=" * 60)

    src_base = BASE / 'SportsMOT'
    target_base = BASE / 'SportsMOT'

    for tar_name in ['train.tar', 'val.tar', 'test.tar']:
        tar_path = src_base / tar_name
        if not tar_path.exists():
            continue

        extract_dir = src_base / tar_name.replace('.tar', '')
        if extract_dir.exists() and list(extract_dir.iterdir()):
            print(f"  Already extracted: {tar_name}")
            continue

        size_gb = tar_path.stat().st_size / 1e9
        print(f"  Extracting {tar_name} ({size_gb:.2f} GB)...")
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(str(tar_path), 'r') as tar:
            members = [m for m in tar.getmembers() if not m.name.split('/')[-1].startswith('._')]
            tar.extractall(path=str(extract_dir), members=members)
        print(f"  Extraction done: {tar_name}")

    v_counter = 1
    all_mapping = OrderedDict()

    for split in ['train', 'val', 'test']:
        split_dir = src_base / split
        if not split_dir.exists():
            continue

        seqs = get_seq_dirs(split_dir)
        print(f"  [{split}] Found {len(seqs)} sequences")

        for d in seqs:
            v_name = f"V{v_counter:03d}"
            all_mapping[v_name] = f"{split}/{d.name}"

            dst_seq = target_base / v_name
            dst_img1 = dst_seq / 'img1'
            dst_gt = dst_seq / 'gt'
            dst_img1.mkdir(parents=True, exist_ok=True)
            dst_gt.mkdir(parents=True, exist_ok=True)

            src_img1 = d / 'img1'
            if src_img1.exists():
                for f in src_img1.iterdir():
                    if f.is_file() and not f.name.startswith('._'):
                        shutil.move(str(f), str(dst_img1 / f.name))

            src_gt = d / 'gt'
            if src_gt.exists():
                for f in src_gt.iterdir():
                    if f.is_file() and not f.name.startswith('._'):
                        shutil.move(str(f), str(dst_gt / f.name))

            for f in d.iterdir():
                if f.is_file() and not f.name.startswith('._'):
                    shutil.move(str(f), str(dst_seq / f.name))

            v_counter += 1

        try:
            remaining = list(split_dir.rglob('*'))
            if not remaining or all(f.name.startswith('._') for f in remaining):
                shutil.rmtree(str(split_dir))
        except OSError:
            pass

    save_mapping(target_base / 'mapping.txt', all_mapping)
    print("  SportsMOT done!")


if __name__ == '__main__':
    import sys

    print(f"Working dir: {os.getcwd()}")
    print(f"BASE path: {BASE}")

    datasets_to_process = sys.argv[1:] if len(sys.argv) > 1 else ['BFT', 'MOT17', 'MOT20', 'SportsMOT']

    for ds in datasets_to_process:
        if ds == 'BFT':
            restructure_bft()
        elif ds == 'MOT17':
            restructure_mot17()
        elif ds == 'MOT20':
            restructure_mot20()
        elif ds == 'SportsMOT':
            restructure_sportsmot()
        else:
            print(f"Unknown dataset: {ds}")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
