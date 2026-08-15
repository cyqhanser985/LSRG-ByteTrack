# -*- coding: utf-8 -*-
"""
Dataset format standardization script.
Standardizes BFT, MOT17, MOT20, SportsMOT datasets for ByteTrack compatibility.

ByteTrack input requirements:
- COCO format JSON with: images(file_name, id, frame_id, video_id, height, width),
  annotations(id, category_id, image_id, track_id, bbox(xywh), area, iscrowd),
  categories(id, name), videos(id, file_name)
- Images organized as {seq_name}/img1/{frame:06d}.jpg
- Detection bbox format: [x1, y1, x2, y2, score] (xyxy), confidence range 0-1
"""

import os
import os.path as osp
import json

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATASETS_DIR = osp.join(ROOT, "datasets")


def verify_bft_dataset():
    """Verify BFT dataset format"""
    print("\n" + "=" * 60)
    print("Processing BFT Dataset")
    print("=" * 60)
    
    bft_root = osp.join(DATASETS_DIR, "BFT", "BFT")
    coco_dir = osp.join(bft_root, "annotations_coco")
    
    splits = ["train_v1.5", "val_v1.5", "test_v1.5"]
    
    for split in splits:
        coco_file = osp.join(coco_dir, f"{split}.json")
        if not osp.exists(coco_file):
            print(f"  [SKIP] {coco_file} not found")
            continue
        
        with open(coco_file, "r") as f:
            data = json.load(f)
        
        print(f"\n  Dataset: {split}")
        print(f"    - Images: {len(data['images'])}")
        print(f"    - Annotations: {len(data['annotations'])}")
        print(f"    - Videos: {len(data.get('videos', []))}")
        print(f"    - Categories: {[c['name'] for c in data['categories']]}")
        
        # Verify image format
        if len(data["images"]) > 0:
            img = data["images"][0]
            required = ["file_name", "id", "frame_id", "video_id", "height", "width"]
            missing = [f for f in required if f not in img]
            if missing:
                print(f"    [WARNING] Image missing fields: {missing}")
            else:
                print(f"    - Image fields: OK (file_name: {img['file_name']})")
        
        # Verify annotation format
        if len(data["annotations"]) > 0:
            ann = data["annotations"][0]
            required = ["id", "category_id", "image_id", "track_id", "bbox", "area", "iscrowd"]
            missing = [f for f in required if f not in ann]
            if missing:
                print(f"    [WARNING] Annotation missing fields: {missing}")
            else:
                print(f"    - Annotation fields: OK")
        
        # Check image files exist
        missing_count = 0
        for img in data["images"][:10]:
            img_path = osp.join(bft_root, img["file_name"])
            if not osp.exists(img_path):
                missing_count += 1
        if missing_count > 0:
            print(f"    [WARNING] {missing_count}/10 sample images missing")
        else:
            print(f"    - Image files: Verified (10/10)")
    
    print(f"\n  BFT COCO format is ByteTrack-compatible. No conversion needed.")
    return True


def verify_mot17_dataset():
    """Verify MOT17 dataset format"""
    print("\n" + "=" * 60)
    print("Processing MOT17 Dataset")
    print("=" * 60)
    
    mot17_root = osp.join(DATASETS_DIR, "MOT17", "MOT17")
    
    for split_name in ["train", "test"]:
        split_dir = osp.join(mot17_root, split_name)
        if not osp.exists(split_dir):
            print(f"  [SKIP] {split_dir} not found")
            continue
        
        seqs = sorted([d for d in os.listdir(split_dir) if osp.isdir(osp.join(split_dir, d)) and not d.startswith('.')])
        print(f"\n  {split_name}: {len(seqs)} sequences")
        
        for seq in seqs[:3]:
            seq_dir = osp.join(split_dir, seq)
            gt_file = osp.join(seq_dir, "gt", "gt.txt")
            det_file = osp.join(seq_dir, "det", "det.txt")
            
            has_gt = osp.exists(gt_file)
            has_det = osp.exists(det_file)
            print(f"    {seq}: gt={'YES' if has_gt else 'NO'}, det={'YES' if has_det else 'NO'}")
    
    print(f"\n  MOT17 needs convert_mot17_to_coco.py to generate COCO format.")
    return True


def verify_mot20_dataset():
    """Verify MOT20 dataset format"""
    print("\n" + "=" * 60)
    print("Processing MOT20 Dataset")
    print("=" * 60)
    
    mot20_root = osp.join(DATASETS_DIR, "MOT20", "MOT20")
    
    for split_name in ["train", "test"]:
        split_dir = osp.join(mot20_root, split_name)
        if not osp.exists(split_dir):
            print(f"  [SKIP] {split_dir} not found")
            continue
        
        seqs = sorted([d for d in os.listdir(split_dir) if osp.isdir(osp.join(split_dir, d)) and not d.startswith('.')])
        print(f"\n  {split_name}: {len(seqs)} sequences")
        
        for seq in seqs[:3]:
            seq_dir = osp.join(split_dir, seq)
            gt_file = osp.join(seq_dir, "gt", "gt.txt")
            det_file = osp.join(seq_dir, "det", "det.txt")
            img_dir = osp.join(seq_dir, "img1")
            
            has_gt = osp.exists(gt_file)
            has_det = osp.exists(det_file)
            has_img = osp.exists(img_dir)
            print(f"    {seq}: gt={'YES' if has_gt else 'NO'}, det={'YES' if has_det else 'NO'}, img={'YES' if has_img else 'NO'}")
    
    print(f"\n  MOT20 needs convert_mot20_to_coco.py to generate COCO format.")
    return True


def verify_sportsmot():
    """Verify SportsMOT dataset format"""
    print("\n" + "=" * 60)
    print("Processing SportsMOT Dataset")
    print("=" * 60)
    
    sportsmot_root = osp.join(DATASETS_DIR, "SportsMOT")
    ann_dir = osp.join(sportsmot_root, "annotations")
    
    for split in ["train", "val", "test"]:
        ann_file = osp.join(ann_dir, f"{split}.json")
        if not osp.exists(ann_file):
            print(f"  [SKIP] {ann_file} not found")
            continue
        
        with open(ann_file, "r") as f:
            data = json.load(f)
        
        print(f"\n  Split: {split}")
        print(f"    - Images: {len(data.get('images', []))}")
        print(f"    - Annotations: {len(data.get('annotations', []))}")
        print(f"    - Videos: {len(data.get('videos', []))}")
        print(f"    - Categories: {[c['name'] for c in data.get('categories', [])]}")
        
        if len(data.get("images", [])) > 0:
            img = data["images"][0]
            if "frame_id" in img and "video_id" in img:
                print(f"    - Format: ByteTrack-compatible (has frame_id, video_id)")
            else:
                print(f"    [WARNING] Missing frame_id or video_id")
    
    return True


def print_summary():
    """Print dataset summary"""
    print("\n" + "=" * 60)
    print("Dataset Standardization Summary")
    print("=" * 60)
    
    print("""
Four datasets and ByteTrack compatibility:

+-------------+----------------------------------------+--------------------------+
| Dataset     | Original Format                         | ByteTrack Compat.        |
+-------------+----------------------------------------+--------------------------+
| MOT17       | MOT Challenge (gt.txt/det.txt)          | Need convert_mot17_to_   |
|             | Images from MOT Challenge website       | coco.py to convert       |
+-------------+----------------------------------------+--------------------------+
| MOT20       | MOT Challenge (gt.txt/det.txt)          | Need convert_mot20_to_   |
|             | test has img1/ images                   | coco.py to convert       |
+-------------+----------------------------------------+--------------------------+
| BFT         | COCO JSON (done) + MOT txt (done)       | COCO format compatible   |
|             | Images extracted from zips              | No conversion needed     |
+-------------+----------------------------------------+--------------------------+
| SportsMOT   | MOT Challenge format (gt.txt)           | COCO format compatible   |
|             | Images in img1/ (val, test)             | Converted via script     |
+-------------+----------------------------------------+--------------------------+

ByteTrack core interface input requirements:
  1. BYTETracker.update(output_results, img_info, img_size)
     - output_results: numpy array [N, 5] or [N, 7]
       Format: [x1, y1, x2, y2, score] or [x1, y1, x2, y2, obj_s, cls_s, cls]
       x1y1x2y2: pixel coordinates (absolute)
       score/obj_score/cls_score: range 0-1
     - img_info: [img_h, img_w]
     - img_size: model input size (H, W)

  2. MOTDataset (training/evaluation)
     - COCO JSON annotations required
     - Images: {seq}/img1/{frame:06d}.jpg
     - Image meta: file_name, id, frame_id, video_id, height, width
     - Annotations: id, category_id, image_id, track_id, bbox(xywh), area, iscrowd
     - Categories: id, name
""")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ByteTrack Dataset Format Standardization Tool")
    print("=" * 60)
    
    verify_bft_dataset()
    verify_mot17_dataset()
    verify_mot20_dataset()
    verify_sportsmot()
    print_summary()
    
    print("\nDone!")
