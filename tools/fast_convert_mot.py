# -*- coding: utf-8 -*-
"""
Fast MOT-format to COCO converter for ByteTrack.
Reads only first image per sequence for dimensions (much faster than per-image imread).
"""
import os
import numpy as np
import json
import cv2
import argparse

def fast_convert(data_path, out_dir, name):
    """Convert MOT format to COCO JSON. 
    data_path should contain train/ and test/ with {seq}/img1/ and {seq}/gt/gt.txt
    """
    os.makedirs(out_dir, exist_ok=True)

    for split in ['train', 'test']:
        split_dir = os.path.join(data_path, split)
        if not os.path.exists(split_dir):
            print(f"  [SKIP] {split_dir} not found")
            continue

        out_path = os.path.join(out_dir, f'{split}.json')
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'pedestrian'}]}

        seqs = sorted([d for d in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')])

        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0

        for seq in seqs:
            video_cnt += 1
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(split_dir, seq)
            img_dir = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(seq_path, 'gt', 'gt.txt')

            if not os.path.exists(img_dir):
                print(f"    [SKIP] {seq}: no img1")
                continue

            images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            num_images = len(images)

            # Fast: only read first image for dimensions
            first_img_path = os.path.join(img_dir, images[0])
            img = cv2.imread(first_img_path)
            if img is not None:
                height, width = img.shape[:2]
            else:
                height, width = 1080, 1920

            # Add image entries
            for i in range(num_images):
                image_info = {
                    'file_name': f'{seq}/img1/{i+1:06d}.jpg',
                    'id': image_cnt + i + 1,
                    'frame_id': i + 1,
                    'prev_image_id': image_cnt + i if i > 0 else -1,
                    'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                    'video_id': video_cnt,
                    'height': height,
                    'width': width
                }
                out['images'].append(image_info)

            # Process annotations
            if os.path.exists(ann_path):
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                if anns.ndim == 1:
                    anns = anns.reshape(1, -1)

                tid_map = {}
                tid_curr = 0

                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id < 1 or frame_id > num_images:
                        continue

                    track_id = int(anns[i][1])

                    # Assign sequential track IDs
                    if track_id not in tid_map:
                        tid_curr += 1
                        tid_map[track_id] = tid_curr

                    ann_cnt += 1
                    ann = {
                        'id': ann_cnt,
                        'category_id': 1,
                        'image_id': image_cnt + frame_id,
                        'track_id': tid_map[track_id],
                        'bbox': anns[i][2:6].tolist(),
                        'conf': float(anns[i][6]) if anns.shape[1] > 6 else 1.0,
                        'iscrowd': 0,
                        'area': float(anns[i][4] * anns[i][5])
                    }
                    out['annotations'].append(ann)

            print(f"    {seq}: {num_images} images, {len(out['annotations']) - (ann_cnt - len([a for a in out['annotations'] if a['image_id'] > image_cnt and a['image_id'] <= image_cnt + num_images]))} annots")
            image_cnt += num_images

        print(f"  [{split}] {len(out['images'])} images, {len(out['annotations'])} annotations, {video_cnt} videos")
        
        if len(out['images']) > 0:
            with open(out_path, 'w') as f:
                json.dump(out, f)
            print(f"  Saved: {out_path}")

    # Also generate train_half / val_half
    train_file = os.path.join(out_dir, 'train.json')
    if os.path.exists(train_file):
        with open(train_file) as f:
            train_data = json.load(f)
        
        # Split each video in half
        half_train = {'images': [], 'annotations': [], 'videos': train_data['videos'],
                       'categories': train_data['categories']}
        half_val = {'images': [], 'annotations': [], 'videos': train_data['videos'],
                     'categories': train_data['categories']}

        # Build video_id -> image list
        vid_images = {}
        for img in train_data['images']:
            vid = img['video_id']
            if vid not in vid_images:
                vid_images[vid] = []
            vid_images[vid].append(img)

        # Keep track of image_id -> frame_id for annotations
        for vid, imgs in vid_images.items():
            n = len(imgs)
            mid = n // 2
            
            # First half -> train_half
            for img in imgs[:mid]:
                new_img = dict(img)
                new_img['frame_id'] = new_img['frame_id']  # keep original frame_id since imgs are contiguous
                half_train['images'].append(new_img)
            
            # Second half -> val_half
            for img in imgs[mid:]:
                new_img = dict(img)
                half_val['images'].append(new_img)

        # Map old image_id to half sets
        train_img_ids = {img['id'] for img in half_train['images']}
        val_img_ids = {img['id'] for img in half_val['images']}

        for ann in train_data['annotations']:
            if ann['image_id'] in train_img_ids:
                half_train['annotations'].append(ann)
            elif ann['image_id'] in val_img_ids:
                half_val['annotations'].append(ann)

        with open(os.path.join(out_dir, 'train_half.json'), 'w') as f:
            json.dump(half_train, f)
        with open(os.path.join(out_dir, 'val_half.json'), 'w') as f:
            json.dump(half_val, f)
        print(f"  train_half: {len(half_train['images'])} images, {len(half_train['annotations'])} annots")
        print(f"  val_half: {len(half_val['images'])} images, {len(half_val['annotations'])} annots")

    print(f"\n{name} conversion done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Convert both MOT17 and MOT20')
    args = parser.parse_args()

    if args.all:
        print("=" * 50)
        print("Converting MOT17...")
        fast_convert('datasets/MOT17/MOT17', 'datasets/MOT17/MOT17/annotations', 'MOT17')
        
        print("\n" + "=" * 50)
        print("Converting MOT20...")
        fast_convert('datasets/MOT20/MOT20', 'datasets/MOT20/MOT20/annotations', 'MOT20')
    else:
        print("Usage: python tools/fast_convert_mot.py --all")
