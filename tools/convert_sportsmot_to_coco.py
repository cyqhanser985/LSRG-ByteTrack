# -*- coding: utf-8 -*-
"""
Convert SportsMOT dataset from MOT format to COCO format (compatible with ByteTrack)
SportsMOT format: {split}/{seq}/gt/gt.txt  + {split}/{seq}/img1/{frame}.jpg
MOT format: [frame, id, x, y, w, h, conf, class, visibility]
"""

import os
import os.path as osp
import numpy as np
import json
import cv2

DATA_PATH = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'datasets', 'SportsMOT')
OUT_PATH = osp.join(DATA_PATH, 'annotations')
SPLITS = ['train', 'val', 'test']
HALF_VIDEO = False

if __name__ == '__main__':
    if not osp.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        data_path = osp.join(DATA_PATH, split)
        if not osp.exists(data_path):
            print(f"[SKIP] Directory not found: {data_path}")
            continue

        out_path = osp.join(OUT_PATH, '{}.json'.format(split))
        out = {
            'images': [],
            'annotations': [],
            'videos': [],
            'categories': [{'id': 1, 'name': 'person'},
                           {'id': 2, 'name': 'ball'},
                           {'id': 3, 'name': 'goalkeeper'},
                           {'id': 4, 'name': 'referee'}]
        }

        seqs = sorted([d for d in os.listdir(data_path) 
                      if osp.isdir(osp.join(data_path, d)) and not d.startswith('.')])
        
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0

        for seq in seqs:
            video_cnt += 1
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = osp.join(data_path, seq)
            img_dir = osp.join(seq_path, 'img1')

            # Check for images
            if not osp.exists(img_dir):
                print(f"  [WARNING] {seq}: img1 not found, skipping images")
                ann_path = osp.join(seq_path, 'gt', 'gt.txt')
                if not osp.exists(ann_path):
                    print(f"  [WARNING] {seq}: gt.txt also not found, skipping")
                    continue
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                if anns.ndim == 1:
                    anns = anns.reshape(1, -1)
                num_images = int(anns[:, 0].max())
                height, width = 720, 1280
            else:
                images = sorted(os.listdir(img_dir))
                num_images = len([img for img in images if img.endswith('.jpg')])
                
                first_img = osp.join(img_dir, images[0])
                if osp.exists(first_img):
                    img = cv2.imread(first_img)
                    if img is not None:
                        height, width = img.shape[:2]
                    else:
                        height, width = 720, 1280
                else:
                    height, width = 720, 1280

            image_range = [0, num_images - 1]

            # Add image metadata
            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                image_info = {
                    'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                    'id': image_cnt + i + 1,
                    'frame_id': i + 1 - image_range[0],
                    'prev_image_id': image_cnt + i if i > 0 else -1,
                    'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                    'video_id': video_cnt,
                    'height': height,
                    'width': width
                }
                out['images'].append(image_info)
            
            print(f'  {seq}: {num_images} images')
            
            # Process GT annotations
            if split != 'test':
                ann_path = osp.join(seq_path, 'gt', 'gt.txt')
                if not osp.exists(ann_path):
                    print(f'  [WARNING] {seq}: gt.txt not found, skipping annotations')
                    image_cnt += num_images
                    continue
                
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                if anns.ndim == 1:
                    anns = anns.reshape(1, -1)
                
                tid_curr = 0
                tid_last = -1

                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    
                    track_id = int(anns[i][1])
                    ann_cnt += 1
                    
                    # SportsMOT category_id at column 7 (0-indexed)
                    if anns.shape[1] > 7:
                        cat_id = int(anns[i][7])
                    else:
                        cat_id = 1
                    
                    if not (track_id == tid_last):
                        tid_curr += 1
                        tid_last = track_id
                    
                    ann = {
                        'id': ann_cnt,
                        'category_id': cat_id,
                        'image_id': image_cnt + frame_id,
                        'track_id': tid_curr,
                        'bbox': anns[i][2:6].tolist(),  # x, y, w, h
                        'conf': float(anns[i][6]) if anns.shape[1] > 6 else 1.0,
                        'iscrowd': 0,
                        'area': float(anns[i][4] * anns[i][5])
                    }
                    out['annotations'].append(ann)
            
            image_cnt += num_images

        total_images = len(out['images'])
        total_anns = len(out['annotations'])
        print(f'\n  [{split}] Total: {total_images} images, {total_anns} annotations, {video_cnt} videos')
        
        if total_images > 0:
            json.dump(out, open(out_path, 'w'))
            print(f'  Saved to: {out_path}')
        else:
            print(f'  [SKIP] No valid data, skipping save')

    print('\nSportsMOT COCO conversion complete!')
