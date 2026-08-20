# -*- coding: utf-8 -*-
"""id_recovery_dino.py — DINO 视觉特征余弦相似度匹配与 ID 恢复评测

Part of LSRG-ByteTrack Research Workspace.

This script extracts query patches for ID switch events and candidate patches
from history tracks in the last 100 frames. It runs a pre-trained DINOv2 model
via PyTorch Hub, computes cosine similarities, and outputs recovery rates
and Hard Cases for VLM evaluation.
"""

import os
import csv
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TAXONOMY_DIR = os.path.join(ROOT, "research", "taxonomy")
DATA_DIR = os.path.join(ROOT, "research", "data")
SCRATCH_DIR = os.path.join(ROOT, "research", "scratch")
CROPS_DIR = os.path.join(SCRATCH_DIR, "crops")

os.makedirs(CROPS_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def log(msg):
    print(f"[id_recovery_dino] {msg}")

def compute_iou(box1, box2):
    """Compute IoU between box1 and box2 in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def load_gt_boxes(gt_path):
    """Load GT boxes per frame: {frame: {gt_id: [x, y, w, h]}}."""
    gt_by_frame = {}
    if not os.path.exists(gt_path):
        return gt_by_frame
    with open(gt_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            gt_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 else 1
            if conf == 0:
                continue
            if frame not in gt_by_frame:
                gt_by_frame[frame] = {}
            gt_by_frame[frame][gt_id] = [x, y, w, h]
    return gt_by_frame

def load_track_results(track_path):
    """Load tracking results: {frame: {track_id: [x, y, w, h]}} and historical index."""
    tracks_by_frame = {}
    if not os.path.exists(track_path):
        return tracks_by_frame
    with open(track_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            if frame not in tracks_by_frame:
                tracks_by_frame[frame] = {}
            tracks_by_frame[frame][track_id] = [x, y, w, h]
    return tracks_by_frame

# --------------------------------------------------------------------------
# Main DINO matching pipeline
# --------------------------------------------------------------------------
def main():
    # 1. Setup DINO model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    log("Loading pre-trained DINOv2 model (dinov2_vits14)...")
    try:
        # Load from PyTorch Hub
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dino = dino.to(device)
        dino.eval()
        log("DINOv2 model loaded successfully.")
    except Exception as e:
        log(f"Error loading model: {e}. Please ensure internet connection.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datasets = ["MOT17", "MOT20", "SportsMOT"]
    recovery_stats = []
    hard_cases = []

    for ds in datasets:
        events_path = os.path.join(DATA_DIR, f"{ds}_events.csv")
        if not os.path.exists(events_path):
            log(f"Events file not found: {events_path}")
            continue

        # Load events
        events = []
        with open(events_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["type"] == "switch":  # Target only ID switches
                    events.append(row)

        log(f"Dataset {ds}: Found {len(events)} ID Switch events.")
        
        # Load track folder path
        track_dir = os.path.join(ROOT, "YOLOX_outputs", f"{ds.lower()}_v001_full", "track_results")
        
        # Process each event
        tp_recovered = 0
        total_eval = 0

        for event in events:
            seq = event["seq"]
            frame = int(event["frame"])
            query_tid = int(event["track_id"])
            gt_new = int(event["gt_id_new"])
            
            # Load track files & gt files for this sequence
            seq_track_path = os.path.join(track_dir, f"{seq}.txt")
            seq_gt_path = os.path.join(ROOT, "datasets", ds, seq, "gt", "gt.txt")
            
            tracks = load_track_results(seq_track_path)
            gt_boxes = load_gt_boxes(seq_gt_path)
            
            # Ensure query box exists in tracks
            if frame not in tracks or query_tid not in tracks[frame]:
                continue
                
            query_box = tracks[frame][query_tid]
            
            # Lookback window [F-100, F-1] to gather candidate track_ids
            candidates = {}
            for f in range(max(1, frame - 100), frame):
                if f in tracks:
                    for tid, box in tracks[f].items():
                        if tid != query_tid:
                            candidates[tid] = (f, box)  # overwrite keeps latest appearance

            if not candidates:
                continue

            # We need to map candidate tracks to GT IDs for correctness check
            candidate_list = []
            for tid, (f_cand, b_cand) in candidates.items():
                # Find GT ID at f_cand
                gt_id_mapped = -1
                if f_cand in gt_boxes:
                    best_iou = 0.5
                    for g_id, g_box in gt_boxes[f_cand].items():
                        iou = compute_iou(b_cand, g_box)
                        if iou >= best_iou:
                            best_iou = iou
                            gt_id_mapped = g_id
                
                # We only keep candidates that mapped to a valid GT ID
                if gt_id_mapped != -1:
                    candidate_list.append({
                        "track_id": tid,
                        "frame": f_cand,
                        "box": b_cand,
                        "gt_id": gt_id_mapped
                    })

            if not candidate_list:
                continue

            total_eval += 1

            # Crop Query patch
            q_img_path = os.path.join(ROOT, "datasets", ds, seq, "img1", f"{frame:06d}.jpg")
            if not os.path.exists(q_img_path):
                continue
            
            q_img = cv2.imread(q_img_path)
            h_q, w_q = q_img.shape[:2]
            qx1, qy1, qw, qh = query_box
            qx1, qy1 = int(max(0, qx1)), int(max(0, qy1))
            qx2, qy2 = int(min(w_q, qx1 + qw)), int(min(h_q, qy1 + qh))
            
            query_crop = q_img[qy1:qy2, qx1:qx2]
            if query_crop.size == 0:
                continue
            
            # Prepare Query DINO feature
            q_pil = Image.fromarray(cv2.cvtColor(query_crop, cv2.COLOR_BGR2RGB))
            q_tensor = transform(q_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_feat = dino(q_tensor).squeeze().cpu().numpy()
            q_feat /= np.linalg.norm(q_feat)

            # Compute similarities for all candidates
            best_sim = -1.0
            best_cand = None
            
            candidate_crops_meta = []

            for cand in candidate_list:
                cf = cand["frame"]
                cb = cand["box"]
                c_img_path = os.path.join(ROOT, "datasets", ds, seq, "img1", f"{cf:06d}.jpg")
                if not os.path.exists(c_img_path):
                    continue
                
                c_img = cv2.imread(c_img_path)
                h_c, w_c = c_img.shape[:2]
                cx1, cy1, cw, ch = cb
                cx1, cy1 = int(max(0, cx1)), int(max(0, cy1))
                cx2, cy2 = int(min(w_c, cx1 + cw)), int(min(h_c, cy1 + ch))
                
                c_crop = c_img[cy1:cy2, cx1:cx2]
                if c_crop.size == 0:
                    continue

                # Prepare Candidate DINO feature
                c_pil = Image.fromarray(cv2.cvtColor(c_crop, cv2.COLOR_BGR2RGB))
                c_tensor = transform(c_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    c_feat = dino(c_tensor).squeeze().cpu().numpy()
                c_feat /= np.linalg.norm(c_feat)

                sim = float(np.dot(q_feat, c_feat))
                cand["similarity"] = sim
                candidate_crops_meta.append((cand, c_crop))

                if sim > best_sim:
                    best_sim = sim
                    best_cand = cand

            # Evaluate DINO choice
            is_correct = False
            if best_cand is not None and best_cand["gt_id"] == gt_new:
                tp_recovered += 1
                is_correct = True

            # If incorrect or low margin, flag as Hard Case for VLM
            # (Margin is diff between top1 and top2 similarities)
            sorted_cands = sorted(candidate_list, key=lambda x: x.get("similarity", -1), reverse=True)
            margin = 1.0
            if len(sorted_cands) > 1:
                margin = sorted_cands[0]["similarity"] - sorted_cands[1]["similarity"]

            if not is_correct or margin < 0.15:
                # Save metadata for VLM execution
                query_crop_path = os.path.join(CROPS_DIR, f"{ds}_{seq}_F{frame}_Q.jpg")
                cv2.imwrite(query_crop_path, query_crop)
                
                cand_paths = []
                for idx, (cand, crop) in enumerate(candidate_crops_meta):
                    c_path = os.path.join(CROPS_DIR, f"{ds}_{seq}_F{frame}_C{idx}_T{cand['track_id']}_G{cand['gt_id']}.jpg")
                    cv2.imwrite(c_path, crop)
                    cand_paths.append({
                        "track_id": cand["track_id"],
                        "gt_id": cand["gt_id"],
                        "crop_path": c_path,
                        "similarity": cand["similarity"]
                    })

                hard_cases.append({
                    "dataset": ds,
                    "seq": seq,
                    "frame": frame,
                    "query_track_id": query_tid,
                    "gt_correct_id": gt_new,
                    "query_crop_path": query_crop_path,
                    "candidates": cand_paths,
                    "dino_predicted_gt_id": best_cand["gt_id"] if best_cand else -1,
                    "dino_correct": int(is_correct),
                    "margin": margin
                })

        accuracy = tp_recovered / total_eval if total_eval > 0 else 0
        log(f"Dataset {ds}: DINO Recovery Accuracy = {accuracy*100:.2f}% ({tp_recovered}/{total_eval})")
        recovery_stats.append({
            "Dataset": ds,
            "Total_Evaluated": total_eval,
            "DINO_Recovered": tp_recovered,
            "DINO_Accuracy": f"{accuracy*100:.2f}%"
        })

    # Save summary stats
    summary_path = os.path.join(TAXONOMY_DIR, "id_recovery_dino_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Total_Evaluated", "DINO_Recovered", "DINO_Accuracy"])
        writer.writeheader()
        writer.writerows(recovery_stats)

    # Save Hard Cases metadata to NPZ for Task 3 VLM execution
    hard_cases_path = os.path.join(SCRATCH_DIR, "id_recovery_hard_cases.npz")
    np.savez_compressed(hard_cases_path, hard_cases=np.array(hard_cases, dtype=object))
    log(f"Saved DINO stats to {summary_path}")
    log(f"Saved {len(hard_cases)} Hard Cases to {hard_cases_path} for VLM evaluation.")

if __name__ == "__main__":
    main()
