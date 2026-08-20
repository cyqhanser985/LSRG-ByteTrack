# -*- coding: utf-8 -*-
"""eval_confusion_matrix.py — 计算检测结果在不同数据集上的混淆矩阵（TP, FP, FN, TN）

Part of LSRG-ByteTrack Research Workspace.

This script evaluates risk detection performance on MOT17, MOT20, and SportsMOT
at three critical working thresholds (TPR=90%, TPR=95%, and optimal F1-score).
We use the baseline Max aggregation operator on the calibrated risk vector [r_weak, r_comp, r_swap]
to compute the risk score for each event (positive) and C-group normal detection (negative).
"""

import os
import csv
import numpy as np

# Resolve path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TAXONOMY_DIR = os.path.join(ROOT, "research", "taxonomy")

def log(msg):
    print(f"[eval_confusion_matrix] {msg}")

def main():
    events_npz_path = os.path.join(TAXONOMY_DIR, "risk_features_events.npz")
    negatives_npz_path = os.path.join(TAXONOMY_DIR, "risk_features_negatives.npz")

    if not os.path.exists(events_npz_path) or not os.path.exists(negatives_npz_path):
        log(f"Error: Missing input NPZ files in {TAXONOMY_DIR}.")
        log("Please run risk_features.py first.")
        return

    # 1. Load data
    pos_data = np.load(events_npz_path)
    neg_data = np.load(negatives_npz_path)

    pos_risk = pos_data["risk_matrix"]       # [4713, 3]
    pos_ds = pos_data["dataset"]

    neg_risk = neg_data["risk_matrix"]       # [1647180, 3]
    neg_seq_datasets = neg_data["seq_datasets"]
    neg_seq_starts = neg_data["seq_starts"]
    neg_seq_ends = neg_data["seq_ends"]

    n_events = len(pos_risk)
    n_negs = len(neg_risk)
    log(f"Loaded {n_events} positive events and {n_negs} negative detections.")

    # 2. Compute risk scores using Max operator: R = max(r_weak, r_comp, r_swap)
    pos_scores = np.max(pos_risk, axis=1)
    neg_scores = np.max(neg_risk, axis=1)

    # Resolve negative dataset label per detection
    neg_ds = np.empty(n_negs, dtype="<U10")
    for ds, st, en in zip(neg_seq_datasets, neg_seq_starts, neg_seq_ends):
        neg_ds[int(st):int(en)] = str(ds)

    datasets = ["MOT17", "MOT20", "SportsMOT"]
    rows = []

    # 3. Compute Confusion Matrix per dataset
    for ds in datasets:
        ds_pos_scores = pos_scores[pos_ds == ds]
        ds_neg_scores = neg_scores[neg_ds == ds]

        P = len(ds_pos_scores)
        N = len(ds_neg_scores)
        log(f"Dataset {ds}: Positives (ID Switch) = {P}, Negatives = {N}")

        # Thresholds to evaluate: TPR=90%, TPR=95%, and Optimal F1
        # Find threshold corresponding to TPR = 90% and 95%
        # TPR = TP / P => TP = P * TPR
        sorted_pos = np.sort(ds_pos_scores)
        thresh_tpr90 = sorted_pos[int(np.floor(P * (1.0 - 0.90)))]
        thresh_tpr95 = sorted_pos[int(np.floor(P * (1.0 - 0.95)))]

        # Find Optimal F1 threshold
        # Combine predictions to search threshold
        all_scores = np.concatenate([ds_pos_scores, ds_neg_scores])
        all_labels = np.concatenate([np.ones(P), np.zeros(N)])
        
        # Grid search over candidate thresholds to find max F1
        best_f1 = -1.0
        best_thresh = 0.5
        # Sample candidates from positives to save memory
        candidates = np.percentile(ds_pos_scores, np.linspace(1, 99, 100))
        for c in candidates:
            preds = (all_scores >= c)
            tp = np.sum((preds == 1) & (all_labels == 1))
            fp = np.sum((preds == 1) & (all_labels == 0))
            fn = np.sum((preds == 0) & (all_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = c

        thresholds = [
            ("TPR=90%", thresh_tpr90),
            ("TPR=95%", thresh_tpr95),
            ("Optimal F1", best_thresh)
        ]

        for name, thresh in thresholds:
            # Predictions
            pos_pred = (ds_pos_scores >= thresh)
            neg_pred = (ds_neg_scores >= thresh)

            TP = int(np.sum(pos_pred))
            FN = int(np.sum(~pos_pred))
            FP = int(np.sum(neg_pred))
            TN = int(np.sum(~neg_pred))

            # Quality metrics
            tpr = TP / P if P > 0 else 0
            fpr = FP / N if N > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0

            rows.append({
                "Dataset": ds,
                "Operating Point": name,
                "Threshold": f"{thresh:.6f}",
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "TPR (Recall)": f"{tpr*100:.2f}%",
                "FPR": f"{fpr*100:.2f}%",
                "Precision": f"{precision*100:.4f}%",
                "F1-Score": f"{f1:.4f}"
            })

            log(f"  {name} Thresh={thresh:.4f} -> TP={TP}, FP={FP}, FN={FN}, TN={TN}, FPR={fpr*100:.2f}%")

    # 4. Save results to CSV
    csv_path = os.path.join(TAXONOMY_DIR, "confusion_matrix_report.csv")
    fields = ["Dataset", "Operating Point", "Threshold", "TP", "FP", "FN", "TN", "TPR (Recall)", "FPR", "Precision", "F1-Score"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Confusion matrix report successfully written to {csv_path}")

if __name__ == "__main__":
    main()
