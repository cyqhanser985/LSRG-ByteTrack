# risk_features.py
# ==============================================================================
# Unified Gate-Free Causal Risk Feature Extraction and ECDF Calibration
#
# Part of LSRG-ByteTrack Research Workspace.
#
# This script extracts 3 online causal risk features for all detection boxes
# (negatives/C-group) and all IDS switch events without any pre-qualification
# gates (i.e. completely removing 'top1>=0.2 and top2>=0.2'):
#
#   1. f_weak = 1.0 - top1       (Under-matching risk; range [0, 1])
#   2. f_comp = top2             (Competition ambiguity risk; range [0, 1])
#                                (0.0 when m < 2, i.e. no competing track)
#   3. f_swap = \Delta C_swap    (2x2 assignment swap instability; variant B)
#                                (-2.0 when m < 2, n < 2, or no D2 exists)
#
# For events without an F-frame detection box (no_box == 1), the deterministic
# causal fallback vector [1.0, 0.0, -2.0] is assigned.
#
# All features are monotonically calibrated into [0, 1] risk components:
#   r_weak, r_comp, r_swap \in [0.0, 1.0]
# based on the Empirical Cumulative Distribution Function (ECDF) of the
# ~1.65 million negative sample detections (C-group normal frames).
#
# Outputs:
#   - research/taxonomy/risk_features_events.npy      (Core [N_events, 3] tensor)
#   - research/taxonomy/risk_features_events.npz      (Full package with metadata)
#   - research/taxonomy/risk_features_events.csv      (Human-readable event CSV)
#   - research/taxonomy/risk_ecdf_calibrator.npz      (Frozen ECDF calibrator)
#
# Pure stdlib + numpy + scipy; ASCII comments only.
# Run in-place with the bytetrack conda interpreter.
# ==============================================================================

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as A

DATASETS = ["MOT17", "MOT20", "SportsMOT"]
EXPNS = {
    "MOT17": "mot17_v001_full",
    "MOT20": "mot20_v001_full",
    "SportsMOT": "sportsmot_v001_full"
}

ROOT = A._repo_root()
DATA_DIR = os.path.join(ROOT, "research", "data")
TAXONOMY_DIR = os.path.join(ROOT, "research", "taxonomy")

_t0 = time.time()


def log(msg):
    print("[+%6.1fs] %s" % (time.time() - _t0, msg))


# --------------------------------------------------------------------------
# 2x2 Swap Instability Feature (Variant B)
# --------------------------------------------------------------------------
def compute_swap_delta_cost(iou):
    """Vectorized 2x2 assignment swap instability on dets x preds IoU matrix.
    
    For detection i: T1 = best track, T2 = second-best track.
    D2 = T2's best matching detection (i != i, or second-best if i is best).
    Delta C_swap = (c12 + c21) - (c11 + c22), where:
      c11 = top1 = IoU(D1, T1)
      c12 = top2 = IoU(D1, T2)
      c21 = IoU(D2, T1)
      c22 = IoU(D2, T2)
    Returns float array (N,) with NaN where m < 2 or no valid D2 exists.
    """
    n, m = iou.shape
    if n == 0 or m < 2:
        return np.full(n, np.nan, dtype=np.float64)
    rows = np.arange(n)
    idx = np.argpartition(iou, -2, axis=1)[:, -2:]
    t2, t1 = idx[:, 0], idx[:, 1]
    sw = iou[rows, t2] > iou[rows, t1]
    t1x = np.where(sw, t2, t1)          # best track per detection
    t2x = np.where(sw, t1, t2)          # second-best track per detection

    if n >= 2:
        kdx = np.argpartition(iou.T, -2, axis=1)[:, -2:]
        k2, k1 = kdx[:, 0], kdx[:, 1]
        trows = np.arange(m)
        sk = iou.T[trows, k2] > iou.T[trows, k1]
        d1 = np.where(sk, k2, k1)       # best det per track
        d2 = np.where(sk, k1, k2)       # second-best det per track
    else:
        d1 = np.zeros(m, dtype=int)
        d2 = np.full(m, -1, dtype=int)

    # Variant B: D2 = T2's own detection
    own2 = d1[t2x]
    d2b = np.where(own2 != rows, own2, d2[t2x])
    rb = np.maximum(d2b, 0)
    c21b = np.where(d2b >= 0, iou[rb, t1x], np.nan)
    c22b = np.where(d2b >= 0, iou[rb, t2x], np.nan)
    dcb = (iou[rows, t2x] + c21b) - (iou[rows, t1x] + c22b)
    return dcb


# --------------------------------------------------------------------------
# Collection Context
# --------------------------------------------------------------------------
class FeatureExtractorContext(object):
    def __init__(self, ds):
        self.ds = ds
        self.events = A.load_events(ds)
        self.ev_frames = defaultdict(set)
        self.ev_by_frame = defaultdict(list)
        for r, cls in self.events:
            s, F = r["seq"], int(r["frame"])
            self.ev_frames[s].add(F)
            self.ev_by_frame[(s, F)].append((int(r["track_id"]), cls, r))
        self.event_records = []
        self.neg_feature_arrays = []
        self.n_neg_detections = 0
        self.n_no_box_events = 0


def extract_sequence_features(ctx, seq, frames):
    """Extract 3-dimensional causal features without qualification gating."""
    ds = ctx.ds
    evf = ctx.ev_frames.get(seq, set())

    for F in sorted(frames):
        dets = frames[F]
        det_arr, det_ids = A.boxes_array(dets)
        n_det = len(det_ids)
        in_ev = F in evf
        cur_evs = ctx.ev_by_frame.get((seq, F))
        prev = frames.get(F - 1)
        prev2 = frames.get(F - 2)

        if prev is None:
            # Frame without active tracks (F-1 has no outputs)
            # Default causal fallback: f_weak=1.0, f_comp=0.0, f_swap=-2.0
            fallback_feats = np.zeros((n_det, 3), dtype=np.float64)
            fallback_feats[:, 0] = 1.0
            fallback_feats[:, 1] = 0.0
            fallback_feats[:, 2] = -2.0

            if not in_ev:
                if n_det > 0:
                    ctx.neg_feature_arrays.append(fallback_feats)
                    ctx.n_neg_detections += n_det
            else:
                for tid, cls, r in cur_evs:
                    ctx.n_no_box_events += 1
                    ctx.event_records.append({
                        "dataset": ds, "seq": seq, "frame": F, "class": cls,
                        "track_id": tid, "gt_id_new": r["gt_id_new"],
                        "no_box": 1,
                        "f_weak": 1.0, "f_comp": 0.0, "f_swap": -2.0
                    })
            continue

        prev_arr, prev_ids = A.boxes_array(prev)
        pred_boxes = {
            tid: A.extrapolate_box(b, prev2.get(tid) if prev2 else None)
            for tid, b in prev.items()
        }
        pred_arr, _ = A.boxes_array(pred_boxes)
        iou = A.iou_matrix(det_arr, pred_arr)

        top1v, top2v, _ = A._top1_top2_margin(iou)
        dcb = compute_swap_delta_cost(iou)

        # 3 raw causal risk features:
        # 1. f_weak = 1.0 - top1 (in [0, 1])
        f_weak = 1.0 - top1v
        # 2. f_comp = top2 (in [0, 1]; 0.0 if m < 2)
        f_comp = top2v
        # 3. f_swap = dcb (swap variant B; default -2.0 if undefined)
        f_swap = np.where(np.isnan(dcb), -2.0, dcb)

        det_feats = np.column_stack([f_weak, f_comp, f_swap])

        # Negative samples (non-event frames)
        if not in_ev:
            if n_det > 0:
                ctx.neg_feature_arrays.append(det_feats)
                ctx.n_neg_detections += n_det

        # Positive event samples
        if cur_evs:
            idx = {t: i for i, t in enumerate(det_ids)}
            for tid, cls, r in cur_evs:
                i = idx.get(tid)
                if i is None:
                    # Receiving tracker has no F-frame output box
                    ctx.n_no_box_events += 1
                    ctx.event_records.append({
                        "dataset": ds, "seq": seq, "frame": F, "class": cls,
                        "track_id": tid, "gt_id_new": r["gt_id_new"],
                        "no_box": 1,
                        "f_weak": 1.0, "f_comp": 0.0, "f_swap": -2.0
                    })
                else:
                    ctx.event_records.append({
                        "dataset": ds, "seq": seq, "frame": F, "class": cls,
                        "track_id": tid, "gt_id_new": r["gt_id_new"],
                        "no_box": 0,
                        "f_weak": float(f_weak[i]),
                        "f_comp": float(f_comp[i]),
                        "f_swap": float(f_swap[i])
                    })


# --------------------------------------------------------------------------
# ECDF Calibrator
# --------------------------------------------------------------------------
class ECDFCalibrator(object):
    """Empirical Cumulative Distribution Function calibrator for risk features."""
    def __init__(self, neg_matrix):
        self.n_samples = len(neg_matrix)
        self.sorted_feats = [
            np.sort(neg_matrix[:, j]) for j in range(3)
        ]
        self.feature_names = ["r_weak", "r_comp", "r_swap"]

    def calibrate(self, raw_array, feat_idx):
        """Map raw 1D array to [0, 1] empirical CDF values."""
        val = np.asarray(raw_array, dtype=np.float64)
        ranks = np.searchsorted(self.sorted_feats[feat_idx], val, side="right")
        return np.clip(ranks / float(self.n_samples), 0.0, 1.0)

    def calibrate_matrix(self, raw_matrix):
        """Map [N, 3] raw matrix to [N, 3] calibrated risk scores."""
        mat = np.asarray(raw_matrix, dtype=np.float64)
        res = np.zeros_like(mat, dtype=np.float64)
        for j in range(3):
            res[:, j] = self.calibrate(mat[:, j], j)
        return res


# --------------------------------------------------------------------------
# Main Execution Pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract gate-free causal risk features and perform ECDF calibration."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, choices=DATASETS,
        help="Datasets to process (default: MOT17 MOT20 SportsMOT)"
    )
    args = parser.parse_args()

    log("Starting gate-free causal risk feature extraction...")
    log("Datasets: %s" % ", ".join(args.datasets))

    all_contexts = []
    for ds in args.datasets:
        log("Processing dataset %s..." % ds)
        ctx = FeatureExtractorContext(ds)
        frames_dir = os.path.join(ROOT, "YOLOX_outputs", EXPNS[ds], "track_results")
        if not os.path.isdir(frames_dir):
            sys.exit("Error: track_results directory not found: %s" % frames_dir)
        txt_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".txt")])

        for f in txt_files:
            seq = f.replace(".txt", "")
            frames = A.load_frames(os.path.join(frames_dir, f))
            extract_sequence_features(ctx, seq, frames)

        log("  %s done: %d events (%d no-box), %d negative detections" % (
            ds, len(ctx.event_records), ctx.n_no_box_events, ctx.n_neg_detections
        ))
        all_contexts.append(ctx)

    # 1. Aggregate Negatives
    log("Aggregating negative sample detections across datasets...")
    neg_arrays = []
    for ctx in all_contexts:
        neg_arrays.extend(ctx.neg_feature_arrays)
    neg_matrix = np.concatenate(neg_arrays, axis=0)
    n_neg = len(neg_matrix)
    log("Total negative detection samples: %d, shape: %s" % (n_neg, str(neg_matrix.shape)))

    # 2. Aggregate Positives (Events)
    all_events = []
    for ctx in all_contexts:
        all_events.extend(ctx.event_records)
    n_events = len(all_events)
    log("Total IDS switch events: %d" % n_events)

    raw_event_matrix = np.array([
        [r["f_weak"], r["f_comp"], r["f_swap"]] for r in all_events
    ], dtype=np.float64)

    # 3. Fit ECDF Calibrator
    log("Fitting empirical cumulative distribution function (ECDF) on negative population...")
    calibrator = ECDFCalibrator(neg_matrix)

    # 4. Calibrate Events and Negatives
    log("Calibrating event risk features into [0, 1] risk scores...")
    risk_event_matrix = calibrator.calibrate_matrix(raw_event_matrix)

    # --------------------------------------------------------------------------
    # Sanity Checks (Hard Asserts)
    # --------------------------------------------------------------------------
    log("Running sanity checks...")

    # (a) Shape checks
    assert risk_event_matrix.shape == (n_events, 3), (
        "Shape mismatch: expected (%d, 3), got %s" % (n_events, str(risk_event_matrix.shape))
    )
    assert raw_event_matrix.shape == (n_events, 3)

    # (b) Zero NaN & range [0, 1]
    n_nans_risk = np.isnan(risk_event_matrix).sum()
    n_nans_raw = np.isnan(raw_event_matrix).sum()
    assert n_nans_risk == 0, "Sanity failure: found %d NaNs in risk_event_matrix!" % n_nans_risk
    assert n_nans_raw == 0, "Sanity failure: found %d NaNs in raw_event_matrix!" % n_nans_raw
    assert (risk_event_matrix >= 0.0).all() and (risk_event_matrix <= 1.0).all(), (
        "Sanity failure: risk_event_matrix values out of [0, 1]!"
    )

    # (c) Event count check
    if set(args.datasets) == set(DATASETS):
        assert n_events == 4713, (
            "Sanity failure: expected 4713 switch events, got %d" % n_events
        )
        class_counts = Counter(r["class"] for r in all_events)
        assert class_counts["S_c"] == 1828, "S_c count mismatch: %d" % class_counts["S_c"]
        assert class_counts["S_r"] == 1899, "S_r count mismatch: %d" % class_counts["S_r"]
        assert class_counts["S_h"] == 986, "S_h count mismatch: %d" % class_counts["S_h"]

    log("All sanity checks passed successfully!")

    # --------------------------------------------------------------------------
    # Export Deliverables
    # --------------------------------------------------------------------------
    os.makedirs(TAXONOMY_DIR, exist_ok=True)

    # 1. Primary NPY Tensor: [N_events, 3]
    npy_path = os.path.join(TAXONOMY_DIR, "risk_features_events.npy")
    np.save(npy_path, risk_event_matrix)
    log("Saved primary risk tensor: %s (shape: %s)" % (npy_path, str(risk_event_matrix.shape)))

    # 2. Comprehensive NPZ Package
    npz_path = os.path.join(TAXONOMY_DIR, "risk_features_events.npz")
    np.savez_compressed(
        npz_path,
        risk_matrix=risk_event_matrix,
        raw_matrix=raw_event_matrix,
        dataset=np.array([r["dataset"] for r in all_events]),
        seq=np.array([r["seq"] for r in all_events]),
        frame=np.array([r["frame"] for r in all_events], dtype=np.int32),
        class_labels=np.array([r["class"] for r in all_events]),
        track_id=np.array([r["track_id"] for r in all_events], dtype=np.int32),
        gt_id_new=np.array([r["gt_id_new"] for r in all_events]),
        no_box=np.array([r["no_box"] for r in all_events], dtype=np.int8),
        feature_names=np.array(["r_weak", "r_comp", "r_swap"]),
        n_neg_samples=np.int64(n_neg)
    )
    log("Saved compressed package: %s" % npz_path)

    # 3. Human-Readable CSV
    csv_path = os.path.join(TAXONOMY_DIR, "risk_features_events.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "seq", "frame", "class", "track_id", "gt_id_new", "no_box",
            "f_weak", "f_comp", "f_swap", "r_weak", "r_comp", "r_swap"
        ])
        for i, r in enumerate(all_events):
            writer.writerow([
                r["dataset"], r["seq"], r["frame"], r["class"], r["track_id"],
                r["gt_id_new"], r["no_box"],
                "%.6f" % raw_event_matrix[i, 0],
                "%.6f" % raw_event_matrix[i, 1],
                "%.6f" % raw_event_matrix[i, 2],
                "%.6f" % risk_event_matrix[i, 0],
                "%.6f" % risk_event_matrix[i, 1],
                "%.6f" % risk_event_matrix[i, 2]
            ])
    log("Saved human-readable CSV: %s" % csv_path)

    # 4. Frozen ECDF Calibrator Model
    calibrator_path = os.path.join(TAXONOMY_DIR, "risk_ecdf_calibrator.npz")
    grid_probs = np.linspace(0.0, 1.0, 1001)
    quantiles_weak = np.quantile(neg_matrix[:, 0], grid_probs)
    quantiles_comp = np.quantile(neg_matrix[:, 1], grid_probs)
    quantiles_swap = np.quantile(neg_matrix[:, 2], grid_probs)

    np.savez_compressed(
        calibrator_path,
        sorted_neg_weak=calibrator.sorted_feats[0],
        sorted_neg_comp=calibrator.sorted_feats[1],
        sorted_neg_swap=calibrator.sorted_feats[2],
        grid_probs=grid_probs,
        quantiles_weak=quantiles_weak,
        quantiles_comp=quantiles_comp,
        quantiles_swap=quantiles_swap,
        n_neg=np.int64(n_neg)
    )
    log("Saved frozen ECDF calibrator: %s" % calibrator_path)

    # --------------------------------------------------------------------------
    # Statistical Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STATISTICAL SUMMARY: GATE-FREE CAUSAL RISK FEATURES & ECDF CALIBRATION")
    print("=" * 78)
    print("Total Negative Detection Population: %d" % n_neg)
    print("Total IDS Switch Events:             %d" % n_events)
    print("-" * 78)
    print("Negative Population Raw Stats:")
    print("  f_weak (1 - top1):  p10=%.4f, p50=%.4f, p90=%.4f, p99=%.4f" % (
        np.percentile(neg_matrix[:, 0], 10), np.percentile(neg_matrix[:, 0], 50),
        np.percentile(neg_matrix[:, 0], 90), np.percentile(neg_matrix[:, 0], 99)
    ))
    print("  f_comp (top2):      p10=%.4f, p50=%.4f, p90=%.4f, p99=%.4f" % (
        np.percentile(neg_matrix[:, 1], 10), np.percentile(neg_matrix[:, 1], 50),
        np.percentile(neg_matrix[:, 1], 90), np.percentile(neg_matrix[:, 1], 99)
    ))
    print("  f_swap (dC_swap):   p10=%.4f, p50=%.4f, p90=%.4f, p99=%.4f" % (
        np.percentile(neg_matrix[:, 2], 10), np.percentile(neg_matrix[:, 2], 50),
        np.percentile(neg_matrix[:, 2], 90), np.percentile(neg_matrix[:, 2], 99)
    ))
    print("-" * 78)
    print("Event Population Calibrated Risk Scores [0, 1]:")
    feat_names = ["r_weak", "r_comp", "r_swap"]
    for j, name in enumerate(feat_names):
        col = risk_event_matrix[:, j]
        print("  %-8s: Mean=%.4f, Std=%.4f, P10=%.4f, P50=%.4f, P90=%.4f, HighRisk(>=0.99)=%5.2f%%" % (
            name, col.mean(), col.std(), np.percentile(col, 10),
            np.percentile(col, 50), np.percentile(col, 90),
            (col >= 0.99).mean() * 100.0
        ))
    print("-" * 78)
    print("Breakdown by Failure Class:")
    classes = [r["class"] for r in all_events]
    for cls in ["S_c", "S_r", "S_h"]:
        mask = np.array([c == cls for c in classes])
        sub = risk_event_matrix[mask]
        print("  Class %-3s (n=%4d, %5.1f%%):" % (cls, mask.sum(), mask.sum() / float(n_events) * 100.0))
        for j, name in enumerate(feat_names):
            col = sub[:, j]
            print("    %-8s: Mean=%.4f, P50=%.4f, P90=%.4f, Risk>=0.99 (FPR<=1%%)=%5.2f%%" % (
                name, col.mean(), np.percentile(col, 50), np.percentile(col, 90),
                (col >= 0.99).mean() * 100.0
            ))
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
