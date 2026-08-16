# run_diag.py
# Day-3 observability diagnostic experiment (Go/No-Go decision, see
# 研究方向.md). Four orthogonal, cheap, online-observable gate signals are
# measured on the frozen switch event pool (S_r active takeover / S_c cold
# start / S_h history reactivation, each analyzed separately plus 'overall')
# vs the C-group (normal-frame) detection population:
#
#   f1 margin      m = top1 - top2            (IoU margin, trigger low)
#   f2 motion      cos_theta = cos(angle(v_hist, v_obs))  (trigger low;
#                  v_obs_norm = |d_t - x_{t-1}| / diag     trigger high)
#   f3 occlusion   Oc_i = max_j |B_i n B_j| / |B_i|        (same-frame IoF,
#                  trigger high)
#   f4 swap        2x2 assignment delta cost
#                  dC = (c12 + c21) - (c11 + c22), D2 = T1's own detection
#                  (variant A) / T2's own detection (variant B)  (trigger high)
#
# Protocol (mirrors V8 / analysis.py):
#   positive = frozen S_r events with a box (246/943/710), negative = all
#   detections in non-event frames with F-1 active tracks (C group);
#   shared eligibility E = top1>=0.2 & top2>=0.2 baked into every trigger;
#   TPR reported at FPR<=1% (primary) and FPR<=2% (sensitivity), thresholds
#   searched per dataset; union rows (1 u 2) and (1 u 2 u 3 u 4) by joint
#   grid search maximizing TPR subject to the FPR budget.
#
# Sanity (hard, SystemExit): S_r counts match the frozen taxonomy, and
# recomputed top1/top2/margin/cos_theta/v_obs_norm must equal the frozen
# taxonomy/gate_feasibility_events.csv values (same code path as
# analysis.py, so any mismatch means a bug in this loop).
#
# Reuses analysis.py utilities by import (guarded main, no side effects).
# Pure stdlib + numpy + scipy + matplotlib(Agg); ASCII comments only.
# Run in-place with the bytetrack conda interpreter:
#   python run_diag.py --datasets mot17
#   python run_diag.py                 # full
# Outputs -> results/ (this folder; no writes to data/ or taxonomy/).

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import analysis as A            # reuses _repo_root/ROOT/load_frames/... (no side effects)

OUT = os.path.join(HERE, "results")
FPR_TARGETS = [0.01, 0.02]
EPS_MARGIN = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
# candidate thresholds for the union search, expressed as neg quantiles
# (per feature, per dataset)
Q_LOW = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.20]
Q_HIGH = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]

_t0 = time.time()


def log(msg):
    print("[+%6.1fs] %s" % (time.time() - _t0, msg))


# --------------------------------------------------------------------------
# New features (f3, f4)
# --------------------------------------------------------------------------
def occlusion_iouf(arr):
    """Oc_i = max_j |B_i n B_j| / |B_i| over same-frame detections j != i.
    Vectorized; 0 for an isolated detection."""
    n = arr.shape[0]
    if n == 0:
        return np.zeros(0)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    np.fill_diagonal(inter, 0.0)
    area = arr[:, 2] * arr[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        iof = inter / np.maximum(area[:, None], 1e-9)
    # clip: IoF is bounded by 1.0 by definition; floating-point residuals can
    # exceed it slightly, which made a threshold candidate exactly at 1.0
    # silently fire on those events (oc>1.0 with zero FPR on negatives).
    return np.clip(iof.max(axis=1), 0.0, 1.0)


def swap_delta_cost(iou):
    """Per-detection 2x2 swap instability on the dets x preds IoU matrix.

    For detection i: T1 = best-matching track, T2 = second-best track.
    D2 = "own detection" of T1 (its best-matching detection, i != i, or its
    second best when i is its best) [variant A]; same for T2 [variant B].
    Costs are IoU similarities; the doc formula
        dC = (c12 + c21) - (c11 + c22),  c11=top1, c12=top2,
        c21=IoU(D2,T1), c22=IoU(D2,T2)
    is the swap-assignment cost minus the current-assignment cost; dC > 0
    means the swap is cheaper -> the current (top1) assignment is unstable.
    NaN where M < 2 or no D2 exists (no local competition)."""
    n, m = iou.shape
    if n == 0 or m < 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    rows = np.arange(n)
    # best/second tracks per detection
    idx = np.argpartition(iou, -2, axis=1)[:, -2:]
    t2, t1 = idx[:, 0], idx[:, 1]
    sw = iou[rows, t2] > iou[rows, t1]
    t1x = np.where(sw, t2, t1)          # best track
    t2x = np.where(sw, t1, t2)          # second track
    # best/second detections per track (iou.T is (M, N))
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
    # variant A: D2 = T1's own detection, != i
    own = d1[t1x]
    d2a = np.where(own != rows, own, d2[t1x])
    ra = np.maximum(d2a, 0)
    c21a = np.where(d2a >= 0, iou[ra, t1x], np.nan)
    c22a = np.where(d2a >= 0, iou[ra, t2x], np.nan)
    dca = (iou[rows, t2x] + c21a) - (iou[rows, t1x] + c22a)
    # variant B: D2 = T2's own detection, != i
    own2 = d1[t2x]
    d2b = np.where(own2 != rows, own2, d2[t2x])
    rb = np.maximum(d2b, 0)
    c21b = np.where(d2b >= 0, iou[rb, t1x], np.nan)
    c22b = np.where(d2b >= 0, iou[rb, t2x], np.nan)
    dcb = (iou[rows, t2x] + c21b) - (iou[rows, t1x] + c22b)
    return dca, dcb


# --------------------------------------------------------------------------
# Per-dataset collection
# --------------------------------------------------------------------------
class DiagCtx(object):
    def __init__(self, ds):
        self.ds = ds
        self.events = A.load_events(ds)
        self.ev_frames = defaultdict(set)
        self.ev_by_frame = defaultdict(list)
        for r, cls in self.events:
            s, F = r["seq"], int(r["frame"])
            self.ev_frames[s].add(F)
            self.ev_by_frame[(s, F)].append((int(r["track_id"]), cls, r))
        self.event_rows = []
        self.n_no_box = 0                # events without an F-frame detection
        self.n_c_det = 0                 # C-group detection count
        self.neg = []                    # per-frame float32 arrays (N, 8)
        self.c_frames = 0

    def add_neg(self, arr):
        self.neg.append(arr.astype(np.float32))
        self.n_c_det += arr.shape[0]

    def neg_all(self):
        if not self.neg:
            return np.zeros((0, 8), dtype=np.float64)
        return np.concatenate(self.neg).astype(np.float64)


def collect_sequence(ctx, seq, frames):
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
            # C group also counts non-event frames without F-1 outputs
            # (analysis.py convention): no tracks -> top1=top2=0 -> can never
            # trigger any feature; they only dilute the FPR denominator
            if not in_ev:
                ctx.c_frames += 1
                ctx.add_neg(np.full((n_det, 8), np.nan))
            else:
                # event frames without F-1 outputs: gate features undefined
                # (no active tracks) -> no-box semantics, same as analysis.py
                for tid, cls, r in cur_evs:
                    ctx.n_no_box += 1
                    ctx.event_rows.append({
                        "ds": ds, "seq": seq, "frame": F, "class": cls,
                        "track_id": tid, "gt_id_new": r["gt_id_new"],
                        "old_hid": r["old_hid"], "no_box": 1,
                        "top1": None, "top2": None, "margin": None,
                        "cos_theta": None, "v_obs_norm": None,
                        "occlusion": None, "swap_A": None, "swap_B": None})
            continue
        prev_arr, prev_ids = A.boxes_array(prev)
        prev2_arr, prev2_ids = (A.boxes_array(prev2) if prev2
                                else (np.zeros((0, 4)), []))
        pred_boxes = {tid: A.extrapolate_box(b, prev2.get(tid) if prev2 else None)
                      for tid, b in prev.items()}
        pred_arr, _ = A.boxes_array(pred_boxes)
        iou = A.iou_matrix(det_arr, pred_arr)
        top1v, top2v, marginv = A._top1_top2_margin(iou)
        ocv = occlusion_iouf(det_arr)
        cosv, _rv, vobsn = A.kmc_arrays(det_ids, det_arr, prev_ids, prev_arr,
                                        prev2_ids, prev2_arr)
        dca, dcb = swap_delta_cost(iou)

        # event rows (all classes; S_c / S_h / S_r each analyzed downstream)
        if cur_evs:
            idx = {t: i for i, t in enumerate(det_ids)}
            for tid, cls, r in cur_evs:
                i = idx.get(tid)
                if i is None:
                    # no F-frame output for the receiving tracker: invisible
                    # to any detection-level gate (counted, features None)
                    ctx.n_no_box += 1
                    ctx.event_rows.append({
                        "ds": ds, "seq": seq, "frame": F, "class": cls,
                        "track_id": tid, "gt_id_new": r["gt_id_new"],
                        "old_hid": r["old_hid"], "no_box": 1,
                        "top1": None, "top2": None, "margin": None,
                        "cos_theta": None, "v_obs_norm": None,
                        "occlusion": None, "swap_A": None, "swap_B": None})
                    continue
                ctx.event_rows.append({
                    "ds": ds, "seq": seq, "frame": F, "class": cls,
                    "track_id": tid, "gt_id_new": r["gt_id_new"],
                    "old_hid": r["old_hid"], "no_box": 0,
                    "top1": float(top1v[i]), "top2": float(top2v[i]),
                    "margin": float(marginv[i]),
                    "cos_theta": None if np.isnan(cosv[i]) else float(cosv[i]),
                    "v_obs_norm": None if np.isnan(vobsn[i]) else float(vobsn[i]),
                    "occlusion": float(ocv[i]),
                    "swap_A": None if np.isnan(dca[i]) else float(dca[i]),
                    "swap_B": None if np.isnan(dcb[i]) else float(dcb[i])})

        # C-group negatives (non-event frames)
        if not in_ev:
            ctx.c_frames += 1
            ctx.add_neg(np.column_stack([top1v, top2v, marginv, cosv, vobsn,
                                         ocv, dca, dcb]))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def feat_defined(col):
    return ~np.isnan(col)


def best_trigger(pos_col, pos_el, neg_col, neg_el, direction, cand):
    """Per-feature trigger = E & (feat < thr) or E & (feat > thr); pick the
    candidate threshold maximizing TPR at each FPR target. FPR denominator
    = ALL C detections (undefined feature -> no trigger)."""
    pv = np.asarray(pos_col, dtype=np.float64)
    nv = np.asarray(neg_col, dtype=np.float64)
    pdef = feat_defined(pv)
    ndef = feat_defined(nv)
    out = {}
    for tgt in FPR_TARGETS:
        best = None
        for thr in cand:
            if direction == "low":
                pt = (pos_el & pdef & (pv < thr)).mean()
                nt = (neg_el & ndef & (nv < thr)).mean()
            else:
                pt = (pos_el & pdef & (pv > thr)).mean()
                nt = (neg_el & ndef & (nv > thr)).mean()
            if nt <= tgt and (best is None or pt > best[0]):
                best = (pt, nt, thr)
        out[tgt] = best
    return out


def union_best(feat_specs, pos_el, pos_feats, neg_el, neg_feats):
    """Joint grid search over per-feature candidate triggers; returns the
    best (tpr, fpr, threshold tuple) per FPR target. An explicit "off" level
    (never-trigger sentinel) is appended to every feature's candidate list so
    the union can always reproduce any subset's operating point."""
    # precompute one trigger array per candidate level
    p_trig = {}                          # (fname, level) -> bool pos array
    n_trig = {}
    for fname, col_i, direction, cand in feat_specs:
        # "off" sentinel: cannot fire (all real values are far from it)
        if direction == "low":
            cand.append(-1e9)
        else:
            cand.append(1e9)
        pv = pos_feats[:, col_i]
        nv = neg_feats[:, col_i]
        pdef = feat_defined(pv)
        ndef = feat_defined(nv)
        for k, thr in enumerate(cand):
            if direction == "low":
                p_trig[(fname, k)] = pos_el & pdef & (pv < thr)
                n_trig[(fname, k)] = neg_el & ndef & (nv < thr)
            else:
                p_trig[(fname, k)] = pos_el & pdef & (pv > thr)
                n_trig[(fname, k)] = neg_el & ndef & (nv > thr)
    out = {}
    for tgt in FPR_TARGETS:
        best = None
        # product over candidate levels, one per feature
        levels = [0] * len(feat_specs)
        while True:
            pt = np.zeros(len(pos_el), dtype=bool)
            nt = np.zeros(len(neg_el), dtype=bool)
            for fi, lv in enumerate(levels):
                pt |= p_trig[(feat_specs[fi][0], lv)]
                nt |= n_trig[(feat_specs[fi][0], lv)]
            fpr = nt.mean()
            if fpr <= tgt:
                tpr = pt.mean()
                if best is None or tpr > best[0]:
                    best = (tpr, fpr, tuple(levels))
            # increment the mixed-radix counter
            i = 0
            while i < len(feat_specs):
                levels[i] += 1
                if levels[i] < len(feat_specs[i][3]):
                    break
                levels[i] = 0
                i += 1
            if i == len(feat_specs):
                break
        out[tgt] = best
    return out


def union_thresholds(feat_specs, res, tgt):
    """Human-readable threshold tuple of a union result (best[2] levels).
    The "off" sentinel levels (+/-1e9 appended in union_best) render as
    "fname=off" so a closed gate is unambiguous at a glance (a threshold
    rendered as e.g. "oc>=1" previously read as closed while still firing)."""
    if res is None:
        return ""
    names = []
    for fi, lv in enumerate(res[2]):
        fname, _, direction, cand = feat_specs[fi]
        thr = cand[lv]
        if abs(thr) >= 1e8:          # off sentinel
            names.append("%s=off" % fname)
        elif fname == "oc" and direction == "high" and thr >= 1.0:
            # oc is an IoF in [0,1] (clipped in occlusion_iouf): a candidate
            # at the bound can never fire; render as off to keep "oc>=1"
            # from reading as a closed gate while actually being live
            names.append("%s=off" % fname)
        else:
            names.append("%s%s=%.4g" % (fname,
                                        "<" if direction == "low" else ">",
                                        thr))
    return " ".join(names)


def pos_arrays(rows):
    """Per-event feature arrays from the collected rows (no-box rows -> NaN)."""
    pos = {k: np.full(len(rows), np.nan) for k in
           ("margin", "cos", "vobs", "oc", "dca", "dcb")}
    pos_t12 = np.zeros((len(rows), 2))
    for i, r in enumerate(rows):
        if r["top1"] is None:
            continue                     # no-box event: all features undefined
        pos_t12[i] = (r["top1"], r["top2"])
        pos["margin"][i] = r["margin"]
        if r["cos_theta"] is not None:
            pos["cos"][i] = r["cos_theta"]
        if r["v_obs_norm"] is not None:
            pos["vobs"][i] = r["v_obs_norm"]
        pos["oc"][i] = r["occlusion"]
        if r["swap_A"] is not None:
            pos["dca"][i] = r["swap_A"]
        if r["swap_B"] is not None:
            pos["dcb"][i] = r["swap_B"]
    return pos, pos_t12


CLASSES = ("S_c", "S_r", "S_h", "overall")


def analyze_ds(ds, ctx, frozen):
    """Per-dataset analysis: per event class (S_c / S_r / S_h / overall)."""
    log("analyzing %s" % ds)
    neg = ctx.neg_all()                  # (N, 8) [t1, t2, margin, cos, vobs, oc, dca, dcb]
    roc_rows = []
    union_rows = []
    for cls in CLASSES:
        rows = ctx.event_rows if cls == "overall" else \
            [r for r in ctx.event_rows if r["class"] == cls]
        pos, pos_t12 = pos_arrays(rows)
        rr, ur, _per = analyze_arrays(ds, cls, rows, pos, pos_t12, neg)
        roc_rows.extend(rr)
        union_rows.extend(ur)
    return roc_rows, union_rows


# --------------------------------------------------------------------------
# Sanity vs the frozen analysis products
# --------------------------------------------------------------------------
def run_sanity(ds, ctx, frozen):
    # per-class counts == frozen gate_feasibility_events.csv
    got_c = Counter(r["class"] for r in ctx.event_rows)
    exp_c = Counter(fr["class"] for fr in frozen.values())
    for cls in ("S_c", "S_r", "S_h"):
        if got_c[cls] != exp_c[cls]:
            raise SystemExit("SANITY FAIL: %s class %s count %d != frozen %d"
                             % (ds, cls, got_c[cls], exp_c[cls]))
    if ctx.n_no_box != sum(1 for fr in frozen.values() if fr["no_box"] == "1"):
        raise SystemExit("SANITY FAIL: %s no-box count %d != frozen"
                         % (ds, ctx.n_no_box))
    # per-value match against taxonomy/gate_feasibility_events.csv
    mism = 0
    for r in ctx.event_rows:
        key = (r["seq"], r["frame"], r["track_id"])
        fr = frozen.get(key)
        if fr is None:
            raise SystemExit("SANITY FAIL: %s frozen row missing for %s"
                             % (ds, key))
        if r["no_box"] == 1:
            continue                     # features None on both sides
        for f, frozen_col in (("top1", "top1"), ("top2", "top2"),
                              ("margin", "margin"), ("cos_theta", "cos_theta"),
                              ("v_obs_norm", "v_obs_norm")):
            mine = r[f]
            theirs = fr[frozen_col]
            if mine is None or theirs in ("", None):
                if not (mine is None and theirs in ("", None)):
                    mism += 1
            # frozen CSV stores 4-decimal-rounded values; identical math
            # reproduces them up to rounding (float noise <= 5e-5)
            elif abs(mine - float(theirs)) > 5.0001e-5:
                mism += 1
    if mism:
        raise SystemExit("SANITY FAIL: %s feature mismatches vs frozen: %d"
                         % (ds, mism))
    log("   sanity %s OK: S_c=%d S_r=%d S_h=%d no_box=%d, per-value match vs "
        "frozen CSV" % (ds, got_c["S_c"], got_c["S_r"], got_c["S_h"],
                        ctx.n_no_box))


def load_frozen(ds):
    """All classes from the frozen taxonomy/gate_feasibility_events.csv."""
    path = os.path.join(A.OUT, "gate_feasibility_events.csv")
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != ds:
                continue
            out[(r["seq"], int(r["frame"]), int(r["track_id"]))] = r
    return out


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
def write_events_csv(rows):
    path = os.path.join(OUT, "diag_features_events.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seq", "frame", "class", "track_id",
                    "gt_id_new", "old_hid", "no_box", "top1", "top2", "margin",
                    "cos_theta", "v_obs_norm", "occlusion", "swap_A", "swap_B"])
        for r in rows:
            w.writerow([r["ds"], r["seq"], r["frame"], r["class"],
                        r["track_id"], r["gt_id_new"], r["old_hid"], r["no_box"],
                        "" if r["top1"] is None else "%.6f" % r["top1"],
                        "" if r["top2"] is None else "%.6f" % r["top2"],
                        "" if r["margin"] is None else "%.6f" % r["margin"],
                        "" if r["cos_theta"] is None else "%.6f" % r["cos_theta"],
                        "" if r["v_obs_norm"] is None else "%.6f" % r["v_obs_norm"],
                        "" if r["occlusion"] is None else "%.6f" % r["occlusion"],
                        "" if r["swap_A"] is None else "%.6f" % r["swap_A"],
                        "" if r["swap_B"] is None else "%.6f" % r["swap_B"]])
    log("wrote %s (%d rows)" % (path, len(rows)))


def write_roc_summary(rows):
    path = os.path.join(OUT, "diag_roc_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "class", "feature", "direction", "n_pos",
                    "n_pos_defined", "auc", "tpr_1", "fpr_1", "tpr_2", "fpr_2"])
        for r in rows:
            w.writerow([r["dataset"], r["class"], r["feature"], r["direction"],
                        r["n_pos"], r["n_pos_defined"], r["auc"],
                        r["tpr_1"], r["fpr_1"], r["tpr_2"], r["fpr_2"]])
    log("wrote %s (%d rows)" % (path, len(rows)))


def write_union_summary(rows):
    path = os.path.join(OUT, "diag_union_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "class", "union", "motion_signal", "tpr_1",
                    "fpr_1", "thr_1", "tpr_2", "fpr_2", "thr_2"])
        for r in rows:
            w.writerow([r["dataset"], r["class"], r["union"],
                        r["motion_signal"],
                        r["tpr_1"], r["fpr_1"], r["thr_1"],
                        r["tpr_2"], r["fpr_2"], r["thr_2"]])
    log("wrote %s (%d rows)" % (path, len(rows)))


def make_figure(all_roc):
    """2x2 panels (combined + 3 datasets) with the 4-6 feature ROC curves.
    ROC computed on the defined/eligible subpopulation (pure signal power)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"margin": "#2a78d6", "cos": "#eb6834", "vobs": "#e08a2e",
              "oc": "#1baf7a", "dca": "#9b59b6", "dcb": "#c9a227"}
    labels = {"margin": "margin", "cos": "cos_theta", "vobs": "v_obs_norm",
              "oc": "occlusion IoF", "dca": "swap dC (A)", "dcb": "swap dC (B)"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=200)
    panels = [("combined", "Combined"), ("MOT17", "MOT17"),
              ("MOT20", "MOT20"), ("SportsMOT", "SportsMOT")]
    for ax, (key, title) in zip(axes.flat, panels):
        curves = {}
        for ds in ("MOT17", "MOT20", "SportsMOT"):
            for fname, (fpr, tpr) in all_roc.get(ds, {}).items():
                if len(fpr) < 3:
                    continue
                if key == "combined":
                    curves.setdefault(fname, ([], []))
                    curves[fname][0].extend(fpr.tolist())
                    curves[fname][1].extend(tpr.tolist())
                elif ds == key:
                    curves[fname] = (fpr, tpr)
        for fname, (fpr, tpr) in curves.items():
            ax.plot(fpr, tpr, color=colors[fname],
                    linewidth=1.4 if key == "combined" else 2.0,
                    alpha=0.55 if key == "combined" else 1.0,
                    label=labels[fname])
        ax.plot([0, 0.05], [0, 0.05], "--", color="#9a9890", linewidth=1.0)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 0.05)
        ax.set_ylim(0, 1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True, color="#e1e0d9", linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT, "diag_roc.png")
    fig.savefig(path, dpi=200)
    log("wrote %s" % path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="observability diagnostic (Go/No-Go)")
    ap.add_argument("--datasets", default="mot17,mot20,sportsmot")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    ds_map = {"mot17": "MOT17", "mot20": "MOT20", "sportsmot": "SportsMOT"}
    datasets = [ds_map[d.strip().lower()] for d in args.datasets.split(",")]
    os.makedirs(OUT, exist_ok=True)
    log("diag start | datasets=%s" % datasets)

    all_rows = []
    all_roc = {}
    roc_rows = []
    union_rows = []
    ctxs = {}
    for ds in datasets:
        t_ds = time.time()
        ctx = DiagCtx(ds)
        ctxs[ds] = ctx
        frozen = load_frozen(ds)
        tr_dir = os.path.join(A.ROOT, "YOLOX_outputs", A.EXPNS[ds], "track_results")
        for fn in sorted(os.listdir(tr_dir)):
            if not fn.endswith(".txt"):
                continue
            seq = fn[:-4]
            frames = A.load_frames(os.path.join(tr_dir, fn))
            collect_sequence(ctx, seq, frames)
            del frames
        run_sanity(ds, ctx, frozen)
        log("   %s: events=%d C dets=%d C frames=%d | %.1fs"
            % (ds, len(ctx.event_rows), ctx.n_c_det, ctx.c_frames,
               time.time() - t_ds))
        rr, ur = analyze_ds(ds, ctx, frozen)
        roc_rows.extend(rr)
        union_rows.extend(ur)
        all_rows.extend(ctx.event_rows)
        # full ROC curves for the figure (S_r only, defined/eligible pool)
        sr_rows = [r for r in ctx.event_rows if r["class"] == "S_r"]
        pos, pos_t12 = pos_arrays(sr_rows)
        neg = ctx.neg_all()
        pos_el = (pos_t12[:, 0] >= A.TOP1_TH) & (pos_t12[:, 1] >= A.TOP2_TH)
        neg_el = (neg[:, 0] >= A.TOP1_TH) & (neg[:, 1] >= A.TOP2_TH)
        all_roc[ds] = {}
        for fname, ci, direction in (("margin", 2, "low"), ("cos", 3, "low"),
                                     ("vobs", 4, "high"), ("oc", 5, "high"),
                                     ("dca", 6, "high"), ("dcb", 7, "high")):
            pm = pos_el & feat_defined(pos[fname])
            nm = neg_el & feat_defined(neg[:, ci])
            all_roc[ds][fname] = A.roc_curve(pos[fname][pm],
                                             neg[:, ci][nm], direction)

    # combined pass: pool all positives and negatives, per class
    if len(datasets) > 1:
        comb_neg = np.concatenate([ctxs[ds].neg_all() for ds in datasets])
        for cls in CLASSES:
            comb_rows = [r for ds in datasets for r in ctxs[ds].event_rows
                         if cls == "overall" or r["class"] == cls]
            pos, pos_t12 = pos_arrays(comb_rows)
            crows, cunion, _cper = analyze_arrays("combined", cls, comb_rows,
                                                  pos, pos_t12, comb_neg)
            roc_rows.extend(crows)
            union_rows.extend(cunion)

    write_events_csv(all_rows)
    write_roc_summary(roc_rows)
    write_union_summary(union_rows)
    if not args.no_figure:
        make_figure(all_roc)
    log("diag done")


def analyze_arrays(ds, cls, rows, pos, pos_t12, neg):
    """Feature/union analysis on arrays for one event class (cls='overall'
    pools all switch events)."""
    pos_el = (pos_t12[:, 0] >= A.TOP1_TH) & (pos_t12[:, 1] >= A.TOP2_TH)
    neg_el = (neg[:, 0] >= A.TOP1_TH) & (neg[:, 1] >= A.TOP2_TH)
    log("   %s %s: events=%d (eligible %d) | C dets=%d (eligible %d)"
        % (ds, cls, len(rows), int(pos_el.sum()), len(neg), int(neg_el.sum())))
    feats = [("margin", 2, "low", list(EPS_MARGIN)),
             ("cos", 3, "low", None),
             ("vobs", 4, "high", None),
             ("oc", 5, "high", None),
             ("dca", 6, "high", None),
             ("dcb", 7, "high", None)]
    roc_rows = []
    per_feat = {}
    for fname, ci, direction, cand in feats:
        pv = pos[fname]
        nv = neg[:, ci]
        nv_def = nv[feat_defined(nv)]
        if cand is None:
            qs = Q_LOW if direction == "low" else Q_HIGH
            cand = [float(np.quantile(nv_def, q)) for q in qs]
        res = best_trigger(pv, pos_el, nv, neg_el, direction, cand)
        pm = pos_el & feat_defined(pv)
        nm = neg_el & feat_defined(nv)
        fpr, tpr = A.roc_curve(pv[pm], nv[nm], direction)
        auc = float(np.trapz(tpr, fpr)) if len(fpr) > 2 else float("nan")
        row = {"dataset": ds, "class": cls, "feature": fname,
               "direction": direction, "n_pos": len(rows),
               "n_pos_defined": int(pm.sum()), "auc": "%.4f" % auc}
        for tgt in FPR_TARGETS:
            b = res[tgt]
            row["tpr_%d" % int(100 * tgt)] = A.fmt_rate(b[0]) if b else "n/a"
            row["fpr_%d" % int(100 * tgt)] = A.fmt_rate(b[1]) if b else "n/a"
        roc_rows.append(row)
        per_feat[fname] = res
        log("   %s %s: %s TPR@1%%=%s TPR@2%%=%s AUC=%.4f (defined %d/%d)"
            % (ds, cls, fname, row["tpr_1"], row["tpr_2"], auc,
               int(pm.sum()), len(rows)))
    motion_name = "cos" if float(roc_rows[1]["auc"]) >= float(roc_rows[2]["auc"]) \
        else "vobs"
    pos_feats = np.column_stack([pos["margin"], pos["cos"], pos["vobs"],
                                 pos["oc"], pos["dca"], pos["dcb"]])
    neg_feats = neg[:, [2, 3, 4, 5, 6, 7]]

    def cand_for(fname):
        nv = neg_feats[:, {"margin": 0, "cos": 1, "vobs": 2, "oc": 3,
                           "dca": 4, "dcb": 5}[fname]]
        nv_def = nv[feat_defined(nv)]
        qs = Q_LOW if fname in ("margin", "cos") else Q_HIGH
        return [float(np.quantile(nv_def, q)) for q in qs]

    union_rows = []
    for uname, spec in [
            ("u12", [("margin", 0, "low", list(EPS_MARGIN)),
                     (motion_name, 2 if motion_name == "vobs" else 1,
                      "high" if motion_name == "vobs" else "low",
                      cand_for(motion_name))]),
            ("u1234", [("margin", 0, "low", list(EPS_MARGIN)),
                       (motion_name, 2 if motion_name == "vobs" else 1,
                        "high" if motion_name == "vobs" else "low",
                        cand_for(motion_name)),
                       ("oc", 3, "high", cand_for("oc")),
                       ("dcb", 5, "high", cand_for("dcb"))]),
            # sensitivity: swap variant A (D2 = T1's own detection) instead
            # of B in the 4-signal union; differs mainly for S_c/S_h
            ("u1234a", [("margin", 0, "low", list(EPS_MARGIN)),
                        (motion_name, 2 if motion_name == "vobs" else 1,
                         "high" if motion_name == "vobs" else "low",
                         cand_for(motion_name)),
                        ("oc", 3, "high", cand_for("oc")),
                        ("dca", 4, "high", cand_for("dca"))])]:
        res = union_best(spec, pos_el, pos_feats, neg_el, neg_feats)
        row = {"dataset": ds, "class": cls, "union": uname,
               "motion_signal": motion_name}
        for tgt in FPR_TARGETS:
            b = res[tgt]
            row["tpr_%d" % int(100 * tgt)] = A.fmt_rate(b[0]) if b else "n/a"
            row["fpr_%d" % int(100 * tgt)] = A.fmt_rate(b[1]) if b else "n/a"
            row["thr_%d" % int(100 * tgt)] = union_thresholds(spec, b, tgt)
        union_rows.append(row)
        log("   %s %s %s: TPR@1%%=%s (thr %s)"
            % (ds, cls, uname, row["tpr_1"], row["thr_1"]))

    # u2: fixed-threshold 2-signal union (margin + swap-B). Each signal's
    # threshold is chosen by its OWN FPR<=tgt diagnostic (best_trigger);
    # the union then applies those thresholds as-is WITHOUT a joint FPR
    # constraint -- the joint FPR is reported and may exceed tgt by the
    # union of the two individual FPRs. This avoids the joint search
    # trading one signal off against the other (e.g. closing vobs to keep
    # the overall FPR within budget), and is the reported headline union.
    b_m = best_trigger(pos["margin"], pos_el, neg_feats[:, 0], neg_el,
                       "low", list(EPS_MARGIN))
    b_d = best_trigger(pos["dcb"], pos_el, neg_feats[:, 5], neg_el,
                       "high", cand_for("dcb"))
    u2row = {"dataset": ds, "class": cls, "union": "u2",
             "motion_signal": motion_name}
    for tgt in FPR_TARGETS:
        rm, rd = b_m[tgt], b_d[tgt]
        if rm is None or rd is None:
            u2row["tpr_%d" % int(100 * tgt)] = "n/a"
            u2row["fpr_%d" % int(100 * tgt)] = "n/a"
            u2row["thr_%d" % int(100 * tgt)] = ""
            continue
        thr_m, thr_d = rm[2], rd[2]
        pm = pos_el & ~np.isnan(pos["margin"]) & (pos["margin"] < thr_m)
        pd = pos_el & ~np.isnan(pos["dcb"]) & (pos["dcb"] > thr_d)
        nm = neg_el & ~np.isnan(neg_feats[:, 0]) & (neg_feats[:, 0] < thr_m)
        nd = neg_el & ~np.isnan(neg_feats[:, 5]) & (neg_feats[:, 5] > thr_d)
        u2row["tpr_%d" % int(100 * tgt)] = A.fmt_rate((pm | pd).mean())
        u2row["fpr_%d" % int(100 * tgt)] = A.fmt_rate((nm | nd).mean())
        u2row["thr_%d" % int(100 * tgt)] = "margin<=%.4g dcb>=%.4g" \
            % (thr_m, thr_d)
    union_rows.append(u2row)
    log("   %s %s u2: TPR@1%%=%s FPR@1%%=%s (thr %s)"
        % (ds, cls, u2row["tpr_1"], u2row["fpr_1"], u2row["thr_1"]))
    return roc_rows, union_rows, per_feat


if __name__ == "__main__":
    main()
