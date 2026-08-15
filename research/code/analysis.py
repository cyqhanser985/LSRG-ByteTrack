# analysis.py
# Offline gate-feasibility analysis for the S_r (active takeover) detection
# problem, on real tracking outputs. Single entry point for the whole
# analysis chain:
#
#   (a) geometric gate baseline (V7): for each detection box D at frame F vs
#       the constant-velocity predicted boxes of all trackers active at F-1,
#       top1/top2/margin IoU; trigger <=> top1>=0.2 and top2>=0.2 and
#       margin<eps. Group A = S_r event detections, B = S_c u S_h,
#       C = normal (non-event) frames.
#   (b) three candidate mechanisms (V8) on the same population:
#       - KMC kinematic consistency: cos_theta (angle between v_hist =
#         F-1->F-2 and v_obs = F->F-1 center displacements) and r_v =
#         max/min speed ratio; trigger = geometric OR (cos<thr and top1>=0.2)
#         OR (r_v>thr and top1>=0.2).
#       - KF adaptive margin: sigma = sqrt(P00+P11) of the Kalman prediction
#         covariance (replayed offline over the output boxes; the covariance
#         evolution is measurement-independent, so the replayed sigma matches
#         the online value), eps_i = eps0 * (1 + alpha * sigma_norm_i).
#       - DEN local density: N_neighbor(i) = number of other trackers whose
#         F-1 box overlaps track i's F-1 box; eps_i = eps0 * gamma(N).
#
# Protocol
# --------
# Positive samples: S_r switch events (1921; gap always 1), match pair
# (receiving tracker tid, its F detection box). Negative samples: all
# detections in non-event frames (V7 group-C population), with per-mechanism
# eligibility filters. All signals are causal (<= F-1 outputs + F detection).
# Reported metrics: TPR at fixed FPR=1% (Wilson CI) for the absolute
# coverage, plus the mechanism gain over the plain fixed-eps gate at the
# same FPR budget (coverage can be high while the mechanism adds ~0).
#
# Outputs (data only; reports are maintained by hand as static documents):
#   taxonomy/gate_feasibility_events.csv   per-event features (4756 rows)
#   taxonomy/gate_feasibility_summary.csv  long-form trigger-rate table
#   taxonomy/gate_feasibility_roc.png      ROC figure (optional, --no-figure)
#
# Sanity (hard asserts, SystemExit on failure): V6 population landmarks
# (frames/dets/margin counts), event class counts vs
# event_counts_by_sequence.csv, (seq,frame,tid) uniqueness + S_r gap=1,
# KF replay triple sanity (sigma steady state / updates+init==rows / mean
# deviation).
#
# Pure stdlib + numpy + scipy + matplotlib(Agg). ASCII comments only.
# Run in-place from research/code/ with the bytetrack conda interpreter.

import argparse
import csv
import importlib.util
import os
import time
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
EPS = [0.05, 0.10, 0.15, 0.20, 0.25]
TOP1_TH = 0.2
TOP2_TH = 0.2
DATASETS = ["MOT17", "MOT20", "SportsMOT"]
EXPNS = {"MOT17": "mot17_v001_full", "MOT20": "mot20_v001_full",
         "SportsMOT": "sportsmot_v001_full"}
# V6 exact landmarks (frozen; the analysis population never changes)
V6_FRAMES = {"MOT17": 7956, "MOT20": 8927, "SportsMOT": 55450}
V6_DETS = {"MOT17": 152868, "MOT20": 1116661, "SportsMOT": 599109}
V6_DET_MARGIN = {"MOT17": {0.05: 492, 0.10: 774, 0.20: 1467},
                 "MOT20": {0.05: 1963, 0.10: 3401, 0.20: 7161},
                 "SportsMOT": {0.05: 2028, 0.10: 2609, 0.20: 4857}}
V6_FRAMES_ANY = {"MOT17": 6633, "MOT20": 8927, "SportsMOT": 39557}
EPS_IDX = {0.05: 0, 0.10: 1, 0.20: 3}      # index into EPS (V6 landmark eps)
MAX_TIME_LOST = 30            # = buffer_size (track_buffer 30, frame_rate 30)
FPR_TARGET = 0.01             # fixed FPR point where TPR is reported
GRID_COS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
GRID_RV = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
GRID_EPS0 = [0.05, 0.10, 0.15, 0.20, 0.25]
GRID_ALPHA = [0.5, 1.0, 2.0, 4.0]
GRID_GAMMA = [1.0, 1.25, 1.5, 2.0]
EPS0_OR = 0.20                # geometric gate eps used in OR coverage

_t0 = time.time()


def log(msg):
    print("[+%6.1fs] %s" % (time.time() - _t0, msg))


# --------------------------------------------------------------------------
# Track-result utilities
# --------------------------------------------------------------------------
def _repo_root():
    """Repo root = first dir upward containing YOLOX_outputs (works from
    research/code/ or tools/)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, "YOLOX_outputs")):
        nd = os.path.dirname(d)
        if nd == d:
            return d
        d = nd
    return d


ROOT = _repo_root()
DATA = os.path.join(ROOT, "research", "data")
OUT = os.path.join(ROOT, "research", "taxonomy")


def load_frames(txt_path):
    """{frame: {tid: (x, y, w, h)}} for one sequence."""
    frames = defaultdict(dict)
    with open(txt_path, encoding="utf-8-sig") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frames[int(float(p[0]))][int(float(p[1]))] = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
    return frames


def boxes_array(d):
    """dict tid->box -> (np array (N,4) [x,y,w,h], sorted tid list)."""
    tids = sorted(d)
    arr = np.array([[d[t][0], d[t][1], d[t][2], d[t][3]] for t in tids], dtype=np.float64)
    return arr, tids


def iou_matrix(dets, preds):
    """(N,4)x(M,4) -> (N,M) IoU matrix, vectorized."""
    n = dets.shape[0]
    m = preds.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m))
    ax1, ay1 = dets[:, 0], dets[:, 1]
    ax2, ay2 = dets[:, 0] + dets[:, 2], dets[:, 1] + dets[:, 3]
    bx1, by1 = preds[:, 0], preds[:, 1]
    bx2, by2 = preds[:, 0] + preds[:, 2], preds[:, 1] + preds[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_d = dets[:, 2] * dets[:, 3]
    area_p = preds[:, 2] * preds[:, 3]
    union = area_d[:, None] + area_p[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def extrapolate_box(box_at_prev, box_at_prev2):
    """Constant velocity: center shifted by v, size kept; constant-position
    fallback when box_at_prev2 is None. Matches ByteTrack's Kalman mean."""
    if box_at_prev2 is None:
        return box_at_prev
    cx0, cy0 = box_at_prev[0] + box_at_prev[2] / 2.0, box_at_prev[1] + box_at_prev[3] / 2.0
    cx1, cy1 = box_at_prev2[0] + box_at_prev2[2] / 2.0, box_at_prev2[1] + box_at_prev2[3] / 2.0
    vx, vy = cx0 - cx1, cy0 - cy1
    return (cx0 + vx - box_at_prev[2] / 2.0, cy0 + vy - box_at_prev[3] / 2.0,
            box_at_prev[2], box_at_prev[3])


def _top1_top2_margin(iou):
    """(N,M) IoU matrix -> (top1, top2, margin) arrays."""
    n, m = iou.shape
    if m >= 2:
        top = np.partition(iou, -2, axis=1)[:, -2:]
        top2v, top1v = top[:, 0], top[:, 1]
    elif m == 1:
        top1v, top2v = iou[:, 0], np.zeros(n)
    else:
        top1v = np.zeros(n)
        top2v = np.zeros(n)
    return top1v, top2v, top1v - top2v


def _trig_matrix(top1v, top2v, marginv):
    """(N,5) bool matrix: detection x eps trigger flags."""
    eps_arr = np.array(EPS)
    return ((top1v[:, None] >= TOP1_TH) & (top2v[:, None] >= TOP2_TH) &
            (marginv[:, None] < eps_arr[None, :]))


# --------------------------------------------------------------------------
# Event taxonomy (single source of truth)
# --------------------------------------------------------------------------
def flags(row):
    s = row["na_flag"].strip()
    return set(x for x in s.split("|") if x) if s else set()


def classify(fl):
    """S_c cold start (never output before) / S_r active takeover (output at
    F-1) / S_h history reactivation (output earlier but not at F-1)."""
    if "no_last_seen" in fl:
        return "S_c"
    if "no_prev" not in fl:
        return "S_r"
    return "S_h"


def load_events(ds):
    """Frozen switch events for one dataset, classified. type=="switch" rows
    only (reuse rows are the counterpart view of the same events)."""
    out = []
    with open(os.path.join(DATA, "%s_events_metrics.csv" % ds), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["type"].strip() != "switch":
                continue
            out.append((r, classify(flags(r))))
    return out


def load_by_sequence_counts():
    """Per-dataset per-class expected counts from
    taxonomy/event_counts_by_sequence.csv (frozen V1 product)."""
    exp = defaultdict(Counter)
    with open(os.path.join(OUT, "event_counts_by_sequence.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            exp[r["dataset"]]["S_c"] += int(r["n_S_c"])
            exp[r["dataset"]]["S_r"] += int(r["n_S_r"])
            exp[r["dataset"]]["S_h"] += int(r["n_S_h"])
    return exp


# --------------------------------------------------------------------------
# Kalman filter loading (pure numpy module, no package side effects)
# --------------------------------------------------------------------------
def load_kalman_filter():
    kf_path = os.path.join(ROOT, "yolox", "tracker", "kalman_filter.py")
    spec = importlib.util.spec_from_file_location("analysis_kalman_filter", kf_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KalmanFilter


class KfReplayState(object):
    """Replay the ByteTrack Kalman filter over a track_results sequence.

    Schedule simulated: initiate on first output; predict every frame for
    every confirmed track (lost tracks keep predicting with vh zeroed);
    update on output frames; unconfirmed tracks (single output) that
    reappear after a gap are re-initiated; tracks lost for more than
    MAX_TIME_LOST frames are dropped. The covariance evolution is
    measurement-independent, so the replayed sigma (sqrt(P00+P11) after
    predict, before update) matches the online value up to the h coupling
    (<0.1%)."""

    def __init__(self, kf):
        self.kf = kf
        self.tracks = {}       # tid -> dict(mean, cov, last_out, confirmed, lost_age)
        self.n_updates = 0
        self.n_reinit = 0
        self.n_init = 0
        self.mean_checks = 0
        self.mean_diffs = []   # relative mean-vs-output-box deviations
        self.sig_series = defaultdict(list)   # tid -> [(F, sigma_norm)]

    @staticmethod
    def _xyah(box):
        x, y, w, h = box
        return np.array([x + w / 2.0, y + h / 2.0, w / h, h], dtype=np.float64)

    @staticmethod
    def _mean_tlwh(mean):
        x, y, a, h = mean[:4]
        w = a * h
        return np.array([x - w / 2.0, y - h / 2.0, w, h], dtype=np.float64)

    def step(self, F, dets):
        """Advance one frame. dets: {tid: (x,y,w,h)} output boxes of frame F.
        Returns {tid: (sigma_post_predict, h_state)} for all alive tracks."""
        cur = set(dets)
        # 1. predict all confirmed tracks (batch when the pool is large;
        #    multi_predict has array-construction overhead for few tracks)
        alive = [(tid, st) for tid, st in self.tracks.items() if st["confirmed"]]
        if len(alive) >= 50:
            tids = [t for t, _ in alive]
            means = np.asarray([st["mean"] for _, st in alive], dtype=np.float64)
            covs = np.asarray([st["cov"] for _, st in alive], dtype=np.float64)
            for i, tid in enumerate(tids):
                if tid not in cur:
                    means[i, 7] = 0.0    # lost track: zero vh (online behavior)
            pm, pc = self.kf.multi_predict(means, covs)
            for i, tid in enumerate(tids):
                st = self.tracks[tid]
                st["mean"], st["cov"] = pm[i], pc[i]
                st["lost_age"] += 1 if tid not in cur else 0
        else:
            for tid, st in alive:
                mean = st["mean"].copy()
                if tid not in cur:
                    mean[7] = 0.0
                st["mean"], st["cov"] = self.kf.predict(mean, st["cov"])
                st["lost_age"] += 1 if tid not in cur else 0
        # 2. remove over-age lost tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["lost_age"] > MAX_TIME_LOST:
                del self.tracks[tid]
        # 3. update / initiate / re-initiate for current outputs
        sigmas = {}
        for tid, box in dets.items():
            st = self.tracks.get(tid)
            meas = self._xyah(box)
            if st is not None and st["confirmed"]:
                st["mean"], st["cov"] = self.kf.update(st["mean"], st["cov"], meas)
                st["last_out"] = F
                st["lost_age"] = 0
                self.n_updates += 1
                self._sample_mean_check(F, tid, box)
            elif st is not None and st["last_out"] == F - 1:
                # unconfirmed but consecutive second output: update + confirm
                st["mean"], st["cov"] = self.kf.update(st["mean"], st["cov"], meas)
                st["confirmed"] = True
                st["last_out"] = F
                st["lost_age"] = 0
                self.n_updates += 1
                self._sample_mean_check(F, tid, box)
            else:
                if st is not None:
                    del self.tracks[tid]     # unconfirmed reappeared after gap
                    self.n_reinit += 1
                mean, cov = self.kf.initiate(meas)
                self.tracks[tid] = {"mean": mean, "cov": cov,
                                    "last_out": F, "confirmed": False, "lost_age": 0}
                self.n_init += 1
        # 4. collect post-predict sigma; steady-state series records OUTPUT
        #    frames only, normalized by the filter's h (mean[3], which scales
        #    Q/R) -- the measured h jitters with detection noise
        for tid, st in self.tracks.items():
            h_state = float(st["mean"][3])
            sigmas[tid] = (float(np.sqrt(st["cov"][0, 0] + st["cov"][1, 1])),
                           h_state)
            if st["confirmed"] and tid in cur:
                self.sig_series[tid].append(
                    (F, sigmas[tid][0] / h_state if h_state > 0 else float("nan")))
        return sigmas

    def _sample_mean_check(self, F, tid, box):
        """Relative deviation of the replayed filtered mean (after update,
        converted to tlwh) vs the output box, sub-sampled."""
        self.mean_checks += 1
        if self.mean_checks % 2000 != 0:
            return
        st = self.tracks[tid]
        tlwh = self._mean_tlwh(st["mean"])
        d = np.abs(tlwh - np.asarray(box, dtype=np.float64))
        diag = float((box[2] ** 2 + box[3] ** 2) ** 0.5)
        if diag > 0:
            self.mean_diffs.append(float(d.max()) / diag)

    def sigma_steady_check(self):
        """ANY 10-frame window of the (output-only) sigma_norm series must be
        steady: (P90-P10)/median < 5%. The first ~30 frames contain the
        covariance convergence, and short tracks / rapid size changes are
        physically non-steady, so the hard assert is on LONG tracks
        (>= 100 output frames): a replay bug (e.g. missed update) makes every
        window of even long tracks fail."""
        bad = 0
        checked = 0
        long_bad = 0
        long_checked = 0
        all_s = []
        for tid, series in self.sig_series.items():
            series.sort()
            r = np.asarray([s for _, s in series], dtype=np.float64)
            r = r[~np.isnan(r)]
            if len(r) < 10:
                continue
            all_s.extend(r.tolist())
            checked += 1
            ok = False
            for i in range(0, len(r) - 9, 5):
                w = r[i:i + 10]
                med = np.median(w)
                if med > 0 and (np.percentile(w, 90) - np.percentile(w, 10)) / med < 0.05:
                    ok = True
                    break
            if not ok:
                bad += 1
                if len(r) >= 100:
                    long_bad += 1
            elif len(r) >= 100:
                long_checked += 1
        p99 = float(np.percentile(all_s, 99)) if all_s else float("nan")
        return checked, bad, long_checked, long_bad, p99


class ReplayAgg(object):
    """Per-dataset aggregation of per-sequence replay sanity stats."""

    def __init__(self):
        self.n_updates = 0
        self.n_init = 0
        self.n_dets = 0
        self.n_reinit = 0
        self.mean_checks = 0
        self.mean_diffs = []
        self.steady = [0, 0]       # (checked, bad)
        self.long_steady = [0, 0]  # (long_checked, long_bad)
        self.sig_p99 = 0.0

    def add(self, replay):
        self.n_updates += replay.n_updates
        self.n_init += replay.n_init
        self.n_reinit += replay.n_reinit
        self.mean_checks += replay.mean_checks
        self.mean_diffs.extend(replay.mean_diffs)
        checked, bad, lc, lb, p99 = replay.sigma_steady_check()
        self.steady[0] += checked
        self.steady[1] += bad
        self.long_steady[0] += lc
        self.long_steady[1] += lb
        self.sig_p99 = max(self.sig_p99, p99)


# --------------------------------------------------------------------------
# Per-frame features
# --------------------------------------------------------------------------
def kmc_arrays(det_ids, det_arr, prev_ids, prev_arr, prev2_ids, prev2_arr):
    """Kinematic features aligned to det_ids: (cosv, rvv, vobsn) arrays with
    NaN where undefined (missing F-2/F-1 output or near-zero speed)."""
    n = len(det_ids)
    cosv = np.full(n, np.nan)
    rvv = np.full(n, np.nan)
    vobsn = np.full(n, np.nan)
    if n == 0:
        return cosv, rvv, vobsn
    det_ids = np.asarray(det_ids)
    prev_ids = np.asarray(prev_ids)
    prev2_ids = np.asarray(prev2_ids)
    cF = det_arr[:, :2] + det_arr[:, 2:] / 2.0          # (n, 2)
    pidx = np.searchsorted(prev_ids, det_ids)
    m1 = pidx < len(prev_ids)
    if m1.any():
        m1 = m1.copy()
        m1[m1] = np.asarray(prev_ids)[pidx[m1]] == det_ids[m1]
    if m1.any():
        c1 = prev_arr[pidx[m1], :2] + prev_arr[pidx[m1], 2:] / 2.0
        vobs = cF[m1] - c1
        n1 = np.linalg.norm(vobs, axis=1)
        d1 = np.sqrt(prev_arr[pidx[m1], 2] ** 2 + prev_arr[pidx[m1], 3] ** 2)
        vobsn[m1] = n1 / np.maximum(d1, 1e-6)
        p2idx = np.searchsorted(prev2_ids, det_ids)
        m2 = p2idx < len(prev2_ids)
        if m2.any():
            m2 = m2.copy()
            m2[m2] = np.asarray(prev2_ids)[p2idx[m2]] == det_ids[m2]
        full = m1 & m2
        if full.any():
            full_m1 = full[m1]              # project onto the m1 subspace
            c2 = prev2_arr[p2idx[full], :2] + prev2_arr[p2idx[full], 2:] / 2.0
            vhist = c1[full_m1] - c2
            n2 = np.linalg.norm(vhist, axis=1)
            v1 = n1[full_m1]
            # near-zero speeds are numerical noise (still targets): undefined
            d2 = np.sqrt(prev2_arr[p2idx[full], 2] ** 2 +
                         prev2_arr[p2idx[full], 3] ** 2)
            d1m = np.sqrt(prev_arr[pidx[m1], 2] ** 2 +
                          prev_arr[pidx[m1], 3] ** 2)[full_m1]
            zero = (v1 < 1e-3 * d1m) | (n2 < 1e-3 * d2)
            ok = (v1 > 0) & (n2 > 0) & ~zero
            with np.errstate(invalid="ignore", divide="ignore"):
                cosv[full] = np.where(
                    ok,
                    (vobs[full_m1] * vhist).sum(axis=1) / (v1 * n2),
                    np.nan)
                rvv[full] = np.where(
                    ok,
                    np.maximum(v1, n2) / np.minimum(v1, n2),
                    np.nan)
    return cosv, rvv, vobsn


def neighbor_counts(box_arr):
    """N_neighbor(i) = # other boxes in box_arr overlapping box i."""
    if box_arr.shape[0] == 0:
        return np.zeros(0, dtype=int)
    iou = iou_matrix(box_arr, box_arr)
    np.fill_diagonal(iou, 0.0)
    return (iou > 0).sum(axis=1)


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def quantile_table(vals):
    """N, mean, min, P25, P50, P75, P90, P99, max of finite values."""
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [""] * 8
    q = np.percentile(a, [0, 25, 50, 75, 90, 99, 100])
    return ["%d" % len(a), "%.4f" % a.mean()] + ["%.4f" % x for x in q]


def make_grid(neg):
    """Threshold grid: neg quantiles with a dense 0.99-1.0 tail so the
    FPR=1% region is well resolved."""
    a = np.asarray(neg, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.array([0.0, 1.0])
    return np.unique(np.concatenate([
        np.quantile(a, np.linspace(0.0, 0.90, 50)),
        np.quantile(a, np.linspace(0.90, 0.99, 100)),
        np.quantile(a, np.linspace(0.99, 1.0, 800))]))


def roc_curve(pos, neg, direction):
    """Trigger = v < t ('low') or v > t ('high'). Returns ascending
    (fpr, tpr) on a quantile grid of neg via searchsorted."""
    p = np.asarray(pos, dtype=np.float64)
    p = p[~np.isnan(p)]
    neg_a = np.asarray(neg, dtype=np.float64)
    neg_a = neg_a[~np.isnan(neg_a)]
    if len(p) == 0 or len(neg_a) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    grid = make_grid(neg_a)
    sneg = np.sort(neg_a)
    if direction == "low":
        fpr = np.searchsorted(sneg, grid, side="left") / len(sneg)
        tpr = (p[:, None] < grid[None, :]).mean(axis=0)
    else:
        fpr = 1.0 - np.searchsorted(sneg, grid, side="right") / len(sneg)
        tpr = (p[:, None] > grid[None, :]).mean(axis=0)
    if direction == "high":
        fpr = fpr[::-1]
        tpr = tpr[::-1]
    return fpr, tpr


def tpr_at_fpr(fpr, tpr, target):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    if len(fpr) < 2:
        return float("nan")
    if fpr[0] > target:
        return float("nan")
    if fpr[-1] < target:
        return float(tpr[-1])
    return float(np.interp(target, fpr, tpr))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = float(k) / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (max(0.0, (c - w) / d), min(1.0, (c + w) / d))


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return (float("nan"), float("nan"))
    r, pv = stats.spearmanr(a[m], b[m])
    return (float(r), float(pv))


def verdict_line(tpr):
    if tpr >= 0.25:
        return "SUPPORTED"
    if tpr >= 0.10:
        return "PARTIAL"
    return "NOT SUPPORTED"


def fmt_rate(x):
    if x != x:
        return "nan"
    return "%.4f%%" % (100.0 * x)


# --------------------------------------------------------------------------
# Per-sequence collection
# --------------------------------------------------------------------------
class DatasetCtx(object):
    def __init__(self, ds):
        self.ds = ds
        self.events = load_events(ds)
        self.ev_frames = defaultdict(set)
        self.ev_by_frame = defaultdict(list)
        for r, cls in self.events:
            s, F = r["seq"], int(r["frame"])
            self.ev_frames[s].add(F)
            self.ev_by_frame[(s, F)].append((int(r["track_id"]), cls, r))
        self.event_rows = []          # one dict per switch event
        self.n_no_box = 0
        # V6 landmark counters (frames WITH active tracks only)
        self.n_frames = 0
        self.n_det = 0
        self.det_margin = Counter()
        self.frames_any_top2_0_2 = 0
        # group C (normal frames) trigger counters for the geometric gate
        self.c_n = 0
        self.c_trig = Counter()
        # negative-sample pools (per mechanism; freed after dataset analysis)
        self.neg_kmc = []
        self.neg_kf = []
        self.neg_den = []

    def add_event_row(self, seq, F, r, cls, tid, feats):
        row = {"ds": self.ds, "seq": seq, "frame": F, "type": r["type"],
               "class": cls, "track_id": tid, "gt_id_new": r["gt_id_new"],
               "old_hid": r["old_hid"]}
        row.update(feats)
        self.event_rows.append(row)


def collect_sequence(ctx, seq, frames):
    ds = ctx.ds
    evf = ctx.ev_frames.get(seq, set())
    replay = KfReplayState(ctx.kf)
    for F in sorted(frames):
        dets = frames[F]
        det_arr, det_ids = boxes_array(dets)
        n_det = len(det_ids)
        in_ev = F in evf
        cur_evs = ctx.ev_by_frame.get((seq, F))

        prev = frames.get(F - 1)
        prev2 = frames.get(F - 2)
        if prev is None:
            if not in_ev:
                ctx.c_n += n_det                   # group C: never triggered
            if in_ev:
                for tid, cls, r in cur_evs:
                    ctx.n_no_box += 1
                    ctx.add_event_row(seq, F, r, cls, tid, {
                        "top1": None, "top2": None, "margin": None,
                        "no_box": 1, "cos_theta": None, "r_v": None,
                        "v_obs_norm": None, "sigma_norm": None,
                        "n_neighbor": None, "geom_020": None})
            replay.step(F, dets)
            continue

        # V6-landmark counters follow the V6 definition: frames WITH active
        # tracks only
        ctx.n_frames += 1
        ctx.n_det += n_det

        prev_arr, prev_ids = boxes_array(prev)
        prev2_arr, prev2_ids = boxes_array(prev2) if prev2 else (np.zeros((0, 4)), [])
        pred_boxes = {tid: extrapolate_box(b, prev2.get(tid) if prev2 else None)
                      for tid, b in prev.items()}
        pred_arr, _ = boxes_array(pred_boxes)
        iou = iou_matrix(det_arr, pred_arr)
        top1v, top2v, marginv = _top1_top2_margin(iou)
        trig = _trig_matrix(top1v, top2v, marginv)   # (N, 5)

        # V6 landmarks
        ctx.frames_any_top2_0_2 += int((top2v >= TOP2_TH).any())
        for e in EPS_IDX:
            ctx.det_margin[e] += int((marginv < e).sum())

        # per-mechanism features
        cosv, rvv, vobsn = kmc_arrays(det_ids, det_arr, prev_ids, prev_arr,
                                      prev2_ids, prev2_arr)
        nn_prev = neighbor_counts(prev_arr)
        sigmas = replay.step(F, dets)
        sigma_n_arr = np.full(n_det, np.nan)
        for i, t in enumerate(det_ids):
            s = sigmas.get(t)
            if s is not None:
                h_state = s[1]
                sigma_n_arr[i] = s[0] / h_state if h_state > 0 else np.nan

        # group C (normal frames): geometric-gate trigger counts
        if not in_ev:
            ctx.c_n += n_det
            for e in EPS_IDX:
                ctx.c_trig[e] += int(trig[:, EPS_IDX[e]].sum())
            for i in range(n_det):
                if not np.isnan(cosv[i]):
                    ctx.neg_kmc.append([cosv[i], rvv[i], vobsn[i], top1v[i]])
                if not np.isnan(sigma_n_arr[i]):
                    ctx.neg_kf.append([sigma_n_arr[i], top1v[i], top2v[i], marginv[i]])
                if det_ids[i] in prev:
                    idx_p = int(np.searchsorted(prev_ids, det_ids[i]))
                    if idx_p < len(prev_ids) and prev_ids[idx_p] == det_ids[i]:
                        ctx.neg_den.append([float(nn_prev[idx_p]), top1v[i], top2v[i], marginv[i]])

        # event frames
        if cur_evs:
            idx = {t: i for i, t in enumerate(det_ids)}
            for tid, cls, r in cur_evs:
                i = idx.get(tid)
                if i is None:
                    ctx.n_no_box += 1
                    ctx.add_event_row(seq, F, r, cls, tid, {
                        "top1": None, "top2": None, "margin": None,
                        "no_box": 1, "cos_theta": None, "r_v": None,
                        "v_obs_norm": None, "sigma_norm": None,
                        "n_neighbor": None, "geom_020": None})
                    continue
                t1, t2, m = float(top1v[i]), float(top2v[i]), float(marginv[i])
                nn = float(nn_prev[int(np.searchsorted(prev_ids, det_ids[i]))]) \
                    if det_ids[i] in prev else float("nan")
                ctx.add_event_row(seq, F, r, cls, tid, {
                    "top1": t1, "top2": t2, "margin": m, "no_box": 0,
                    "cos_theta": None if np.isnan(cosv[i]) else float(cosv[i]),
                    "r_v": None if np.isnan(rvv[i]) else float(rvv[i]),
                    "v_obs_norm": None if np.isnan(vobsn[i]) else float(vobsn[i]),
                    "sigma_norm": None if np.isnan(sigma_n_arr[i]) else float(sigma_n_arr[i]),
                    "n_neighbor": None if np.isnan(nn) else nn,
                    "geom_020": bool(trig[i, EPS_IDX[EPS0_OR]])})
    return replay


# --------------------------------------------------------------------------
# Mechanism analysis
# --------------------------------------------------------------------------
def kmc_trigger_rate(cos, rv, top1, thr_c, thr_r):
    """Trigger = (top1>=0.2) & (cos<thr_c | rv>thr_r); NaN -> False."""
    if len(cos) == 0:
        return 0.0
    c = np.asarray(cos, dtype=np.float64)
    r = np.asarray(rv, dtype=np.float64)
    t = np.asarray(top1, dtype=np.float64)
    return float(((t >= TOP1_TH) & ((c < thr_c) | (r > thr_r))).mean())


def analyze_kmc(p_kmc, neg):
    """KMC kinematic gate. Returns dict of results + summary rows."""
    p_cos = np.asarray([x[0] for x in p_kmc], dtype=np.float64)
    p_rv = np.asarray([x[1] for x in p_kmc], dtype=np.float64)
    p_vobs = np.asarray([x[2] for x in p_kmc], dtype=np.float64)
    p_top1 = np.asarray([x[3] for x in p_kmc], dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    n_cos, n_rv, n_vobs, n_top1 = n[:, 0], n[:, 1], n[:, 2], n[:, 3]
    res = {"n_pos": len(p_cos)}
    log("   KMC: feature medians  pos  cos=%.3f r_v=%.2f v_obs_norm=%.4f | "
        "neg  cos=%.3f r_v=%.2f v_obs_norm=%.4f"
        % (np.nanmedian(p_cos), np.nanmedian(p_rv), np.nanmedian(p_vobs),
           np.nanmedian(n_cos), np.nanmedian(n_rv), np.nanmedian(n_vobs)))
    for name, pv, nv, dr in (("cos_theta", p_cos, n_cos, "low"),
                             ("r_v", p_rv, n_rv, "high"),
                             ("v_obs_norm", p_vobs, n_vobs, "high")):
        fpr, tpr = roc_curve(pv, nv, dr)
        tt = tpr_at_fpr(fpr, tpr, FPR_TARGET)
        n_pos = int((~np.isnan(pv)).sum())
        k = tt * n_pos if tt == tt else 0
        lo, hi = wilson_ci(k, n_pos)
        log("   KMC single-feature %s: TPR@FPR=1%%=%s (Wilson [%.3f, %.3f])"
            % (name, fmt_rate(tt), lo, hi))
    best = None
    for tc in GRID_COS:
        for tr in GRID_RV:
            fpr = kmc_trigger_rate(n_cos, n_rv, n_top1, tc, tr)
            tpr = kmc_trigger_rate(p_cos, p_rv, p_top1, tc, tr)
            if fpr <= FPR_TARGET and (best is None or tpr > best[2]):
                best = (tc, tr, tpr, fpr)
    res["best"] = best
    if best:
        res["best_tpr"] = best[2]
        res["verdict"] = verdict_line(best[2])
        log("   KMC grid best thr_c=%.2f thr_r=%.2f: TPR=%s FPR=%s | verdict %s"
            % (best[0], best[1], fmt_rate(best[2]), fmt_rate(best[3]),
               res["verdict"]))
    else:
        res["best_tpr"] = 0.0
        res["verdict"] = "NOT SUPPORTED"
        min_fpr = min(kmc_trigger_rate(n_cos, n_rv, n_top1, tc, tr)
                      for tc in GRID_COS for tr in GRID_RV)
        log("   KMC grid: no point under FPR<=1%% (grid min FPR=%.2f%%) | "
            "verdict NOT SUPPORTED" % (100.0 * min_fpr))
    return res


def analyze_kf(p_kf, neg, a_geom, c_geom):
    """KF adaptive margin. Returns dict of results + summary rows."""
    p = np.asarray(p_kf, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    sn_p, sn_n = p[:, 0], n[:, 0]
    t1p, t2p, mp = p[:, 1], p[:, 2], p[:, 3]
    t1n, t2n, mn = n[:, 1], n[:, 2], n[:, 3]
    med = float(np.nanmedian(sn_n))
    rho, pv = spearman(mp, sn_p)
    log("   KF: sigma_norm medians  S_r=%.4f normal=%.4f | Spearman(margin,"
        " sigma)=%.4f (p=%.3g)" % (np.nanmedian(sn_p), med, rho, pv))
    best_fixed = None
    for e in EPS:
        na, ta = a_geom.get(e, (0, 0))
        nc, tc = c_geom.get(e, (0, 0))
        fpr = float(tc) / nc if nc else 0.0
        tpr = float(ta) / na if na else 0.0
        if fpr <= FPR_TARGET and (best_fixed is None or tpr > best_fixed[1]):
            best_fixed = (e, tpr, fpr)
    best_adapt = None
    for e0 in GRID_EPS0:
        for al in GRID_ALPHA:
            eps = e0 * (1.0 + al * sn_p / med)
            tpr = float(np.mean((t1p >= TOP1_TH) & (t2p >= TOP2_TH) & (mp < eps)))
            epsn = e0 * (1.0 + al * sn_n / med)
            fpr = float(np.mean((t1n >= TOP1_TH) & (t2n >= TOP2_TH) & (mn < epsn)))
            if fpr <= FPR_TARGET and (best_adapt is None or tpr > best_adapt[2]):
                best_adapt = (e0, al, tpr, fpr)
    bf = best_fixed if best_fixed else (np.nan, 0.0, float("nan"))
    ba = best_adapt if best_adapt else (np.nan, np.nan, 0.0, float("nan"))
    gain = ba[2] - bf[1]
    res = {"best_adapt_tpr": ba[2], "best_fixed_tpr": bf[1],
           "gain": gain, "verdict": verdict_line(ba[2]), "n_pos": len(p)}
    log("   KF: fixed-eps best TPR=%s (eps=%.2f) | adaptive best TPR=%s "
        "(eps0=%.2f alpha=%.2f) | gain %+.2fpp"
        % (fmt_rate(bf[1]), bf[0], fmt_rate(ba[2]), ba[0], ba[1], 100.0 * gain))
    return res


def analyze_den(p_den, neg):
    """DEN local-density gamma gate. Returns dict of results."""
    p = np.asarray(p_den, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    nn_p, nn_n = p[:, 0], n[:, 0]
    t1p, t2p, mp = p[:, 1], p[:, 2], p[:, 3]
    t1n, t2n, mn = n[:, 1], n[:, 2], n[:, 3]
    log("   DEN: N_neighbor medians  S_r=%.1f normal=%.1f | N=0 share "
        "S_r=%.1f%% normal=%.1f%%"
        % (np.nanmedian(nn_p), np.nanmedian(nn_n),
           100.0 * float((nn_p == 0).mean()) if len(nn_p) else float("nan"),
           100.0 * float((nn_n == 0).mean())))
    fpr, tpr = roc_curve(nn_p, nn_n, "high")
    tt = tpr_at_fpr(fpr, tpr, FPR_TARGET)
    best = None
    for e0 in [0.15, 0.20]:
        for g in GRID_GAMMA:
            eps = e0 * np.where(nn_p > 0, g, 1.0)
            tpr = float(np.mean((t1p >= TOP1_TH) & (t2p >= TOP2_TH) & (mp < eps)))
            epsn = e0 * np.where(nn_n > 0, g, 1.0)
            fpr = float(np.mean((t1n >= TOP1_TH) & (t2n >= TOP2_TH) & (mn < epsn)))
            if fpr <= FPR_TARGET and (best is None or tpr > best[2]):
                best = (e0, g, tpr, fpr)
    base_tpr = 0.0
    for e0 in [0.15, 0.20]:
        for g in GRID_GAMMA:
            if g == 1.0:
                eps = e0 * np.ones_like(nn_p)
                tpr = float(np.mean((t1p >= TOP1_TH) & (t2p >= TOP2_TH) & (mp < eps)))
                epsn = e0 * np.ones_like(nn_n)
                fpr = float(np.mean((t1n >= TOP1_TH) & (t2n >= TOP2_TH) & (mn < epsn)))
                if fpr <= FPR_TARGET and tpr > base_tpr:
                    base_tpr = tpr
    b = best if best else (np.nan, np.nan, 0.0, float("nan"))
    gain = b[2] - base_tpr
    res = {"best_tpr": b[2], "gain": gain, "verdict": verdict_line(b[2]),
           "n_pos": len(p), "single_tpr": tt}
    log("   DEN: single-feature TPR@FPR=1%%=%s | gamma best eps0=%.2f "
        "gamma=%.2f TPR=%s (gain over gamma=1.0: %+.2fpp)"
        % (fmt_rate(tt), b[0], b[1], fmt_rate(b[2]), 100.0 * gain))
    return res


# --------------------------------------------------------------------------
# Sanity (dataset level)
# --------------------------------------------------------------------------
def run_sanity(ds, ctx, exp_counts, agg):
    # class counts == frozen event_counts_by_sequence.csv
    for c in ["S_c", "S_r", "S_h"]:
        got = sum(1 for r, cls in ctx.events if cls == c)
        if got != exp_counts[ds][c]:
            raise SystemExit("SANITY FAIL: %s class %s count %d != expected %d"
                             % (ds, c, got, exp_counts[ds][c]))
    # V6 population landmarks
    if ctx.n_frames != V6_FRAMES[ds] or ctx.n_det != V6_DETS[ds]:
        raise SystemExit("SANITY FAIL: %s population frames=%d dets=%d vs V6 %d/%d"
                         % (ds, ctx.n_frames, ctx.n_det, V6_FRAMES[ds], V6_DETS[ds]))
    for e in EPS_IDX:
        if ctx.det_margin[e] != V6_DET_MARGIN[ds][e]:
            raise SystemExit("SANITY FAIL: %s det_margin(%.2f)=%d vs V6 %d"
                             % (ds, e, ctx.det_margin[e], V6_DET_MARGIN[ds][e]))
    if ctx.frames_any_top2_0_2 != V6_FRAMES_ANY[ds]:
        raise SystemExit("SANITY FAIL: %s frames_any(top2>=0.2)=%d vs V6 %d"
                         % (ds, ctx.frames_any_top2_0_2, V6_FRAMES_ANY[ds]))
    # event keys unique, S_r gap always 1
    keys = set()
    for r, cls in ctx.events:
        k = (r["seq"], int(r["frame"]), int(r["track_id"]))
        if k in keys:
            raise SystemExit("SANITY FAIL: %s duplicate event key %s" % (ds, k))
        keys.add(k)
        if cls == "S_r" and r["gap"].strip() != "1":
            raise SystemExit("SANITY FAIL: %s S_r gap != 1 at %s" % (ds, k))
    # KF replay
    if agg.steady[0] == 0:
        raise SystemExit("SANITY FAIL: %s KF replay: no steady-state tracks" % ds)
    if agg.long_steady[0] > 0 and agg.long_steady[1] / agg.long_steady[0] > 0.2:
        raise SystemExit("SANITY FAIL: %s KF replay sigma steady: %d/%d LONG "
                         "tracks unstable" % (ds, agg.long_steady[1], agg.long_steady[0]))
    if agg.sig_p99 > 0.25:
        raise SystemExit("SANITY FAIL: %s KF replay sigma_norm P99=%.4f > 0.25"
                         % (ds, agg.sig_p99))
    if agg.n_updates + agg.n_init != agg.n_dets:
        raise SystemExit("SANITY FAIL: %s KF replay updates+init %d+%d != dets %d"
                         % (ds, agg.n_updates, agg.n_init, agg.n_dets))
    if agg.mean_diffs:
        dd = np.asarray(agg.mean_diffs)
        if np.median(dd) >= 0.05 or np.percentile(dd, 99) >= 0.2:
            raise SystemExit("SANITY FAIL: %s KF mean deviation median=%.4f p99=%.4f"
                             % (ds, np.median(dd), np.percentile(dd, 99)))
    log("   sanity %s OK: classes, V6 landmarks, event keys, KF replay "
        "(steady %d tracks, reinit %d, mean-check %d)"
        % (ds, agg.steady[0], agg.n_reinit, agg.mean_checks))


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
def write_events_csv(all_ev_rows):
    path = os.path.join(OUT, "gate_feasibility_events.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seq", "frame", "type", "class", "track_id",
                    "gt_id_new", "old_hid", "top1", "top2", "margin", "no_box",
                    "cos_theta", "r_v", "v_obs_norm", "sigma_norm",
                    "n_neighbor", "geom_020"])
        for r in all_ev_rows:
            w.writerow([r["ds"], r["seq"], r["frame"], r["type"], r["class"],
                        r["track_id"], r["gt_id_new"], r["old_hid"],
                        "" if r["top1"] is None else "%.4f" % r["top1"],
                        "" if r["top2"] is None else "%.4f" % r["top2"],
                        "" if r["margin"] is None else "%.4f" % r["margin"],
                        r["no_box"],
                        "" if r["cos_theta"] is None else "%.4f" % r["cos_theta"],
                        "" if r["r_v"] is None else "%.6f" % r["r_v"],
                        "" if r["v_obs_norm"] is None else "%.4f" % r["v_obs_norm"],
                        "" if r["sigma_norm"] is None else "%.4f" % r["sigma_norm"],
                        "" if r["n_neighbor"] is None else "%.1f" % r["n_neighbor"],
                        "" if r["geom_020"] is None else ("1" if r["geom_020"] else "0")])
    log("wrote %s (%d rows)" % (path, len(all_ev_rows)))


def write_summary_csv(rows):
    path = os.path.join(OUT, "gate_feasibility_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idea", "dataset", "threshold", "group", "n_samples",
                    "n_triggered", "rate"])
        w.writerows(rows)
    log("wrote %s (%d rows)" % (path, len(rows)))


def make_figure(res_all):
    """2x2 ROC figure: per-dataset panels + combined with light curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"kmc": "#2a78d6", "kf": "#eb6834", "den": "#1baf7a"}
    labels = {"kmc": "KMC (r_v)", "kf": "KF sigma_norm", "den": "DEN N_neighbor"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=200)
    panels = [("combined", "Combined"), ("MOT17", "MOT17"),
              ("MOT20", "MOT20"), ("SportsMOT", "SportsMOT")]
    for ax, (key, title) in zip(axes.flat, panels):
        if key == "combined":
            for ds in ("MOT17", "MOT20", "SportsMOT"):
                d = res_all.get(ds)
                if not d:
                    continue
                for idea in ("kmc", "kf", "den"):
                    rr = d.get(idea)
                    if rr and len(rr.get("fpr", [])):
                        ax.plot(rr["fpr"], rr["tpr"], color=colors[idea],
                                linewidth=1.0, alpha=0.4)
            for idea, c in colors.items():
                ax.plot([], [], color=c, linewidth=2.0, label=labels[idea])
        else:
            d = res_all.get(key)
            for idea in ("kmc", "kf", "den"):
                rr = d.get(idea) if d else None
                if rr and len(rr.get("fpr", [])):
                    ax.plot(rr["fpr"], rr["tpr"], color=colors[idea],
                            linewidth=2.0, label=labels[idea])
        ax.plot([0, 0.05], [0, 0.05], "--", color="#9a9890", linewidth=1.0)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 0.05)
        ax.set_ylim(0, 1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True, color="#e1e0d9", linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT, "gate_feasibility_roc.png")
    fig.savefig(path, dpi=200)
    log("wrote %s" % path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="gate feasibility analysis")
    ap.add_argument("--datasets", default="mot17,mot20,sportsmot",
                    help="comma list of mot17|mot20|sportsmot")
    ap.add_argument("--ideas", default="kmc,kf,den",
                    help="comma list of kmc|kf|den")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    ds_map = {"mot17": "MOT17", "mot20": "MOT20", "sportsmot": "SportsMOT"}
    datasets = [ds_map[d.strip().lower()] for d in args.datasets.split(",")]
    ideas = [d.strip().lower() for d in args.ideas.split(",")]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit("unknown dataset %s" % d)

    os.makedirs(OUT, exist_ok=True)
    log("analysis start | datasets=%s ideas=%s FPR_TARGET=%.2f"
        % (datasets, ideas, FPR_TARGET))
    kf_cls = load_kalman_filter()
    exp_counts = load_by_sequence_counts()

    all_ev_rows = []
    summary_rows = []
    res_all = {}
    geom_a = {}          # group A (S_r) triggers per eps, from event rows
    for ds in datasets:
        t_ds = time.time()
        ctx = DatasetCtx(ds)
        ctx.kf = kf_cls()
        agg = ReplayAgg()
        tr_dir = os.path.join(ROOT, "YOLOX_outputs", EXPNS[ds], "track_results")
        for fn in sorted(os.listdir(tr_dir)):
            if not fn.endswith(".txt"):
                continue
            seq = fn[:-4]
            frames = load_frames(os.path.join(tr_dir, fn))
            replay = collect_sequence(ctx, seq, frames)
            agg.n_dets += sum(len(v) for v in frames.values())
            agg.add(replay)
            del frames
        run_sanity(ds, ctx, exp_counts, agg)
        log("   %s: events=%d no_box=%d | neg pools kmc=%d kf=%d den=%d | %.1fs"
            % (ds, len(ctx.event_rows), ctx.n_no_box, len(ctx.neg_kmc),
               len(ctx.neg_kf), len(ctx.neg_den), time.time() - t_ds))

        # geometric gate baseline rows (groups A/B from event rows, C from
        # the normal-frame counters) -- same math as the mechanism gates
        sr = [r for r in ctx.event_rows if r["class"] == "S_r"]
        sb = [r for r in ctx.event_rows if r["class"] in ("S_c", "S_h")]
        a_trig = {e: 0 for e in EPS}
        b_trig = {e: 0 for e in EPS}
        for r in sr:
            if r["top1"] is not None:
                for e in EPS:
                    a_trig[e] += int(r["top1"] >= TOP1_TH and r["top2"] >= TOP2_TH
                                     and r["top1"] - r["top2"] < e)
        for r in sb:
            if r["top1"] is not None:
                for e in EPS:
                    b_trig[e] += int(r["top1"] >= TOP1_TH and r["top2"] >= TOP2_TH
                                     and r["top1"] - r["top2"] < e)
        geom_a[ds] = {e: (len(sr), a_trig[e]) for e in EPS}
        for e in EPS:
            summary_rows.append(["geom", ds, "eps=%.2f" % e, "A_S_r",
                                 len(sr), a_trig[e],
                                 float(a_trig[e]) / len(sr) if sr else 0.0])
            summary_rows.append(["geom", ds, "eps=%.2f" % e, "B_S_cuSh",
                                 len(sb), b_trig[e],
                                 float(b_trig[e]) / len(sb) if sb else 0.0])
            summary_rows.append(["geom", ds, "eps=%.2f" % e, "C_normal",
                                 ctx.c_n, ctx.c_trig[e],
                                 float(ctx.c_trig[e]) / ctx.c_n if ctx.c_n else 0.0])
        log("   geom gate %s: A(S_r) eps=0.20 -> %.4f%% | C normal eps=0.20 -> %.4f%%"
            % (ds, 100.0 * (a_trig[0.20] / len(sr) if sr else 0.0),
               100.0 * (ctx.c_trig[0.20] / ctx.c_n if ctx.c_n else 0.0)))

        # mechanism analysis
        pos = {"kmc": [], "kf": [], "den": []}
        for r in sr:
            pos["kmc"].append([r["cos_theta"] if r["cos_theta"] is not None else np.nan,
                               r["r_v"] if r["r_v"] is not None else np.nan,
                               r["v_obs_norm"] if r["v_obs_norm"] is not None else np.nan,
                               r["top1"] if r["top1"] is not None else np.nan])
            pos["kf"].append([r["sigma_norm"] if r["sigma_norm"] is not None else np.nan,
                              r["top1"] if r["top1"] is not None else np.nan,
                              r["top2"] if r["top2"] is not None else np.nan,
                              r["margin"] if r["margin"] is not None else np.nan])
            pos["den"].append([r["n_neighbor"] if r["n_neighbor"] is not None else np.nan,
                               r["top1"] if r["top1"] is not None else np.nan,
                               r["top2"] if r["top2"] is not None else np.nan,
                               r["margin"] if r["margin"] is not None else np.nan])
        res_all.setdefault(ds, {})
        if "kmc" in ideas:
            r = analyze_kmc(pos["kmc"], ctx.neg_kmc)
            r["fpr"], r["tpr"] = roc_curve(np.asarray([x[1] for x in pos["kmc"]]),
                                           np.asarray([x[1] for x in ctx.neg_kmc]),
                                           "high")
            res_all[ds]["kmc"] = {"best_tpr": r["best_tpr"], "verdict": r["verdict"]}
            if r["best"]:
                summary_rows.append(["kmc", ds,
                                     "thr_c=%.2f/thr_r=%.2f" % (r["best"][0], r["best"][1]),
                                     "event_S_r", len(pos["kmc"]),
                                     int(round(r["best"][2] * len(pos["kmc"]))),
                                     r["best"][2]])
                summary_rows.append(["kmc", ds, "thr_c=%.2f/thr_r=%.2f" % (r["best"][0], r["best"][1]),
                                     "normal", len(ctx.neg_kmc),
                                     int(round(r["best"][3] * len(ctx.neg_kmc))),
                                     r["best"][3]])
            del ctx.neg_kmc
        if "kf" in ideas:
            r = analyze_kf(pos["kf"], ctx.neg_kf, geom_a[ds],
                           {e: (ctx.c_n, ctx.c_trig[e]) for e in EPS})
            res_all[ds]["kf"] = {"best_tpr": r["best_adapt_tpr"], "verdict": r["verdict"],
                                 "gain": r["gain"]}
            summary_rows.append(["kf", ds, "best_adaptive@FPR<=1%", "event_S_r",
                                 r["n_pos"], int(round(r["best_adapt_tpr"] * r["n_pos"])),
                                 r["best_adapt_tpr"]])
            summary_rows.append(["kf", ds, "best_fixed@FPR<=1%", "event_S_r",
                                 r["n_pos"], int(round(r["best_fixed_tpr"] * r["n_pos"])),
                                 r["best_fixed_tpr"]])
            fpr, tpr = roc_curve(np.asarray([x[0] for x in pos["kf"]]),
                                 np.asarray([x[0] for x in ctx.neg_kf]), "high")
            res_all[ds]["kf"]["fpr"], res_all[ds]["kf"]["tpr"] = fpr, tpr
            del ctx.neg_kf
        if "den" in ideas:
            r = analyze_den(pos["den"], ctx.neg_den)
            res_all[ds]["den"] = {"best_tpr": r["best_tpr"], "verdict": r["verdict"],
                                  "gain": r["gain"]}
            summary_rows.append(["den", ds, "best_gamma@FPR<=1%", "event_S_r",
                                 r["n_pos"], int(round(r["best_tpr"] * r["n_pos"])),
                                 r["best_tpr"]])
            fpr, tpr = roc_curve(np.asarray([x[0] for x in pos["den"]]),
                                 np.asarray([x[0] for x in ctx.neg_den]), "high")
            res_all[ds]["den"]["fpr"], res_all[ds]["den"]["tpr"] = fpr, tpr
            del ctx.neg_den
        all_ev_rows.extend(ctx.event_rows)

    # combined rows: sum n and n_triggered over datasets for identical
    # (idea, threshold, group) rows
    comb_rows = {}
    for row in summary_rows:
        key = (row[0], row[2], row[3])
        if key not in comb_rows:
            comb_rows[key] = [0, 0]
        comb_rows[key][0] += row[4]
        comb_rows[key][1] += row[5]
    for key, (n, t) in sorted(comb_rows.items()):
        summary_rows.append([key[0], "combined", key[1], key[2], n, t,
                             float(t) / n if n else 0.0])

    write_events_csv(all_ev_rows)
    write_summary_csv(summary_rows)
    if not args.no_figure:
        make_figure(res_all)
    log("analysis done")


if __name__ == "__main__":
    main()
