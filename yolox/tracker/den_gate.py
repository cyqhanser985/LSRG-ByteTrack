"""DEN (local-density) gate instrumentation -- pure functions, no state.

Offline semantics replicated from `new project/code/analysis.py` (V8, frozen):
  - neighbor_counts : N(i) = # other F-1 output boxes overlapping box i (IoU > 0)
  - gate features    : per-detection top1/top2/margin over the
                       (det x F-1-predicted-box) IoU matrix
  - trigger          : top1 >= 0.2 and top2 >= 0.2 and margin < eps0 * gamma(N),
                       gamma(0) = 1.0, gamma(N>0) = gamma

Alert mode only: this module never modifies the association cost matrix.
The tracker holds the F-1 snapshot and the per-video log accumulators.
"""
import numpy as np

from yolox.tracker import matching

TOP1_TH = 0.2
TOP2_TH = 0.2


def snapshot_from_outputs(output_stracks, min_box_area):
    """Snapshot of the frames' output boxes, replicating mot_evaluator's
    post-filter (vertical w/h > 1.6, min_box_area) plus the 0.1 rounding of
    write_results, so the snapshot matches the frozen track_results files the
    offline analysis consumed.

    Returns (ids int64 (k,), tlbrs float64 (k,4)); empty-safe.
    """
    ids = []
    tlbrs = []
    for t in output_stracks:
        tlwh = t.tlwh
        if tlwh[2] * tlwh[3] > min_box_area and tlwh[2] / tlwh[3] <= 1.6:
            x1, y1, w, h = np.round(tlwh, 1)
            tlbrs.append((x1, y1, x1 + w, y1 + h))
            ids.append(t.track_id)
    if not ids:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4))
    return (np.asarray(ids, dtype=np.int64),
            np.asarray(tlbrs, dtype=np.float64))


def neighbor_counts(tlbrs):
    """N_neighbor per box: count of other boxes overlapping it (IoU > 0).
    Replicates analysis.py neighbor_counts() (L476-482).
    """
    k = len(tlbrs)
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    iou = matching.ious(tlbrs, tlbrs).astype(np.float64)
    np.fill_diagonal(iou, 0.0)
    return (iou > 0).sum(axis=1).astype(np.int64)


def gate_features(dists, pool_ids, snap_ids, snap_N, eps0, gamma):
    """Gate features for the first-stage association.

    dists   : (R, M) unfused cost matrix (1 - IoU), rows = strack_pool,
              cols = detections -- MUST be computed before fuse_score.
    pool_ids: (R,) track ids of strack_pool.
    snap_ids, snap_N: F-1 snapshot ids and their neighbor counts.
    Candidate rows are restricted to pool tracks present in the snapshot
    (offline pred set == F-1 output boxes); lost tracks (no F-1 output)
    are excluded and get no gate.

    Returns dict(trig, top1_row, top1_tid, N, top1v, top2v, marginv, epsv)
    or None when there is nothing to gate (empty pool/dets/no candidates).
    """
    R, M = dists.shape
    if R == 0 or M == 0:
        return None
    row_in_pool = np.where(np.isin(pool_ids, snap_ids))[0]
    r = len(row_in_pool)
    if r == 0:
        return None
    dr = dists[row_in_pool]
    if r == 1:
        # single candidate track: top2 = 0 (offline _top1_top2_margin, m==1)
        c1 = dr[0].copy()
        c2 = np.zeros(M, dtype=np.float64)
        top1_sub = np.zeros(M, dtype=np.int64)
    else:
        idx = np.argpartition(dr, 1, axis=0)[:2]           # (2, M) two smallest
        v = np.take_along_axis(dr, idx, axis=0)            # (2, M), unordered
        vs = np.sort(v, axis=0)                            # c1 = min, c2 = 2nd min
        c1, c2 = vs[0], vs[1]
        top1_sub = np.where(v[0] == vs[0], idx[0], idx[1])  # row of the minimum
    top1_row = row_in_pool[top1_sub]                       # full-dists row index
    top1v = 1.0 - c1
    top2v = 1.0 - c2
    marginv = c2 - c1
    n_by_tid = dict(zip(snap_ids.tolist(), snap_N.tolist()))
    N = np.asarray([n_by_tid.get(int(t), 0) for t in pool_ids[top1_row]],
                   dtype=np.int64)
    epsv = eps0 * np.where(N > 0, gamma, 1.0)              # gamma(0)=1, gamma(N>0)=gamma
    trig = (top1v >= TOP1_TH) & (top2v >= TOP2_TH) & (marginv < epsv)
    return {"trig": trig, "top1_row": top1_row, "top1_tid": pool_ids[top1_row],
            "N": N, "top1v": top1v, "top2v": top2v, "marginv": marginv,
            "epsv": epsv}


def candidate_rows(gf, cur_frame, det_tlbrs=None):
    """Expand gate features into per-detection log rows.

    Logs every candidate detection (top1>=0.2 and top2>=0.2, the full gate
    decision population) with its raw features, so the eps0 x gamma grid can
    be re-evaluated post-hoc without re-running the tracker. det_tlbrs
    (M, 4) optionally records the detection box for box-level alignment.

    Returns (rows, n_candidate, n_triggered).
    """
    cand = (gf["top1v"] >= TOP1_TH) & (gf["top2v"] >= TOP2_TH)
    idxs = np.nonzero(cand)[0]
    tid = gf["top1_tid"]
    trig = gf["trig"]
    rows = []
    for j in idxs:
        row = [cur_frame, int(tid[j]), float(gf["top1v"][j]), float(gf["top2v"][j]),
               float(gf["marginv"][j]), int(gf["N"][j]), float(gf["epsv"][j]),
               int(bool(trig[j]))]
        if det_tlbrs is not None:
            row.extend(float(v) for v in det_tlbrs[j])
        rows.append(tuple(row))
    return rows, int(cand.sum()), int(trig.sum())
