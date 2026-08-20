# comprehensive_class_dataset_breakdown.py
# ==============================================================================
# Comprehensive Multi-Dataset & Failure-Class Risk Breakdown Engine
#
# Part of LSRG-ByteTrack Research Workspace.
#
# Computes exact empirical evaluation tables across:
#   - Datasets: MOT17 (MOT17-half), MOT20, SportsMOT, and Full Unified Benchmark
#   - Failure Classes: S_c (Cold Start), S_r (Active Takeover), S_h (History Reactivation), and Total
#   - Evaluation Scopes: Benchmark-Specific (Intra-dataset) & Unified (Global Negatives)
#   - Aggregation Operators: Noisy-OR, Max Baseline, Generalized Power Mean, OWA
#   - Single Causal Features: r_weak, r_comp, r_swap
#   - Metric Grids: TPR \in [60%, 100%] in 5% steps + 99% + 100%, pAUC[0.60, 1.0], pAUC[0.95, 1.0],
#                   and fine-grained tail percentiles (Min, P01, P05, P10, P50, P90, P99, Score >= 0.99)
# ==============================================================================

import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as A

ROOT = A._repo_root()
TAXONOMY_DIR = os.path.join(ROOT, "research", "taxonomy")
REPORTS_DIR = os.path.join(ROOT, "research", "reports")

_t0 = time.time()


def log(msg):
    print("[+%6.1fs] %s" % (time.time() - _t0, msg))


# --------------------------------------------------------------------------
# Aggregation Operators (Vectorized [N, 3] -> [N,])
# --------------------------------------------------------------------------
def _chunked_apply(fn, r, chunk_size=200000):
    n = len(r)
    if n <= chunk_size:
        return fn(r)
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        out[i:end] = fn(r[i:end])
    return out


def agg_max(r, params=None):
    return _chunked_apply(lambda x: np.max(x, axis=1).astype(np.float32), r)


def agg_power_mean(r, params):
    p = float(params.get("p", 4.0))
    w = np.asarray(params.get("w", [1/3., 1/3., 1/3.]), dtype=np.float32)
    def _calc(sub):
        weighted_sum = np.sum(w[None, :] * np.power(np.clip(sub, 0.0, 1.0), p), axis=1)
        return np.clip(np.power(weighted_sum, 1.0 / p), 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)


def agg_noisy_or(r, params):
    w = np.asarray(params.get("w", [1.0, 1.0, 1.0]), dtype=np.float32)
    def _calc(sub):
        term1 = 1.0 - w[0] * sub[:, 0]
        term2 = 1.0 - w[1] * sub[:, 1]
        term3 = 1.0 - w[2] * sub[:, 2]
        return np.clip(1.0 - term1 * term2 * term3, 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)


def agg_owa(r, params):
    v = np.asarray(params.get("v", [0.8, 0.15, 0.05]), dtype=np.float32)
    def _calc(sub):
        r_sorted = np.sort(sub, axis=1)[:, ::-1]
        return np.clip(np.sum(r_sorted * v[None, :], axis=1), 0.0, 1.0).astype(np.float32)
    return _chunked_apply(_calc, r)


MODEL_FNS = {
    "Max": agg_max,
    "PowerMean": agg_power_mean,
    "Noisy-OR": agg_noisy_or,
    "OWA": agg_owa
}


def fit_model_parameters(name, train_pos):
    if name == "Max":
        return {}
    elif name == "PowerMean":
        best_val = -1.0
        best_p = 4.0
        best_w = np.array([1/3., 1/3., 1/3.])
        grid_p = [1.0, 2.0, 4.0, 8.0, 16.0]
        weights_pool = [
            np.array([1/3., 1/3., 1/3.]),
            np.array([0.50, 0.25, 0.25]),
            np.array([0.25, 0.50, 0.25]),
            np.array([0.25, 0.25, 0.50]),
            np.array([0.60, 0.20, 0.20]),
            np.array([0.40, 0.40, 0.20]),
        ]
        for p in grid_p:
            for w in weights_pool:
                scores = agg_power_mean(train_pos, {"p": p, "w": w})
                min_s = scores.min()
                if min_s > best_val:
                    best_val = min_s
                    best_p = p
                    best_w = w
        return {"p": best_p, "w": best_w, "opt_min": best_val}
    elif name == "Noisy-OR":
        best_val = -1.0
        best_w = np.array([1.0, 1.0, 1.0])
        w_grid = [0.4, 0.7, 1.0]
        for w1 in w_grid:
            for w2 in w_grid:
                for w3 in w_grid:
                    w = np.array([w1, w2, w3])
                    scores = agg_noisy_or(train_pos, {"w": w})
                    min_s = scores.min()
                    if min_s > best_val:
                        best_val = min_s
                        best_w = w
        return {"w": best_w, "opt_min": best_val}
    elif name == "OWA":
        best_val = -1.0
        best_v = np.array([1.0, 0.0, 0.0])
        v_pool = [
            np.array([1.00, 0.00, 0.00]),
            np.array([0.80, 0.15, 0.05]),
            np.array([0.60, 0.30, 0.10]),
            np.array([1/3., 1/3., 1/3.]),
        ]
        for v in v_pool:
            scores = agg_owa(train_pos, {"v": v})
            min_s = scores.min()
            if min_s > best_val:
                best_val = min_s
                best_v = v
        return {"v": best_v, "opt_min": best_val}


# --------------------------------------------------------------------------
# Evaluation Helper
# --------------------------------------------------------------------------
TPR_STEPS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]


def evaluate_curve(pos_scores, neg_scores):
    pos_arr = np.asarray(pos_scores, dtype=np.float64)
    neg_arr = np.asarray(neg_scores, dtype=np.float64)
    n_pos = len(pos_arr)
    n_neg = len(neg_arr)

    if n_pos == 0 or n_neg == 0:
        return None

    step_metrics = {}
    for tpr in TPR_STEPS:
        q = 100.0 * (1.0 - tpr)
        thr = float(np.percentile(pos_arr, q)) if tpr < 1.0 else float(pos_arr.min())
        fpr = float((neg_arr >= thr).mean())
        step_metrics[tpr] = {"thr": thr, "fpr": fpr}

    # Dense curve for plotting [0.60, 1.00]
    tpr_dense = np.linspace(0.60, 1.0, 101)
    q_dense = 100.0 * (1.0 - tpr_dense)
    thr_dense = np.percentile(pos_arr, q_dense)
    fpr_dense = np.array([(neg_arr >= t).mean() for t in thr_dense], dtype=np.float64)

    norm_pauc_60 = float(np.trapz(1.0 - fpr_dense, tpr_dense) / 0.40)
    mask_95 = tpr_dense >= 0.95
    norm_pauc_95 = float(np.trapz(1.0 - fpr_dense[mask_95], tpr_dense[mask_95]) / 0.05)

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "step_metrics": step_metrics,
        "tpr_dense": tpr_dense.tolist(),
        "fpr_dense": fpr_dense.tolist(),
        "norm_pauc_60": norm_pauc_60,
        "norm_pauc_95": norm_pauc_95,
        "min_score": float(pos_arr.min()),
        "p01_score": float(np.percentile(pos_arr, 1)),
        "p05_score": float(np.percentile(pos_arr, 5)),
        "p10_score": float(np.percentile(pos_arr, 10)),
        "p50_score": float(np.percentile(pos_arr, 50)),
        "p90_score": float(np.percentile(pos_arr, 90)),
        "p99_score": float(np.percentile(pos_arr, 99)),
        "mean_score": float(pos_arr.mean()),
        "std_score": float(pos_arr.std()),
        "frac_ge_099": float((pos_arr >= 0.99).mean()),
        "frac_ge_095": float((pos_arr >= 0.95).mean()),
    }


def main():
    log("Loading precomputed events and negatives...")
    events_npz = os.path.join(TAXONOMY_DIR, "risk_features_events.npz")
    negatives_npz = os.path.join(TAXONOMY_DIR, "risk_features_negatives.npz")

    pos_data = np.load(events_npz)
    neg_data = np.load(negatives_npz)

    pos_risk = pos_data["risk_matrix"]       # [4713, 3]
    pos_raw = pos_data["raw_matrix"]         # [4713, 3]
    pos_seqs = pos_data["seq"]
    pos_ds = pos_data["dataset"]
    pos_cls = pos_data["class_labels"]

    neg_risk = neg_data["risk_matrix"]       # [1647180, 3]
    neg_seq_names = neg_data["seq_names"]
    neg_seq_datasets = neg_data["seq_datasets"]
    neg_seq_starts = neg_data["seq_starts"]
    neg_seq_ends = neg_data["seq_ends"]

    neg_seq_slices = {}
    neg_dataset_masks = {"MOT17": np.zeros(len(neg_risk), dtype=bool),
                         "MOT20": np.zeros(len(neg_risk), dtype=bool),
                         "SportsMOT": np.zeros(len(neg_risk), dtype=bool)}

    for name, ds, st, en in zip(neg_seq_names, neg_seq_datasets, neg_seq_starts, neg_seq_ends):
        ds_str = str(ds)
        neg_seq_slices[(ds_str, str(name))] = (int(st), int(en))
        neg_dataset_masks[ds_str][int(st):int(en)] = True

    n_events = len(pos_risk)
    n_negs = len(neg_risk)
    log(f"Loaded {n_events} positive events and {n_negs} negative detections.")

    # 5-Fold Stratification by Sequence
    unique_seqs = []
    for ds in ["MOT17", "MOT20", "SportsMOT"]:
        pos_s_set = set(pos_seqs[pos_ds == ds])
        neg_s_set = set(s for (d, s) in neg_seq_slices.keys() if d == ds)
        s_list = sorted(list(pos_s_set | neg_s_set))
        unique_seqs.append((ds, s_list))

    np.random.seed(42)
    folds = [[] for _ in range(5)]
    for ds, s_list in unique_seqs:
        shuffled = np.random.permutation(s_list)
        for i, s in enumerate(shuffled):
            folds[i % 5].append((ds, s))

    # Evaluate 5-Fold Test Out-of-Fold predictions for all models
    model_names = ["Max", "PowerMean", "Noisy-OR", "OWA"]
    oof_pos = {m: np.zeros(n_events, dtype=np.float32) for m in model_names}
    oof_neg = {m: np.zeros(n_negs, dtype=np.float32) for m in model_names}

    for name in model_names:
        log(f"Running 5-fold cross-validation for {name}...")
        for k in range(5):
            test_set = set(folds[k])
            pos_test_mask = np.array([(d, s) in test_set for d, s in zip(pos_ds, pos_seqs)])
            neg_test_mask = np.zeros(n_negs, dtype=bool)
            for item in test_set:
                sl = neg_seq_slices.get(item)
                if sl is not None:
                    neg_test_mask[sl[0]:sl[1]] = True

            pos_train_sub = pos_risk[~pos_test_mask]
            params_k = fit_model_parameters(name, pos_train_sub)

            oof_pos[name][pos_test_mask] = MODEL_FNS[name](pos_risk[pos_test_mask], params_k)
            oof_neg[name][neg_test_mask] = MODEL_FNS[name](neg_risk[neg_test_mask], params_k)

    # Prepare datasets and classes
    datasets_list = ["MOT17", "MOT20", "SportsMOT", "Overall"]
    classes_list = ["Overall", "S_c", "S_r", "S_h"]

    # Master results structure
    # master_results[dataset_scope][class_name][model_name][neg_mode]
    # neg_mode: "intra" (within dataset negatives) or "global" (all 1.65M negatives)
    master_results = {}

    for ds in datasets_list:
        master_results[ds] = {}
        for cls in classes_list:
            master_results[ds][cls] = {}
            
            # Positive mask
            if ds == "Overall":
                ds_mask = np.ones(n_events, dtype=bool)
            else:
                ds_mask = (pos_ds == ds)
            
            if cls == "Overall":
                cls_mask = np.ones(n_events, dtype=bool)
            else:
                cls_mask = (pos_cls == cls)
            
            event_mask = ds_mask & cls_mask

            for m in model_names:
                master_results[ds][cls][m] = {}
                
                # Global negative evaluation
                pos_sc = oof_pos[m][event_mask]
                neg_sc_global = oof_neg[m]
                master_results[ds][cls][m]["global"] = evaluate_curve(pos_sc, neg_sc_global)

                # Intra-dataset negative evaluation
                if ds != "Overall":
                    neg_sc_intra = oof_neg[m][neg_dataset_masks[ds]]
                    master_results[ds][cls][m]["intra"] = evaluate_curve(pos_sc, neg_sc_intra)
                else:
                    master_results[ds][cls][m]["intra"] = master_results[ds][cls][m]["global"]

            # Also evaluate single features
            master_results[ds][cls]["single_feats"] = {}
            feat_names = ["r_weak", "r_comp", "r_swap"]
            for j, fn in enumerate(feat_names):
                master_results[ds][cls]["single_feats"][fn] = {}
                pos_sc_feat = pos_risk[event_mask, j]
                
                # Global
                master_results[ds][cls]["single_feats"][fn]["global"] = evaluate_curve(pos_sc_feat, neg_risk[:, j])
                # Intra
                if ds != "Overall":
                    master_results[ds][cls]["single_feats"][fn]["intra"] = evaluate_curve(pos_sc_feat, neg_risk[neg_dataset_masks[ds], j])
                else:
                    master_results[ds][cls]["single_feats"][fn]["intra"] = master_results[ds][cls]["single_feats"][fn]["global"]

    # Save complete JSON database
    json_path = os.path.join(TAXONOMY_DIR, "class_dataset_breakdown_full.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    log(f"Saved complete JSON database: {json_path}")

    return master_results


if __name__ == "__main__":
    main()
