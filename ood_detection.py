"""OOD Detection using features, code adapted from Ghada"""
import argparse
import json
import os
import random
import re
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from scipy.linalg import pinv, cholesky, solve_triangular
from tqdm import tqdm
from sklearn.covariance import LedoitWolf

from utils import set_seed, make_run_dir, run_name

with open("cfg.json", "r") as f:
    cfg = json.load(f)

VERBOSE_EVERY_CALL = False
_MAHA_CACHE = {}

# fancy metric name
metrics = {
    "cosine": "Cosine Similarity",
    "mahalanobis": "Mahalanobis Distance",
    "mahalanobis-solve": "Mahalanobis Distance (direct solve)",
    "mahalanobis-pinv": "Mahalanobis Distance (raw pinv baseline)"
}

"""Apply SPC rules (per image)"""
def apply_spc_rules(data, mean, std, metric_name):
    """Detects SPC rule violations."""
    violations = {"Rule 1": []}

    for i in range(len(data)):
        # Rule 1 (one-sided for OOD): a point beyond the 3σ limit on the side
        # that indicates dissimilarity from the in-distribution centroid.
        # Cosine similarity: flag the LOWER tail (low similarity = OOD).
        # Mahalanobis distance: flag the UPPER tail (large distance = OOD).
        if metric_name == "cosine":
            if data[i] < (mean - 3 * std):
                violations["Rule 1"].append(i)
        elif metric_name.startswith("mahalanobis"):
            if data[i] > (mean + 3 * std):
                violations["Rule 1"].append(i)
        else:
            raise NotImplementedError(f"requested metric does not ahve Rule 1 implemented: {metric_name}")

    return violations

"""Core OOD visualization function"""
def ood_visualization(distances, mean, UCL, LCL, rule, ood_labels=None, metric_name=None, figure_path="./figs/viz.png"):
    def make_confusion_matrix(vls, ood_true):
        ood_pred = [0 for _ in range(len(ood_true))]
        for idx in vls:
            ood_pred[idx] = 1
        return confusion_matrix(ood_true, ood_pred)

    plt.figure(figsize=(16, 6))
    plt.plot(distances, color='black', marker='o', markersize=4, linestyle='-')
    plt.axhline(y=mean, color='black', linestyle='-')
    if metric_name == "cosine":
        plt.axhline(y=np.clip(UCL, a_min=0.0, a_max=1.0), color='black', linestyle='--')
        plt.axhline(y=np.clip(LCL, a_min=0.0, a_max=1.0), color='black', linestyle='--')
        plt.fill_between(range(len(distances)), \
                        np.clip(LCL, a_min=0.0, a_max=1.0), \
                        np.clip(UCL, a_min=0.0, a_max=1.0), \
                        color='grey', alpha=0.1)
    else:
        plt.axhline(y=UCL, color='black', linestyle='--')
        plt.axhline(y=LCL, color='black', linestyle='--')
        plt.fill_between(range(len(distances)), \
                        LCL, \
                        UCL, \
                        color='grey', alpha=0.1)

    # For presentation purposes, disable axes
    # plt.xlabel('Image Sequence', fontsize=12)
    # plt.ylabel(metrics[metric_name], fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    
    plt.xticks([])
    plt.yticks([])

    # Assuming you have a function apply_spc_rules() which was not provided in your code
    violations = apply_spc_rules(distances, mean, (UCL - mean) / 3, metric_name)

    for idx in violations[rule]:
        plt.plot(idx, distances[idx], '*', color='grey', markersize=16, label='Auto OOD')

    if ood_labels is not None:
        for i, is_out in enumerate(ood_labels):
            if is_out:
                plt.plot(i, distances[i], marker='o', markersize=16, linestyle='None', color='black', mfc='none', label='Actual OOD')
        """Compute confusion matirx"""
        cfm = make_confusion_matrix(violations[rule], ood_labels)
        tn, fp, fn, tp = cfm.ravel()
        # Specificity
        if tn + fp == 0:
            specificity = np.nan
        else:
            specificity = tn / (tn + fp)
        # Sensitivity
        if tp + fn == 0:
            sensitivity = np.nan
        else:
            sensitivity = tp / (tp + fn)
        # Acc
        acc = (tp + tn) / (tp + tn + fp + fn)
        # Print
        print(f"Accuracy: {acc:.4f} | Specificity: {specificity:.4f} | Sensitivity: {sensitivity:.4f}")

    plt.xlim(0, len(distances) - 1)
    if metric_name == "cosine":
        plt.ylim(0, 1.5)

    # Ensure no duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.savefig(figure_path)
    plt.close()

# Compute cosine similarity
def compute_cosine_similarity(tr_features_, tt_features_):
    centroid = np.mean(tr_features_, axis=0)
    similarities = []
    epsilon = np.random.normal(loc=1e-6, scale=1e-3, size=centroid.shape[0])
    for feature in tt_features_:
        # Add a small epsilon to make sure cosine distances are not undefined
        similarities.append(
            # Clip the cosine distance to be in (0,1)
            1 - np.clip(distance.cosine(feature + epsilon, centroid), a_min=0.0, a_max=1.0)
        )
    return similarities

# Compute mahalanobis distance
def OLD_compute_mahalanobis_distance(tr_features_, tt_features_):
    centroid = np.mean(tr_features_, axis=0)
    cov_matrix = np.cov(tr_features_, rowvar=False)
    inv_cov_matrix = pinv(cov_matrix)

    distances = []
    for feature in tt_features_:
        diff = feature - centroid
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
        distances.append(dist)
    return distances

def _fingerprint(tr):
    """Cheap content check so a stale cache entry is never silently reused."""
    step = max(1, tr.shape[0] // 7)
    return (tr.shape, str(tr.dtype), float(np.asarray(tr[::step]).sum()))

def _fit_and_diagnose(tr):
    """Fit Ledoit-Wolf on the training features; build the diagnostic report."""
    tr = np.asarray(tr, dtype=np.float64)      # float32 features + kappa ~ 1e14 don't mix
    n, d = tr.shape
    eps = np.finfo(np.float64).eps
    tiny = 1e-300                              # display-only guard against /0

    # --- the matrix the previous code inverted: pinv(np.cov(tr)) ----------
    S_raw = np.cov(tr, rowvar=False)
    lam = np.linalg.eigvalsh(S_raw)            # ascending
    lam_abs = np.abs(lam)                      # numerical noise can dip below 0
    lam_max = lam[-1]
    lam_min = lam[0]
    mean_lam = lam.mean()                      # = trace(S)/d
    cond_raw = lam_max / max(lam_min, tiny)
    n_below_1e8 = int((lam < 1e-8).sum())
    lam_clip = np.clip(lam, 0.0, None)
    eff_rank = float(lam_clip.sum() ** 2 / max((lam_clip ** 2).sum(), tiny))

    # what scipy.linalg.pinv would do with this matrix (default tolerance)
    pinv_cutoff = d * eps * lam_max            # singular values below this are dropped
    retained = lam_abs > pinv_cutoff
    n_truncated = int(d - retained.sum())
    smallest_retained = float(lam_abs[retained].min()) if retained.any() else float("nan")
    amp_pinv = 1.0 / max(smallest_retained, tiny)

    # --- the replacement: Ledoit-Wolf shrinkage ----------------------------
    lw = LedoitWolf().fit(tr)
    alpha = float(lw.shrinkage_)
    lam_lw = np.linalg.eigvalsh(lw.covariance_)
    cond_lw = lam_lw[-1] / max(lam_lw[0], tiny)
    floor_theory = alpha * mean_lam            # min eigenvalue of (1-a)S + a*mean_eig*I
    amp_lw = 1.0 / max(lam_lw[0], tiny)
    # distortion of the dominant direction, compared on the same (1/n) scale
    lam_max_ml = lam_max * (n - 1) / n
    top_shift = abs(lam_lw[-1] - lam_max_ml) / max(lam_max_ml, tiny)

    report = "\n".join([
        "=" * 74,
        "[mahalanobis] training-feature covariance diagnostics",
        f"  features                          : n={n} samples, d={d} dims  (n/d={n/d:.1f})",
        "  --- raw sample covariance: the matrix pinv() previously inverted ---",
        f"  eigenvalues (min / mean / max)    : {lam_min:.2e} / {mean_lam:.2e} / {lam_max:.2e}",
        f"  condition number                  : {cond_raw:.2e}",
        f"  eigenvalues < 1e-8                : {n_below_1e8} of {d}",
        f"  effective rank (participation)    : {eff_rank:.1f} of {d}",
        f"  scipy pinv default cutoff         : {pinv_cutoff:.2e}  (d*eps*lam_max)",
        f"  eigs truncated by pinv            : {n_truncated};  smallest RETAINED eig: {smallest_retained:.2e}",
        f"  => pinv max amplification (1/lam) : {amp_pinv:.2e}",
        "     a tiny test-point component along that direction is squared and",
        "     multiplied by this factor before entering the distance",
        "  --- Ledoit-Wolf shrunk covariance: (1-a)*S + a*mean_eig*I ---",
        f"  shrinkage a                       : {alpha:.3e}",
        f"  eigenvalue floor (a * mean_eig)   : {floor_theory:.2e}",
        f"  eigenvalues (min / max)           : {lam_lw[0]:.2e} / {lam_lw[-1]:.2e}",
        f"  condition number                  : {cond_lw:.2e}   (reduced {cond_raw / max(cond_lw, tiny):.1e} x)",
        f"  => LW max amplification (1/lam)   : {amp_lw:.2e}   (reduced {amp_pinv / max(amp_lw, tiny):.1e} x)",
        f"  top-eigenvalue distortion         : {100.0 * top_shift:.3f} %   (dominant directions ~unchanged)",
        "=" * 74,
    ])
    return lw.location_, lw.get_precision(), report

# Compute mahalanobis distance (Ledoit-Wolf precision; diagnostics on first use)
def compute_mahalanobis_distance(tr_features_, tt_features_):
    key = id(tr_features_)
    fp = _fingerprint(tr_features_)
    entry = _MAHA_CACHE.get(key)
    if entry is None or entry["fp"] != fp:
        centroid, precision, report = _fit_and_diagnose(tr_features_)
        entry = {"fp": fp, "centroid": centroid, "precision": precision, "report": report}
        _MAHA_CACHE[key] = entry
        print(report)
    elif VERBOSE_EVERY_CALL:
        print(entry["report"])

    centroid = entry["centroid"]
    inv_cov_matrix = entry["precision"]
    distances = []
    for feature in tt_features_:
        diff = feature - centroid
        distances.append(np.sqrt(max(diff @ inv_cov_matrix @ diff.T, 0.0)))
    return distances

# ---------------------------------------------------------------------------
# Mahalanobis distance via DIRECT EQUATION SOLVING (no explicit inverse).
# Factors the Ledoit-Wolf covariance (PD by construction) once: Sigma = L L^T,
# then solves L y = (x - mu); d^2 = ||y||^2 is a sum of squares (>= 0 by
# construction, no clamp, no explicit inverse). Solving cannot rescue the raw
# covariance (singular => least-squares solve IS pinv); the one-time report
# probes the raw matrix to print that verdict for the actual data.
# ---------------------------------------------------------------------------
_MAHA_SOLVE_CACHE = {}

def _fingerprint(tr):
    step = max(1, tr.shape[0] // 7)
    return (tr.shape, str(tr.dtype), float(np.asarray(tr[::step]).sum()))

def _fit_solver(tr):
    """Fit Ledoit-Wolf, Cholesky-factor it, and probe the raw covariance."""
    tr64 = np.asarray(tr, dtype=np.float64)
    n, d = tr64.shape

    # Hypothesis probe: would direct solving work on the RAW covariance?
    try:
        cholesky(np.cov(tr64, rowvar=False), lower=True)
        raw_verdict = ("raw covariance happens to be numerically PD here; a direct "
                       "solve on it would run, but its accuracy is still limited by "
                       "the raw conditioning (see the 'mahalanobis' report)")
    except np.linalg.LinAlgError:
        raw_verdict = ("raw covariance is NOT numerically positive definite -> "
                       "Cholesky fails; direct solving alone cannot replace shrinkage")

    lw = LedoitWolf().fit(tr64)
    L = cholesky(lw.covariance_, lower=True)

    report = "\n".join([
        "=" * 74,
        "[mahalanobis-solve] direct-solve diagnostics",
        f"  features            : n={n} samples, d={d} dims",
        f"  raw-covariance probe: {raw_verdict}",
        f"  factored matrix     : Ledoit-Wolf covariance (PD by construction), "
        f"shrinkage a={float(lw.shrinkage_):.3e}",
        "  method              : Sigma = L L^T; solve L y = (x - mu); d^2 = ||y||^2",
        "                        (sum of squares => d^2 >= 0 by construction; no pinv,",
        "                        no explicit inverse, no clamping)",
        "  equivalence         : same Sigma as metric 'mahalanobis' -> distances must",
        "                        agree to ~1e-9 relative; larger deviation = bug",
        "=" * 74,
    ])
    return {"centroid": lw.location_, "L": L, "report": report}

def compute_mahalanobis_distance_solve(tr_features_, tt_features_):
    key = id(tr_features_)
    fp = _fingerprint(tr_features_)
    entry = _MAHA_SOLVE_CACHE.get(key)
    if entry is None or entry["fp"] != fp:
        entry = _fit_solver(tr_features_)
        entry["fp"] = fp
        _MAHA_SOLVE_CACHE[key] = entry
        print(entry["report"])
    elif VERBOSE_EVERY_CALL:
        print(entry["report"])

    diff = np.asarray(tt_features_, dtype=np.float64) - entry["centroid"]  # (n_test, d)
    y = solve_triangular(entry["L"], diff.T, lower=True)                   # L y = diff^T
    return np.sqrt(np.einsum("ij,ij->j", y, y))

# ---------------------------------------------------------------------------
# Mahalanobis distance via the ORIGINAL estimator: raw covariance + pinv.
# Kept as a comparison BASELINE for the LW-based metrics. Faithful to
# OLD_compute_mahalanobis_distance except three controlled deviations:
#   1. the fit is cached (pinv is deterministic -> identical numbers, far
#      less work inside the bootstrap loop);
#   2. features are promoted to float64, matching the LW paths so that the
#      comparison isolates the ESTIMATOR rather than arithmetic precision;
#   3. the quadratic form is clamped at 0 before sqrt: pinv round-off can go
#      slightly negative, and this module promotes RuntimeWarning to a hard
#      error, so without the clamp one bad sample kills the whole eval.
#      Clamp events are evidence of the instability and are reported once.
# ---------------------------------------------------------------------------
_MAHA_PINV_CACHE = {}

def _fit_pinv(tr):
    """Raw sample covariance + scipy pinv; see the raw-covariance section of
    the '[mahalanobis]' diagnostic report for this matrix's conditioning."""
    tr64 = np.asarray(tr, dtype=np.float64)
    n, d = tr64.shape
    centroid = tr64.mean(axis=0)
    precision = pinv(np.cov(tr64, rowvar=False))
    report = "\n".join([
        "=" * 74,
        "[mahalanobis-pinv] baseline (raw covariance + pseudo-inverse)",
        f"  features  : n={n} samples, d={d} dims",
        "  estimator : np.cov + scipy.linalg.pinv -- the pre-shrinkage original;",
        "              conditioning/amplification diagnostics are in the raw-",
        "              covariance section of the '[mahalanobis]' report",
        "  deviations: cached fit; float64 features; quadratic form clamped at 0",
        "              before sqrt (clamp events reported once when first seen)",
        "=" * 74,
    ])
    return {"centroid": centroid, "precision": precision, "report": report}

# Compute mahalanobis distance with the original pinv estimator (cached fit)
def compute_mahalanobis_distance_pinv(tr_features_, tt_features_):
    key = id(tr_features_)
    fp = _fingerprint(tr_features_)
    entry = _MAHA_PINV_CACHE.get(key)
    if entry is None or entry["fp"] != fp:
        entry = _fit_pinv(tr_features_)
        entry["fp"] = fp
        entry["neg_clamped"] = 0
        entry["neg_reported"] = False
        _MAHA_PINV_CACHE[key] = entry
        print(entry["report"])
    elif VERBOSE_EVERY_CALL:
        print(entry["report"])

    diff = np.asarray(tt_features_, dtype=np.float64) - entry["centroid"]  # (n_test, d)
    q = np.einsum("ij,jk,ik->i", diff, entry["precision"], diff)
    n_neg = int((q < 0).sum())
    if n_neg:
        entry["neg_clamped"] += n_neg
        if not entry["neg_reported"]:
            entry["neg_reported"] = True
            print(f"[mahalanobis-pinv] clamped {n_neg} negative quadratic form(s) to 0 "
                  "(pinv round-off; the LW metrics do not produce these); "
                  "further occurrences are counted silently")
    return np.sqrt(np.clip(q, 0.0, None))

"""Compute control limits"""
def compute_control_limits(tr_features, tt_features, metric):
    # Control similarities
    if metric == "cosine":
        train_distances = compute_cosine_similarity(tr_features, tr_features)
        test_distances = compute_cosine_similarity(tr_features, tt_features)
    elif metric == "mahalanobis":
        train_distances = compute_mahalanobis_distance(tr_features, tr_features)
        test_distances = compute_mahalanobis_distance(tr_features, tt_features)
    elif metric == "mahalanobis-solve":
        train_distances = compute_mahalanobis_distance_solve(tr_features, tr_features)
        test_distances = compute_mahalanobis_distance_solve(tr_features, tt_features)
    elif metric == "mahalanobis-pinv":
        train_distances = compute_mahalanobis_distance_pinv(tr_features, tr_features)
        test_distances = compute_mahalanobis_distance_pinv(tr_features, tt_features)
    else:
        raise NotImplementedError(f"Metric is not implemented: {metric}")
    # Control limit calculations
    train_mean = np.mean(train_distances)
    train_std = np.std(train_distances)
    train_UCL = train_mean + 3 * train_std
    train_LCL = train_mean - 3 * train_std

    # Return
    return train_distances, test_distances, train_mean, train_std, train_UCL, train_LCL

"""OOD statistics"""
def ood_statistics(tr_features, tt_features, ood_labels, metric, n=100, rule="Rule 1"):
    # Precompute training similarities
    if metric == "cosine":
        fxn = compute_cosine_similarity
    elif metric == "mahalanobis":
        fxn = compute_mahalanobis_distance
    elif metric == "mahalanobis-solve":
        fxn = compute_mahalanobis_distance_solve
    elif metric == "mahalanobis-pinv":
        fxn = compute_mahalanobis_distance_pinv
    else:
        raise NotImplementedError(f"Metric is not implemented: {metric}")
    train_distances = fxn(tr_features, tr_features)
    # Control limit calculations
    train_mean = np.mean(train_distances)
    train_std = np.std(train_distances)
    # Bootstrap arrays
    accuracy = []
    sensitivity = []
    specificity = []
    # Bootstrap loop
    random.seed(2022)   # author's bootstrap RNG (Python random, not numpy)
    for i in tqdm(range(n), disable=None):
        # Pick a subset of the testing images
        sample = random.sample(list(range(tt_features.shape[0])), k=100)   # author: no replacement, k=100
        tt_subset = tt_features[sample, :]
        ood_labels_subset = ood_labels[sample]
        # Calculate test similarities on subset
        tt_subset_distances = fxn(tr_features, tt_subset)
        # Calculate OOD detection accuracy/precision/sensitivity/specificity
        violations = apply_spc_rules(tt_subset_distances, train_mean, train_std, metric)
        ood_preds_subset = [0 for _ in range(len(tt_subset_distances))]
        for jj in violations[rule]:
            ood_preds_subset[jj] = 1
        cmatrix = confusion_matrix(ood_labels_subset, ood_preds_subset)
        tn, fp, fn, tp = cmatrix.ravel()
        # Specificity
        try:
            specificity.append(tn / (tn + fp))
        except RuntimeWarning:
            specificity.append(np.nan)
        # Sensitivity
        try:
            sensitivity.append(tp / (tp + fn))
        except RuntimeWarning:
            sensitivity.append(np.nan)
        # Accuracy
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
    # Report bootstrap results
    print(
        f"Accuracy: {np.mean(accuracy):.4f} [{(np.mean(accuracy) - np.std(accuracy)):.4f}, {(np.mean(accuracy) + np.std(accuracy)):.4f}]"
    )
    print(
        f"Specificity: {np.nanmean(specificity):.4f} [{(np.nanmean(specificity) - np.nanstd(specificity)):.4f}, {(np.nanmean(specificity) + np.nanstd(specificity)):.4f}]"
    )
    print(
        f"Sensitivity: {np.nanmean(sensitivity):.4f} [{(np.nanmean(sensitivity) - np.nanstd(sensitivity)):.4f}, {(np.nanmean(sensitivity) + np.nanstd(sensitivity)):.4f}]"
    )
    # Return
    return accuracy, specificity, sensitivity

"""Parse args"""
def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--metric", type=str, choices=["cosine", "mahalanobis", "mahalanobis-solve", "mahalanobis-pinv"], help="OOD metric")
    parser.add_argument("--method", type=str, choices=["autoencoder", "cnn", "ctr"], help="method to get features per image")
    parser.add_argument("--bootstrap", type=int, default=100, help="number of bootstrapped samples")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size of the run being evaluated (for output naming)")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DM_SEED", "1001")), help="seed of the run being evaluated (for output naming)")

    # extract
    opt = parser.parse_args()

    # Make options dictionary
    options = {
        "metric": opt.metric,
        "method": opt.method,
        "positive_dataset": "organamnist", # hard coded
        "bootstrap": opt.bootstrap,
        "batch_size": opt.batch_size,
        "seed": opt.seed,
        "data_dir": cfg["data_dir"]
    }

    return options

"""Main"""
def main():
    options = parse_options()
    set_seed(options["seed"])

    # Load data from npz (per-run folder, matching get_features.py)
    numpy_dir = os.path.join("./numpy_files", run_name(options["batch_size"], options["seed"]))
    data_splits_path = os.path.join(numpy_dir, "data_splits.npz")
    D = np.load(data_splits_path)
    Xtr = D["Xtr"]
    ytr = D["ytr"]
    Xvl = D["Xvl"]
    yvl = D["yvl"]
    Xtt = D["Xtt"]
    ytt = D["ytt"]
    print(f"Loaded data splits from: {data_splits_path} !")

    # Get features
    if options["method"] == "autoencoder":
        F = np.load(os.path.join(numpy_dir, 'autoencoder_features.npz'))
        Ftr = F["autoencoder_Ftr"]
        Ftt = F["autoencoder_Ftt"]
        ckpt_pth = str(F["autoencoder_pth"])
    elif options["method"] == "cnn":
        F = np.load(os.path.join(numpy_dir, 'cnn_features.npz'))
        Ftr = F["cnn_Ftr"]
        Ftt = F["cnn_Ftt"]
        ckpt_pth = str(F["cnn_pth"])
    elif options["method"] == "ctr":
        F = np.load(os.path.join(numpy_dir, 'ctr_features.npz'))
        Ftr = F["ctr_Ftr"]
        Ftt = F["ctr_Ftt"]
        ckpt_pth = str(F["ctr_pth"])
    else:
        raise NotImplementedError(f"Requested feature method is not implemented: {options['method']}")

    # Provenance guard: features must come from this run's checkpoint.
    # Hard-fail if the checkpoint path carries a DIFFERENT run key (true mislabeling);
    # warn only if it carries none (legacy/flat checkpoints, e.g. paper-pretrained).
    expected = run_name(options["batch_size"], options["seed"])
    if expected not in ckpt_pth:
        msg = (f"features in {numpy_dir} were extracted from checkpoint '{ckpt_pth}', "
               f"which does not carry this run's key ({expected}).")
        if re.search(r"bsz\d+_seed\d+", ckpt_pth):
            raise RuntimeError(f"Feature provenance mismatch: {msg} Re-run get_features.py for this run.")
        warnings.warn(f"Feature provenance unverified (legacy checkpoint path): {msg}")

    # Get features for all in-distribution (ID) training images
    Ftr_in = Ftr[ytr == 1]
    # Compute OOD statistics using entire test set
    n = options["bootstrap"]
    accuracy, specificity, sensitivity = ood_statistics(Ftr_in, Ftt, 1 - ytt, options["metric"], n=n)

    # per-run output sub-folders: results/<run>, figures/<run>
    results_dir = make_run_dir("./results", options["batch_size"], options["seed"], metric=options["metric"])
    figures_dir = make_run_dir("./figures", options["batch_size"], options["seed"], metric=options["metric"])

    np.savez(
        file=os.path.join(results_dir, f"{options['metric']}_{options['method']}_bootstrap.npz"),
        accuracy=accuracy,
        specificity=specificity,
        sensitivity=sensitivity,
    )

    # Control chart visualization over the full test set
    _, test_distances, train_mean, train_std, train_UCL, train_LCL = \
        compute_control_limits(Ftr_in, Ftt, options["metric"])
    figure_path = os.path.join(figures_dir, f"{options['metric']}_{options['method']}_viz.png")
    ood_visualization(
        test_distances,
        train_mean,
        train_UCL,
        train_LCL,
        rule="Rule 1",
        ood_labels=1 - ytt,
        metric_name=options["metric"],
        figure_path=figure_path,
    )
    print(f"Saved control-chart figure to: {figure_path}")

    # Add to results table
    table_entry = {
        "Batch Size": options["batch_size"],
        "Metric": options["metric"],
        "Seed": options["seed"],
        "Method": options["method"],
        # Accuracy
        "Mean Accuracy": np.nanmean(accuracy),
        "LCL Accuracy": np.nanmean(accuracy) - np.nanstd(accuracy),
        "UCL Accuracy": np.nanmean(accuracy) + np.nanstd(accuracy),
        # Specificity
        "Mean Specificity": np.nanmean(specificity),
        "LCL Specificity": np.nanmean(specificity) - np.nanstd(specificity),
        "UCL Specificity": np.nanmean(specificity) + np.nanstd(specificity),
        # Sensitivity
        "Mean Sensitivity": np.nanmean(sensitivity),
        "LCL Sensitivity": np.nanmean(sensitivity) - np.nanstd(sensitivity),
        "UCL Sensitivity": np.nanmean(sensitivity) + np.nanstd(sensitivity),
    }
    table_entry_df = pd.DataFrame([table_entry])
    # per-run results: accumulate one row per method (replace this method's row on re-run)
    per_run_path = os.path.join(results_dir, "results.csv")
    if os.path.exists(per_run_path):
        existing = pd.read_csv(per_run_path)
        existing = existing[existing["Method"] != options["method"]]   # drop stale row for this method
        run_df = pd.concat([existing, table_entry_df], ignore_index=True)
    else:
        run_df = table_entry_df
    run_df = run_df.sort_values("Method").reset_index(drop=True)
    run_df.to_csv(per_run_path, header=True, index=False)

if __name__ == "__main__":
    main()
