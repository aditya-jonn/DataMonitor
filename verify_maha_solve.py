"""Verification harness for the 'mahalanobis-solve' metric.

Two modes, both automatic:

  A. STANDALONE (always runs): synthetic features engineered into the same
     ill-conditioned regime as the real ones (low effective rank, float32),
     exercising the EXACT production code pasted below. Checks:
       1. hypothesis probe -- does Cholesky on the RAW covariance work at all,
          and does least-squares "direct solving" of the raw system differ
          from pinv (it should NOT: they are the same operation)?
       2. equivalence -- LW+solve vs LW+precision agree to ~1e-9 (same Sigma)
       3. non-negativity -- d^2 is a sum of squares in the solve path
       4. timing -- vectorized triangular solve vs the per-row precision loop

  B. ACCEPTANCE (runs only if `import ood_detection` succeeds, i.e. inside the
     repo venv with cfg.json present): cross-checks the repo's own
     compute_mahalanobis_distance, compute_mahalanobis_distance_solve, and
     (if present) compute_mahalanobis_distance_pinv on
     synthetic data. Run from the repo root after applying the edits:
         python verify_maha_solve.py
"""
import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular, pinv
from sklearn.covariance import LedoitWolf

# ----------------------------------------------------------------------------
# EXACT copies of the production pieces (keep in sync with ood_detection.py)
# ----------------------------------------------------------------------------
VERBOSE_EVERY_CALL = False
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

# ----------------------------------------------------------------------------
# Synthetic data in the real regime: low effective rank + float32 quantization
# ----------------------------------------------------------------------------
def make_features(n=2000, d=512, r=60, noise=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, r))
    W = rng.standard_normal((r, d))
    X = Z @ W + noise * rng.standard_normal((n, d))
    return X.astype(np.float32)


def main():
    print("#" * 74)
    print("# A. STANDALONE verification (exact production code, synthetic data)")
    print("#" * 74)
    Xtr = make_features()
    Xtt = make_features(n=1000, seed=1) + 0.05  # slight shift, irrelevant to checks
    tr64 = np.asarray(Xtr, dtype=np.float64)
    tt64 = np.asarray(Xtt, dtype=np.float64)

    S_raw = np.cov(tr64, rowvar=False)
    lam = np.linalg.eigvalsh(S_raw)
    print(f"\nraw covariance: cond = {lam[-1] / max(abs(lam[0]), 1e-300):.2e}  "
          f"(eig min/max = {lam[0]:.2e} / {lam[-1]:.2e})")

    # --- 1. hypothesis probe: 'direct solving' on the RAW system -------------
    mu_raw = tr64.mean(axis=0)
    P_raw = pinv(S_raw)
    sub = tt64[:40] - mu_raw
    d_pinv = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", sub, P_raw, sub), 0.0))
    d_lstsq = np.sqrt(np.maximum(
        np.array([row @ np.linalg.lstsq(S_raw, row, rcond=None)[0] for row in sub]), 0.0))
    rel = np.max(np.abs(d_lstsq - d_pinv) / np.maximum(d_pinv, 1e-300))
    print(f"least-squares solve of RAW system vs pinv: max rel diff = {rel:.2e}")
    print("  -> 'solving' a singular system IS the pinv operation; same amplification.")

    # --- 2/3. production solve path vs precision path on the SAME LW Sigma ---
    d_solve = compute_mahalanobis_distance_solve(Xtr, Xtt)          # production code
    lw = LedoitWolf().fit(tr64)
    mu, P = lw.location_, lw.get_precision()
    pre_clamp = np.array([(x - mu) @ P @ (x - mu).T for x in tt64])  # current prod loop
    d_prec = np.sqrt(np.maximum(pre_clamp, 0.0))
    agree = np.max(np.abs(d_solve - d_prec) / np.maximum(d_prec, 1e-300))
    print(f"\nLW+solve vs LW+precision: max rel diff = {agree:.2e}  "
          f"({'PASS' if agree < 1e-8 else 'FAIL'} @ 1e-8)")
    print(f"precision path min quadratic form pre-clamp: {pre_clamp.min():.3e}")
    print(f"solve path min d^2 (sum of squares)        : {(d_solve ** 2).min():.3e}  (>= 0 always)")

    # --- 4. timing ------------------------------------------------------------
    t0 = time.perf_counter()
    _ = [np.sqrt(max((x - mu) @ P @ (x - mu).T, 0.0)) for x in tt64]
    t_loop = time.perf_counter() - t0
    diff = tt64 - mu
    t0 = time.perf_counter()
    y = solve_triangular(np.linalg.cholesky(lw.covariance_), diff.T, lower=True)
    _ = np.sqrt(np.einsum("ij,ij->j", y, y))
    t_solve = time.perf_counter() - t0
    print(f"\ntiming, {len(tt64)} test points x {tr64.shape[1]} dims: "
          f"per-row precision loop {t_loop * 1e3:.1f} ms vs vectorized solve {t_solve * 1e3:.1f} ms "
          f"({t_loop / max(t_solve, 1e-12):.1f}x)")

    # --- B. acceptance against the real module --------------------------------
    print("\n" + "#" * 74)
    print("# B. ACCEPTANCE against ood_detection.py (skipped unless importable)")
    print("#" * 74)
    try:
        import ood_detection as od
    except Exception as e:
        print(f"skipped: could not import ood_detection ({type(e).__name__}: {e})")
        print("run this from the repo root inside the venv after applying the edits.")
        return
    d_a = np.asarray(od.compute_mahalanobis_distance(Xtr, Xtt), dtype=np.float64)
    d_b = np.asarray(od.compute_mahalanobis_distance_solve(Xtr, Xtt), dtype=np.float64)
    rel = np.max(np.abs(d_a - d_b) / np.maximum(d_a, 1e-300))
    print(f"repo precision path vs repo solve path: max rel diff = {rel:.2e}  "
          f"({'PASS' if rel < 1e-8 else 'FAIL'} @ 1e-8)")

    # pinv baseline cross-checks (added with the 'mahalanobis-pinv' metric)
    if hasattr(od, "compute_mahalanobis_distance_pinv"):
        d_repo_pinv = np.asarray(od.compute_mahalanobis_distance_pinv(Xtr, Xtt[:40]), dtype=np.float64)
        rel_pinv = np.max(np.abs(d_repo_pinv - d_pinv) / np.maximum(d_pinv, 1e-300))
        print(f"repo pinv path vs reference raw-pinv   : max rel diff = {rel_pinv:.2e}  "
              f"({'PASS' if rel_pinv < 1e-8 else 'FAIL'} @ 1e-8)")
        sep = np.max(np.abs(d_repo_pinv - d_b[:40]) / np.maximum(d_b[:40], 1e-300))
        print(f"pinv baseline vs LW paths (SHOULD differ): max rel diff = {sep:.2e}")
    else:
        print("pinv variant not present in module -- pinv checks skipped")


if __name__ == "__main__":
    main()