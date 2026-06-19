# Data-Drift Monitoring

Beyond per-image OOD detection, DataMonitor simulates a gradual distribution shift over time and detects the change point with a CUSUM control chart. This is stage 4b, which produces the paper's drift figure ("Figure 3"). The two `SPC_Charts/` modules are upstream code (described here at the interface level); the novel part is the inline driver in `datamonitor.sh` that connects them, something the upstream repository only ever did inside example notebooks.

---

## The drift simulation: `simulate_data_shift`

*File: `SPC_Charts/data_shift_simulation.py`.*

`simulate_data_shift` turns two pools of per-image scores (in-distribution and out-of-distribution) into a daily time series with an injected shift.

```python
simulate_data_shift(in_dist_data, out_dist_data, shift_start_day,
                    total_days, images_per_day, shift_percentage)
```

For each day it draws `images_per_day` scores. Before `shift_start_day`, none come from the OOD pool; from `shift_start_day` onward, `shift_percentage`% come from the OOD pool and the rest from the in-distribution pool (both sampled without replacement within a day). It returns `daily_averages` (one mean score per day, the signal the CUSUM chart monitors), `all_data` (the full concatenated per-image scores), and the `shift_start_day`/`shift_percentage` echoed back. The premise is that mixing in OOD images shifts the daily-average score, which a control chart should detect shortly after it begins.

---

## The change detector: `CUSUMChangeDetector`

*File: `SPC_Charts/CUSUM_detector.py`.*

A CUSUM (cumulative sum) chart accumulates demeaned, slack-adjusted deviations and alarms when the running sum crosses a limit. The detector computes a high-side and a low-side statistic:

```
S_hi[t] = max(0, S_hi[t−1] + (x[t] − µ₀ − k))
S_lo[t] = min(0, S_lo[t−1] + (x[t] − µ₀ + k))
alarm when  S_hi > h  or  S_lo < −h
```

where `x[t]` is the daily-average signal, `µ₀` is the in-control mean (estimated from the pre-change days), `k` is the reference value (slack), and `h` is the decision interval (control limit).

`changeDetection(...)` splits the signal into in-control (`[:pre_change_days]`) and out-of-control (`[pre_change_days:total_days]`) periods, estimates `µ₀` and the in-control standard deviation `σ` from the in-control period, and sets:

```
k = (k_th · σ) / 2          # reference value, in multiples of σ
h = control_limit · σ       # decision threshold, in multiples of σ
```

So `k_th = 1` gives `k = 0.5σ`, and `control_limit = 4` gives `h = 4σ`. It computes the high/low CUSUM, plots them (`plotCUSUM`, with the threshold lines and the detected shift), and records a one-row summary: the `k` value, threshold, false-positive count (alarms before the true shift), true-positive count, average detection delay (days from the true shift to the first alarm), the mean time between false alarms (MTBFA), and the false-alarm rate (`1/MTBFA`). `summary()` returns that table as a string.

---

## How the pipeline wires them (stage 4b, Figure 3)

Since the upstream repository never connects `data_shift_simulation.py` and `CUSUM_detector.py` in normal code, stage 4b is an inline Python driver (a heredoc in `datamonitor.sh`) that uses both with the contrastive encoder under cosine scoring (the paper's chosen combination for Figure 3). The driver:

1. Loads the contrastive features and raw test labels for this run (`ctr_features.npz` + `data_splits.npz`).
2. Computes the in-distribution centroid of the training features (`Ftr[ytr == 1].mean(0)`).
3. Scores every test feature by cosine similarity to that centroid, splitting into an in-pool (`ytt == 1`) and an out-pool (`ytt == 0`).
4. Feeds the pools to `simulate_data_shift` and the daily signal to `CUSUMChangeDetector`, configured to the paper's setting: 60 days total, 100 images/day, shift starting at day 31, post-shift OOD rate 4% (the midpoint of the paper's 3–5% range), and CUSUM with `k_th = 1` ⇒ `k = 0.5σ` and `control_limit = 4` ⇒ `h = 4σ`.
5. Saves the figure to `figures/bsz<B>_seed<S>/drift_ctr_cosine_figure3.png` and prints the detector's summary.

```python
daily_avg, _, shift_day, _ = simulate_data_shift(
    in_dist_data=in_pool, out_dist_data=out_pool,
    shift_start_day=31, total_days=60, images_per_day=100,
    shift_percentage=4.0,
)
det = CUSUMChangeDetector(pre_change_days=30, total_days=60)
det.changeDetection(
    CUSUM_data_average_day=daily_avg,
    pre_change_days=30, total_days=60,
    control_limit=4.0,   # h = 4·σ
    k_th=1.0,            # k_th=1 → k = (1·σ)/2 = 0.5σ
    save_plot=False,     # the driver saves to its own path
)
```

The internals of both modules are upstream and unmodified; the driver only sets their parameters and saves the figure. This figure always uses the contrastive/cosine combination, so it is metric-independent in the run-key sense and lives under the `bsz<B>_seed<S>` key.

---

## Relationship to per-image OOD detection

Per-image OOD detection (stage 4a) and drift monitoring (stage 4b) share the same idea (score against an in-distribution reference, then apply an SPC chart) but operate at different granularities:

- **Stage 4a / Shewhart:** one score *per image*; a single image outside the 3σ band is flagged. This measures detection quality on a static test set (the bootstrap table).
- **Stage 4b / CUSUM:** one score *per day* (a daily average); the chart accumulates small deviations over time and alarms when a *sustained* shift has occurred. This measures how quickly a gradual population drift is detected.

A Shewhart chart reacts to large single-point excursions; a CUSUM chart is tuned to detect small, persistent mean shifts, the right tool for a slowly drifting data stream.
