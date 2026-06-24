# Data-Drift Monitoring

Beyond per-image OOD detection, DataMonitor simulates a gradual distribution shift over time and detects the change point with a CUSUM control chart. This is stage 4b, which produces the paper's drift figure ("Figure 3"). The two `SPC_Charts/` modules are upstream code, documented here at the interface level; the inline driver in `datamonitor.sh` is what wires them together.

---

## The drift simulation: `simulate_data_shift`

*File: `SPC_Charts/data_shift_simulation.py`.*

`simulate_data_shift(in_dist_data, out_dist_data, shift_start_day, total_days, images_per_day, shift_percentage)` turns two pools of per-image scores into a daily time series with an injected shift. Each day it draws `images_per_day` scores: before `shift_start_day`, none from the OOD pool; from `shift_start_day` onward, `shift_percentage`% from the OOD pool and the rest from the in-distribution pool (sampled without replacement within a day). It returns `daily_averages` (one mean score per day, the signal the CUSUM chart monitors), the concatenated per-image scores, and the shift day/percentage echoed back. The premise: mixing in OOD images shifts the daily-average score, which a control chart should detect shortly after it begins.

---

## The change detector: `CUSUMChangeDetector`

*File: `SPC_Charts/CUSUM_detector.py`.*

A CUSUM chart accumulates demeaned, slack-adjusted deviations and alarms when the running sum crosses a limit. It tracks a high-side and a low-side statistic:

```
S_hi[t] = max(0, S_hi[t−1] + (x[t] − µ₀ − k))
S_lo[t] = min(0, S_lo[t−1] + (x[t] − µ₀ + k))
alarm when  S_hi > h  or  S_lo < −h
```

where `x[t]` is the daily-average signal, `µ₀` is the in-control mean, `k` is the reference value (slack), and `h` is the decision interval. `changeDetection(...)` splits the signal into in-control (`[:pre_change_days]`) and out-of-control (`[pre_change_days:total_days]`) periods, estimates `µ₀` and the in-control standard deviation `σ` from the in-control period, and sets `k = (k_th · σ)/2` and `h = control_limit · σ`. So `k_th = 1` gives `k = 0.5σ` and `control_limit = 4` gives `h = 4σ`. It computes the CUSUM statistics, plots them (`plotCUSUM`), and records a one-row summary: `k`, threshold, false-positive count (alarms before the true shift), true-positive count, average detection delay, mean time between false alarms (MTBFA), and false-alarm rate. `summary()` returns that table as a string.

---

## How the pipeline wires them (stage 4b, Figure 3)

Stage 4b is an inline Python driver (a heredoc in `datamonitor.sh`) that uses both modules with the contrastive encoder under cosine scoring, the paper's chosen combination for Figure 3. The driver:

1. Loads the contrastive features and raw test labels for this run (`ctr_features.npz` + `data_splits.npz`).
2. Computes the in-distribution centroid of the training features (`Ftr[ytr == 1].mean(0)`).
3. Scores every test feature by cosine similarity to that centroid, splitting into an in-pool (`ytt == 1`) and an out-pool (`ytt == 0`).
4. Feeds the pools to `simulate_data_shift` and the daily signal to `CUSUMChangeDetector`, at the paper's setting: 60 days, 100 images/day, shift at day 31, post-shift OOD rate 4% (the midpoint of the paper's 3–5% range), `k_th = 1`, `control_limit = 4`.
5. Saves the figure to `figures/bsz<B>_seed<S>/drift_ctr_cosine_figure3.png` and prints the detector's summary.

The driver only sets the modules' parameters and saves the figure, leaving their internals unmodified. This figure always uses the contrastive/cosine combination, so in the run-key sense it is metric-independent and lives under the `bsz<B>_seed<S>` key.

---

## Relationship to per-image OOD detection

Per-image OOD detection (stage 4a) and drift monitoring (stage 4b) share the same idea at different granularities:

- **Stage 4a:** one score *per image*; a single image outside the 3σ band is flagged. Measures detection quality on a static test set (the bootstrap table).
- **Stage 4b / CUSUM:** one score *per day* (a daily average); the chart accumulates small deviations and alarms when a *sustained* shift has occurred. Measures how quickly a gradual drift is detected.

A CUSUM chart is tuned to detect small, persistent mean shifts, the right tool for a slowly drifting data stream.
