# nba-hawkes-paper

Code and processed data for the manuscript:

> **Absence of detectable Hawkes self-excitation in NBA substitution timing**
> Pinhe Chen — Fort Hays State University, 2026
> *Manuscript prepared for submission to Scientific Reports.*

## Overview

This repository contains the Python implementation, processed datasets, and figure-generation scripts for a study testing whether NBA substitution timing exhibits detectable Hawkes self-excitation.

The headline finding is that, after same-second mass aggregation and adjustment for an inhomogeneous period-minute baseline, the Hawkes extension does not improve the fitted likelihood over the non-self-exciting baseline. The nesting-enforced M1f fit selects the boundary solution $\widehat\alpha = 0$ across $122{,}747$ atomic events from $7{,}380$ team-level realizations spanning all $30$ NBA teams in the $2022$–$23$ through $2024$–$25$ regular seasons.

The parametric bootstrap output is included to reproduce the boundary-calibrated likelihood-ratio analysis reported in the manuscript.

## Key methodology

- **Mass aggregation** of same-second multi-player substitutions into atomic events
- **Inhomogeneous Poisson baseline (M3)** over 48 period-minute bins
- **Multi-start EM with M3-anchor** to handle the boundary case $\alpha = 0$
- **Parametric bootstrap** ($B=1{,}000$) for finite-sample likelihood-ratio calibration
- **Cluster-robust standard errors** at the game level
- **Random-time-change residuals** for diagnostic checking

## Repository structure

```text
nba-hawkes-paper/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── atomic_events.csv               (mass-aggregated substitution events)
│   ├── m1f_best_baseline.csv           (fitted M3 baseline, 48 bins)
│   ├── bootstrap_v4c_results.csv       (B = 1,000 bootstrap replicates)
│   ├── phase4_per_team.csv             (per-team M1 fits, 30 teams)
│   ├── phase4_model_comparison.csv     (M0 / M1 / M2 / M3 LL, AIC, BIC)
│   ├── residuals_hawkes.npy            (Hawkes RTC residuals)
│   └── residuals_b2.npy                (M3 RTC residuals)
├── code/
│   ├── 01_data_acquisition.py          (NBA API → raw play-by-play)
│   ├── 02_mass_aggregation.py          (raw → atomic events)
│   ├── 03_fit_m0_m1_m2_m3.py           (closed-form + EM-style fits)
│   ├── 04_fit_m1f_multistart.py        (multi-start EM with M3-anchor)
│   ├── 05_bootstrap_v4c.py             (parametric bootstrap)
│   ├── 06_per_team_m1.py               (per-team M1 fits)
│   ├── 07_robustness_r1_r4.py          (R1 power-law / R2 sweep / R3 marked / R4 RTC)
│   └── 08_cluster_robust_se.py         (cluster-robust sandwich estimator)
└── figures/
    ├── figure3_baseline_intensity.py
    ├── figure5_bootstrap_distribution.py
    ├── figure6_team_alpha_aic.py
    ├── figure7_rtc_residuals.py
    └── figure8_pooled_q1_trajectory.py
```

## Requirements

The analysis was developed and run with Python 3.10+. Install dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are NumPy, SciPy, pandas, matplotlib, and the [`nba_api`](https://github.com/swar/nba_api) Python client.

## Reproducing the results

### Full pipeline

The full pipeline runs from raw NBA play-by-play data to the final tables and figures:

```bash
# Step 1: Acquire raw play-by-play data (1–2 hours, NBA API rate-limited)
python code/01_data_acquisition.py

# Step 2: Mass-aggregate same-second substitution events
python code/02_mass_aggregation.py

# Step 3: Fit M0, M1, M2, and M3
python code/03_fit_m0_m1_m2_m3.py

# Step 4: Fit M1f with multi-start EM and M3-anchor
python code/04_fit_m1f_multistart.py

# Step 5: Parametric bootstrap (the full B = 1,000 run takes many hours
#         on a single CPU core; intermediate state is written to disk
#         so the run can be resumed if interrupted)
python code/05_bootstrap_v4c.py

# Step 6: Per-team analysis
python code/06_per_team_m1.py

# Step 7: Robustness checks (R1–R4)
python code/07_robustness_r1_r4.py

# Step 8: Cluster-robust standard errors
python code/08_cluster_robust_se.py
```

### Figure reproduction

```bash
python figures/figure3_baseline_intensity.py
python figures/figure5_bootstrap_distribution.py
python figures/figure6_team_alpha_aic.py
python figures/figure7_rtc_residuals.py
python figures/figure8_pooled_q1_trajectory.py
```

### Faster reproduction path

To reproduce the figures and headline inference without re-downloading raw play-by-play data, use the processed files in `data/` and start from the model-fitting or figure-generation scripts. This avoids the 1–2 hour API download.

## Data availability

Raw NBA play-by-play data are publicly accessible through the NBA Stats endpoints and were retrieved using the `nba_api` Python client.

Processed analysis files — including the mass-aggregated substitution-event dataset, fitted baseline rates, model-comparison output, parametric-bootstrap likelihood-ratio results, and residual arrays — are included in the `data/` directory of this repository.

## Code availability

All code used to acquire the raw play-by-play data, construct the mass-aggregated event dataset, fit the point-process models, run the parametric bootstrap, perform the robustness checks, and reproduce the manuscript figures is included in this repository.

## Citation

If you use this code or data, please cite:

> Chen, P. Absence of detectable Hawkes self-excitation in NBA substitution timing. Manuscript, 2026.

(This citation will be updated once the manuscript is formally accepted.)

## License

MIT License — see [LICENSE](LICENSE).

## Contact

Pinhe Chen
Fort Hays State University, Hays, Kansas, USA
<p_chen10@mail.fhsu.edu>
