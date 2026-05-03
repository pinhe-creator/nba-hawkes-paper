#!/usr/bin/env python3
"""
================================================================================
Figure 3: Inhomogeneous baseline mu_b(t) — 48 period-minute bins
================================================================================

Purpose:
  Visualize the M3 / M1f inhomogeneous baseline rate mu_b(t) across 2880 s
  of regulation play, showing:
    - Sharp valleys in the first minute after each period begins
    - Peaks in the final minute before each period break
    - Late-game variation in Q4 (if present in the fitted baseline)

Inputs:
  m1f_best_baseline.csv   — 48-row CSV with columns [bin_idx, t_start, t_end, mu_b]
                            (output of 08_fit_m1f_v4.py)

Outputs:
  figure3_baseline_intensity.pdf
  figure3_baseline_intensity.png

Usage:
  cd ~/Downloads
  python3 figure3_baseline_intensity.py

Required: matplotlib, pandas, numpy
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = "m1f_best_baseline.csv"
OUTPUT_PDF = "figure3_baseline_intensity.pdf"
OUTPUT_PNG = "figure3_baseline_intensity.png"

T_HORIZON = 2880.0
N_BINS = 48
BIN_WIDTH = T_HORIZON / N_BINS  # 60 s per bin

PERIOD_BOUNDARIES = [720, 1440, 2160]  # Q1/Q2, Q2/Q3, Q3/Q4

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ============================================================
# LOAD DATA
# ============================================================
if not os.path.exists(INPUT_CSV):
    sys.exit(f"[FATAL] {INPUT_CSV} not found in current directory.\n"
             f"        Run 08_fit_m1f_v4.py first to generate it.")

df = pd.read_csv(INPUT_CSV)
print(f"[Loaded] {INPUT_CSV} with {len(df)} bins")
print(f"  Columns: {list(df.columns)}")
print(df.head())

# Try to detect column names robustly
mu_col = None
for cand in ['mu_b_best', 'mu_b', 'rate', 'lambda', 'baseline', 'mu']:
    if cand in df.columns:
        mu_col = cand
        break
if mu_col is None:
    sys.exit(f"[FATAL] Could not find baseline rate column. Available: {list(df.columns)}")

print(f"  Using rate column: '{mu_col}'")

# Try to detect bin time column
t_col = None
for cand in ['t_start', 't_left', 't_lo', 'bin_start', 't']:
    if cand in df.columns:
        t_col = cand
        break

if t_col is None:
    # Use 'bin' column if present, else fall back to row index
    if 'bin' in df.columns:
        print(f"  [INFO] Using 'bin' column * {BIN_WIDTH} for time axis")
        t_starts = df['bin'].values * BIN_WIDTH
    else:
        print(f"  [INFO] No time column found; using row_idx * {BIN_WIDTH}")
        t_starts = np.arange(N_BINS) * BIN_WIDTH
else:
    t_starts = df[t_col].values

t_centers = t_starts + BIN_WIDTH / 2
mu_b = df[mu_col].values

# Sanity check
if len(mu_b) != N_BINS:
    print(f"  [WARNING] Expected {N_BINS} bins, found {len(mu_b)}")

print(f"\n[mu_b stats]")
print(f"  min  = {mu_b.min():.6f} (1/s)")
print(f"  max  = {mu_b.max():.6f} (1/s)")
print(f"  mean = {mu_b.mean():.6f} (1/s)")

# Avoid division by zero on min if it's exactly 0 (rare but defensive)
if mu_b.min() > 0:
    rate_ratio = mu_b.max() / mu_b.min()
    print(f"  max / min      = {rate_ratio:.1f}x  (highest bin / lowest bin)")
else:
    rate_ratio = float('inf')
    print(f"  max / min      = inf (some bin has zero rate)")
range_over_mean = (mu_b.max() - mu_b.min()) / mu_b.mean()
print(f"  range / mean   = {range_over_mean:.2f}")

# Identify peaks and valleys
peak_idx = np.argmax(mu_b)
valley_idx = np.argmin(mu_b)
print(f"\n  Peak at t = {t_centers[peak_idx]:.0f} s (bin {peak_idx}): {mu_b[peak_idx]:.6f}")
print(f"  Valley at t = {t_centers[valley_idx]:.0f} s (bin {valley_idx}): {mu_b[valley_idx]:.6f}")

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.5))

# Build proper bin edges so the 48th bin extends fully to 2880 s
bin_edges = np.append(t_starts, T_HORIZON)  # length 49

# Main step plot via stairs() — clean, no edge artifacts
ax.stairs(mu_b, bin_edges, color='#1F4E79', lw=1.6,
          baseline=None,
          label=r'$\widehat{\mu}_b(t)$ (48 period-minute bins)')

# Fill under via fill_between with step='post'
ax.fill_between(bin_edges, np.r_[mu_b, mu_b[-1]],
                step='post', color='#1F4E79', alpha=0.12)

# Period boundaries
for i, b in enumerate(PERIOD_BOUNDARIES):
    ax.axvline(b, color='#B22222', linestyle='--', lw=1.0, alpha=0.6,
               label='Period boundaries' if i == 0 else None)

# Period labels at top
y_text = ax.get_ylim()[1] * 0.92 if False else None  # set after final ylim
period_labels = ['Q1', 'Q2', 'Q3', 'Q4']
period_centers = [360, 1080, 1800, 2520]

# Homogeneous mean reference line
mu_homog = mu_b.mean()
ax.axhline(mu_homog, color='gray', linestyle=':', lw=1.0, alpha=0.7,
           label=fr'M0 homogeneous mean $\bar\mu = {mu_homog:.4f}$')

# Axis settings
ax.set_xlim(0, T_HORIZON)
ax.set_xticks([0, 360, 720, 1080, 1440, 1800, 2160, 2520, 2880])
ax.set_xticklabels(['0', '360', '720', '1080', '1440',
                    '1800', '2160', '2520', '2880'])

# Add minutes:seconds annotation below
ax.set_xlabel('Game time elapsed (seconds since tip-off)')
ax.set_ylabel(r'Substitution rate $\widehat{\mu}_b(t)$ (events / s / realization)')
ax.set_title(r'Estimated inhomogeneous baseline $\widehat{\mu}_b(t)$, 48 period-minute bins'
             '\n(M3 specification, also nested in M1f at $\\alpha = 0$)')

# Period labels at top — place below title (at 88% of plot height, not 94%)
y_lim = ax.get_ylim()
y_top = y_lim[1] * 0.88
for lbl, ctr in zip(period_labels, period_centers):
    ax.text(ctr, y_top, lbl, ha='center', fontsize=11,
            fontweight='bold', color='#555555',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='lightgray', lw=0.5))

# Annotate peak — place label BELOW-RIGHT of peak to avoid title overlap
peak_t = t_centers[peak_idx]
peak_v = mu_b[peak_idx]
valley_t = t_centers[valley_idx]
valley_v = mu_b[valley_idx]

# Cap y position to stay below 75% of plot height (title sits at ~94%)
y_lim_now = ax.get_ylim()
label_y = min(peak_v * 0.75, y_lim_now[1] * 0.62)

ax.annotate(f'Peak: {peak_v:.4f}\nat $t = {peak_t:.0f}$ s',
            xy=(peak_t, peak_v),
            xytext=(peak_t + 250, label_y),
            fontsize=8, color='#B22222',
            ha='left',
            arrowprops=dict(arrowstyle='->', color='#B22222',
                            lw=0.6, connectionstyle='arc3,rad=0.2'))

ax.legend(loc='upper left', framealpha=0.95, edgecolor='lightgray')
ax.grid(True, alpha=0.25, linestyle='--')

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================
plt.savefig(OUTPUT_PDF, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=200)

print(f"\n[Saved] {OUTPUT_PDF}")
print(f"[Saved] {OUTPUT_PNG}")
print("\nSummary text for paper §3 / §6:")
print("-" * 70)
print(f"  The fitted baseline mu_b(t) ranges from {mu_b.min():.5f} to {mu_b.max():.5f},")
print(f"  with peaks in the final minute before each period break and")
print(f"  troughs in the first minute after the next period begins. The")
print(f"  homogeneous-Poisson mean is mu = {mu_homog:.5f}, but the highest")
print(f"  period-minute baseline rate is {rate_ratio:.1f} times the lowest one")
print(f"  (range / mean = {range_over_mean:.2f}), motivating the")
print(f"  inhomogeneous-baseline specification.")
