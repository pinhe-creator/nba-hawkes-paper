#!/usr/bin/env python3
"""
================================================================================
Figure 5: Parametric Bootstrap LR_sims Distribution
================================================================================

Purpose:
  Visualize the parametric bootstrap null distribution of the LR statistic
  under H0: alpha = 0 (data ~ M3 inhomogeneous Poisson). Two-panel figure:
    Panel (a): Histogram of all 1000 LR_sims with LR_obs = 0 marked.
    Panel (b): Same data on log-1+x scale to show the right tail structure.

  Key narrative:
    - LR_obs = 0 (boundary value, M3-anchor enforced)
    - 49.4% of bootstrap reps yield LR_sim = 0 exactly
    - 50.6% yield LR_sim > 0 due to finite-sample clustering / EM overfit under H0
    - p-value = 1.000 because all LR_sims >= LR_obs by construction
    - This provides no evidence against H0; p = 1.000 is mechanical, not
      unusually strong proof of H0

Inputs:
  bootstrap_v4c_results.csv — 1000 rows from completed bootstrap

Outputs:
  figure5_bootstrap_distribution.pdf
  figure5_bootstrap_distribution.png

Usage:
  cd ~/Downloads
  python3 figure5_bootstrap_distribution.py

Required: matplotlib, pandas, numpy
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = "bootstrap_v4c_results.csv"
OUTPUT_PDF = "figure5_bootstrap_distribution.pdf"
OUTPUT_PNG = "figure5_bootstrap_distribution.png"

LR_OBS = 0.0  # observed LR (boundary value, M3-anchor enforced)

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
# LOAD
# ============================================================
if not os.path.exists(INPUT_CSV):
    sys.exit(f"[FATAL] {INPUT_CSV} not found")

df = pd.read_csv(INPUT_CSV)
print(f"[Loaded] {INPUT_CSV} with {len(df)} rows")
print(f"  Columns: {list(df.columns)}")

LR_sims = df['LR_sim'].values
B = len(LR_sims)

# Stats
print(f"\n[LR_sim distribution]")
print(f"  B           = {B}")
print(f"  min         = {LR_sims.min():+.4f}")
print(f"  q01         = {np.quantile(LR_sims, 0.01):+.4f}")
print(f"  q05         = {np.quantile(LR_sims, 0.05):+.4f}")
print(f"  q25         = {np.quantile(LR_sims, 0.25):+.4f}")
print(f"  median      = {np.median(LR_sims):+.4f}")
print(f"  q75         = {np.quantile(LR_sims, 0.75):+.4f}")
print(f"  q95         = {np.quantile(LR_sims, 0.95):+.4f}")
print(f"  q99         = {np.quantile(LR_sims, 0.99):+.4f}")
print(f"  max         = {LR_sims.max():+.4f}")
print(f"  mean        = {LR_sims.mean():+.4f}")
print(f"  std         = {LR_sims.std():+.4f}")

# Distribution shape
n_at_zero = np.sum(np.abs(LR_sims) < 1e-6)
n_positive = np.sum(LR_sims >= 1e-6)
n_negative = np.sum(LR_sims < -1e-6)
print(f"\n[Distribution shape]")
print(f"  At zero (|LR| < 1e-6): {n_at_zero}/{B} ({100*n_at_zero/B:.1f}%)")
print(f"  Strictly positive:      {n_positive}/{B} ({100*n_positive/B:.1f}%)")
print(f"  Negative (should be 0): {n_negative}/{B} ({100*n_negative/B:.1f}%)")

# p-value
n_extreme = np.sum(LR_sims >= LR_OBS)
p_value = (1 + n_extreme) / (1 + B)
print(f"\n[p-value]")
print(f"  #(LR_sim >= LR_obs=0): {n_extreme}/{B}")
print(f"  p = (1 + {n_extreme}) / (1 + {B}) = {p_value:.4f}")

# ============================================================
# PLOT
# ============================================================
fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.8))

# ----------------------------------------------------------------
# Panel (a): Linear scale histogram with zoom on body
# ----------------------------------------------------------------
# Cap at q99 for visualization (max=17 would make it impossible to read)
xmax_lin = max(np.quantile(LR_sims, 0.99), 6.0)
bins_lin = np.linspace(0, xmax_lin, 50)

# Panel (a): all bars stay BLUE — first bin contains exact-zero mass
# plus small positive LR values; we visualize the exact-zero mass
# separately as a red stem to avoid misleading the reader.
n, bins, patches = ax_lin.hist(np.clip(LR_sims, 0, xmax_lin),
                                bins=bins_lin,
                                color='#1F4E79', edgecolor='black',
                                alpha=0.78, linewidth=0.5,
                                zorder=2)

# Exact-zero point mass: red stem at x=0 with height = n_at_zero.
# Stem is annotated directly with text; NOT added to legend (avoids clutter).
ax_lin.vlines(0, 0, n_at_zero, color='#B22222', lw=4.5, zorder=4)
ax_lin.plot([0], [n_at_zero], 'o', color='#B22222', ms=7, zorder=5,
            markeredgecolor='black', markeredgewidth=0.5)

# Direct annotation pointing at the stem top
ax_lin.annotate(f'Exact $\\mathrm{{LR}}=0$ mass\n$= {n_at_zero}/{B}$ ({100*n_at_zero/B:.1f}\\%)',
                xy=(0, n_at_zero),
                xytext=(0.18 * xmax_lin, n_at_zero * 0.92),
                fontsize=8.5, color='#B22222',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='#B22222', lw=0.6),
                arrowprops=dict(arrowstyle='->', color='#B22222', lw=0.6,
                                connectionstyle='arc3,rad=-0.15'))

# LR_obs vertical reference line (slightly offset so it doesn't hide the stem)
ax_lin.axvline(LR_OBS, color='#B22222', linestyle='--', lw=1.0, zorder=3,
               alpha=0.5,
               label=f'$\\mathrm{{LR}}_{{\\mathrm{{obs}}}} = {LR_OBS:.3f}$ (boundary)')

# p-value box — placed at lower right (right side is sparse since most mass is at zero)
ax_lin.text(0.97, 0.50,
            f'$p$-value $= {p_value:.3f}$\n(boundary, $B = {B}$)',
            transform=ax_lin.transAxes,
            fontsize=9, va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='lightgray', lw=0.5))

ax_lin.set_xlabel(r'Simulated likelihood ratio $\mathrm{LR}_b$')
ax_lin.set_ylabel('Count')
ax_lin.set_title(f'(a) Bootstrap null distribution (values above $\\mathrm{{LR}} = {xmax_lin:.1f}$ clipped for display)')
ax_lin.set_xlim(-xmax_lin * 0.03, xmax_lin)
ax_lin.legend(loc='upper center', framealpha=0.95, edgecolor='lightgray')
ax_lin.grid(True, alpha=0.25, linestyle='--', axis='y')

# ----------------------------------------------------------------
# Panel (b): log(1+LR) scale to show full right-tail structure
# ----------------------------------------------------------------
# Use log(1 + LR_sim) transform (handles 0 mass cleanly)
log_LR = np.log10(1 + LR_sims)
xmax_log = log_LR.max() * 1.05

bins_log = np.linspace(0, xmax_log, 35)

n2, bins2, patches2 = ax_log.hist(log_LR,
                                    bins=bins_log,
                                    color='#1F4E79', edgecolor='black',
                                    alpha=0.78, linewidth=0.5,
                                    zorder=2)

# Exact-zero mass as red stem at x=0 (log10(1+0) = 0).
# Note: in log scale the stem is tall — that's correct, since 494
# is the bulk of the distribution.
ax_log.vlines(0, 0.5, n_at_zero, color='#B22222', lw=4.5, zorder=4)
ax_log.plot([0], [n_at_zero], 'o', color='#B22222', ms=7, zorder=5,
            markeredgecolor='black', markeredgewidth=0.5)

# LR_obs vertical reference line
ax_log.axvline(np.log10(1 + LR_OBS), color='#B22222', linestyle='--', lw=1.0,
               alpha=0.5, zorder=3,
               label=r'$\log_{10}(1 + \mathrm{LR}_{\mathrm{obs}}) = 0$')

# Mark q95 and q99
q95 = np.quantile(LR_sims, 0.95)
q99 = np.quantile(LR_sims, 0.99)
ax_log.axvline(np.log10(1 + q95), color='gray', linestyle=':', lw=1.0, alpha=0.6,
               label=f'q95 = {q95:.2f}')
ax_log.axvline(np.log10(1 + q99), color='gray', linestyle=':', lw=1.0, alpha=0.6,
               label=f'q99 = {q99:.2f}')

# Annotate max
log_max = np.log10(1 + LR_sims.max())
ax_log.annotate(f'max = {LR_sims.max():.2f}',
                xy=(log_max, 1.5),
                xytext=(log_max - 0.3, 30),
                fontsize=8.5, color='#444444', ha='right',
                arrowprops=dict(arrowstyle='->', color='#444444', lw=0.5))

ax_log.set_xlabel(r'$\log_{10}(1 + \mathrm{LR}_b)$')
ax_log.set_ylabel('Count')
ax_log.set_title(f'(b) Right-tail structure (log scale, full range)')
ax_log.legend(loc='upper right', framealpha=0.95, edgecolor='lightgray')
ax_log.grid(True, alpha=0.25, linestyle='--', axis='y')
ax_log.set_yscale('log')

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================
plt.savefig(OUTPUT_PDF, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=200)

print(f"\n[Saved] {OUTPUT_PDF}")
print(f"[Saved] {OUTPUT_PNG}")

print("\n" + "=" * 70)
print("SUMMARY (paste into paper §6.4)")
print("=" * 70)
print(f"  Parametric bootstrap with B = {B} replications under H0:")
print(f"  alpha = 0 yields a null distribution of the likelihood-ratio")
print(f"  statistic with median {np.median(LR_sims):+.4f}, mean {LR_sims.mean():+.4f},")
print(f"  q95 = {q95:.2f}, q99 = {q99:.2f}, max = {LR_sims.max():.2f}.")
print(f"  Of the {B} replications, {n_at_zero} ({100*n_at_zero/B:.1f}%) yield LR_sim = 0")
print(f"  exactly, indicating that the M3-anchor — which guarantees")
print(f"  LR_b >= 0 by construction — defeated all 5 EM initializations")
print(f"  in those replications. The observed LR_obs = 0 lies at the")
print(f"  lower boundary of the bootstrap distribution and ties with the")
print(f"  point mass at zero, yielding the mechanical boundary")
print(f"  p-value of (1 + {n_extreme}) / (1 + {B}) = {p_value:.4f}.")
