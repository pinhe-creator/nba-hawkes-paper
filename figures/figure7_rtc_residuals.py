#!/usr/bin/env python3
"""
================================================================================
Figure 7: Random-Time-Change (RTC) Residual QQ plots — R4 Robustness Check
================================================================================

Purpose:
  Two-panel QQ plot comparing RTC residuals against the Exp(1) reference for
  two competing models:
    Panel (a): M1 self-only Hawkes      (KS = 0.07)
    Panel (b): M3 inhomogeneous Poisson (KS = 0.10)

  Under the correct model, the RTC residuals should be i.i.d. Exp(1).
  Both models depart from the diagonal at large quantiles, but neither
  can be rejected as the "correct" specification — supporting the §7.4
  conclusion that R4 is a SECONDARY diagnostic, not primary evidence.

Inputs:
  residuals_hawkes.npy   — RTC residuals from Hawkes M1 fit (n ≈ 122,000)
  residuals_b2.npy       — RTC residuals from M3 inhom Poisson fit

Outputs:
  figure7_rtc_residuals.pdf
  figure7_rtc_residuals.png

Usage:
  cd ~/Downloads
  python3 figure7_rtc_residuals.py

Required: matplotlib, scipy, numpy
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
import os, sys

# ============================================================
# CONFIG
# ============================================================
HAWKES_FILE = "residuals_hawkes.npy"
M3_FILE = "residuals_b2.npy"
OUTPUT_PDF = "figure7_rtc_residuals.pdf"
OUTPUT_PNG = "figure7_rtc_residuals.png"

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
for f in [HAWKES_FILE, M3_FILE]:
    if not os.path.exists(f):
        sys.exit(f"[FATAL] {f} not found in current directory.\n"
                 f"        Run 02_phase2_hawkes_v4.py first to generate it.")

res_hawkes = np.load(HAWKES_FILE)
res_m3 = np.load(M3_FILE)

# Drop non-positive residuals (defensive — RTC residuals should be positive)
res_hawkes = res_hawkes[res_hawkes > 0]
res_m3 = res_m3[res_m3 > 0]

print(f"[Loaded]")
print(f"  Hawkes residuals: n = {len(res_hawkes):,}")
print(f"    mean = {res_hawkes.mean():.4f} (Exp(1) mean = 1)")
print(f"    var  = {res_hawkes.var():.4f} (Exp(1) var  = 1)")
print(f"  M3 residuals    : n = {len(res_m3):,}")
print(f"    mean = {res_m3.mean():.4f}")
print(f"    var  = {res_m3.var():.4f}")

# KS test
ks_h, p_h = kstest(res_hawkes, 'expon', args=(0, 1))
ks_m, p_m = kstest(res_m3, 'expon', args=(0, 1))
print(f"\n[KS tests vs Exp(1)]")
print(f"  Hawkes M1: KS = {ks_h:.4f}  p = {p_h:.3g}")
print(f"  M3       : KS = {ks_m:.4f}  p = {p_m:.3g}")

# ============================================================
# QQ-plot helper
# ============================================================
def qq_plot(ax, residuals, title, color='steelblue', max_points=10000,
            tail_cap_pct=99.0):
    """
    Plot empirical residual quantiles vs theoretical Exp(1) quantiles.
    Subsamples to max_points to keep figure size reasonable.
    Caps axis at tail_cap_pct percentile to avoid the rare tail
    dominating the visual.
    """
    sorted_res = np.sort(residuals)
    n = len(sorted_res)

    # Subsample for plotting if n is huge
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        sorted_res = sorted_res[idx]
        n_plot = max_points
    else:
        n_plot = n

    # Theoretical Exp(1) quantiles using midpoint plotting positions
    p = (np.arange(n_plot) + 0.5) / n_plot
    theo = -np.log(1 - p)

    # Scatter (deeper alpha for visibility)
    ax.plot(theo, sorted_res, '.', alpha=0.55, ms=2.8, color=color, zorder=3)

    # Reference line y = x — DARK GRAY DASHED, behind scatter
    ax_max_data = max(theo.max(), sorted_res.max())
    ax.plot([0, ax_max_data], [0, ax_max_data],
            color='#444444', linestyle='--', lw=1.0, zorder=2,
            label='Exp(1) reference')

    # Cap axis range at tail_cap_pct percentile to suppress tail dominance
    cap = max(np.percentile(sorted_res, tail_cap_pct),
              np.percentile(theo, tail_cap_pct))
    cap = max(cap, 5.0)  # ensure at least visible range

    ax.set_xlabel(r'Theoretical Exp(1) quantile')
    ax.set_ylabel('Empirical residual quantile')
    ax.set_title(title)
    ax.set_xlim(0, cap * 1.02)
    ax.set_ylim(0, cap * 1.02)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='lightgray')
    ax.grid(True, alpha=0.25, linestyle='--')

    return n_plot

# ============================================================
# PLOT
# ============================================================
fig, (ax_h, ax_m) = plt.subplots(1, 2, figsize=(11, 5))

n_h = qq_plot(
    ax_h, res_hawkes,
    title=f'(a) M1 self-only Hawkes RTC residuals\n'
          f'$n = {len(res_hawkes):,}$,  KS $= {ks_h:.3f}$',
    color='#1F4E79'  # deep blue
)

n_m = qq_plot(
    ax_m, res_m3,
    title=f'(b) M3 inhomogeneous Poisson RTC residuals\n'
          f'$n = {len(res_m3):,}$,  KS $= {ks_m:.3f}$',
    color='#B22222'  # crimson
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================
plt.savefig(OUTPUT_PDF, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=200)

print(f"\n[Saved] {OUTPUT_PDF}")
print(f"[Saved] {OUTPUT_PNG}")

print("\n" + "=" * 70)
print("SUMMARY (paste into paper §7.4)")
print("=" * 70)
print(f"  RTC residuals (Hawkes M1): n = {len(res_hawkes):,}, mean = {res_hawkes.mean():.3f},")
print(f"    KS = {ks_h:.4f}, p = {p_h:.2e}.")
print(f"  RTC residuals (M3):        n = {len(res_m3):,}, mean = {res_m3.mean():.3f},")
print(f"    KS = {ks_m:.4f}, p = {p_m:.2e}.")
print(f"  Both reject Exp(1) at the conventional 5% level, but the small difference")
print(f"  in KS statistic is uninformative as primary evidence given the sample size.")
