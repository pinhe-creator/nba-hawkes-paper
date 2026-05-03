#!/usr/bin/env python3
"""
================================================================================
Figure 6: Per-team alpha distribution + Model AIC comparison
================================================================================

Purpose:
  Two-panel summary for paper section §6.3:
    Panel (a): Distribution of per-team M1 alpha across 30 NBA teams.
               Shows that alpha is uniformly negligible across the league.
    Panel (b): AIC comparison of M0/M1/M2/M3, showing M3 dominates.

Inputs:
  phase4_per_team.csv          — 30 rows: team, n_real, n_evt, mu, alpha, beta, LogLik, AIC, BIC
  phase4_model_comparison.csv  — 4 rows: Model, k, LogLik, AIC, BIC

Outputs:
  figure6_team_alpha_aic.pdf
  figure6_team_alpha_aic.png

Usage:
  cd ~/Downloads
  python3 figure6_team_alpha_aic.py

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
TEAM_CSV = "phase4_per_team.csv"
COMP_CSV = "phase4_model_comparison.csv"
OUTPUT_PDF = "figure6_team_alpha_aic.pdf"
OUTPUT_PNG = "figure6_team_alpha_aic.png"

ALPHA_THRESHOLD = 1e-4

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
for f in [TEAM_CSV, COMP_CSV]:
    if not os.path.exists(f):
        sys.exit(f"[FATAL] {f} not found in current directory.")

team = pd.read_csv(TEAM_CSV)
comp = pd.read_csv(COMP_CSV)

print(f"[Loaded] {TEAM_CSV} with {len(team)} teams")
print(f"  Columns: {list(team.columns)}")
print(f"\n[Loaded] {COMP_CSV} with {len(comp)} models")
print(comp.to_string(index=False, float_format='%.4f'))

# Detect alpha column robustly
alpha_col = None
for cand in ['alpha_M1', 'alpha', 'alpha_hat', 'a']:
    if cand in team.columns:
        alpha_col = cand
        break
if alpha_col is None:
    sys.exit(f"[FATAL] No alpha column. Available: {list(team.columns)}")

team_col = None
for cand in ['team', 'TEAM', 'team_abbrev', 'TEAM_ABBREVIATION']:
    if cand in team.columns:
        team_col = cand
        break

alphas = team[alpha_col].values
print(f"\n[Per-team alpha stats]")
print(f"  n teams       = {len(alphas)}")
print(f"  min           = {alphas.min():.2e}")
print(f"  max           = {alphas.max():.2e}")
print(f"  median        = {np.median(alphas):.2e}")
print(f"  mean          = {alphas.mean():.2e}")
print(f"  > 1e-4        = {(alphas > 1e-4).sum()} / {len(alphas)}")
print(f"  > 1e-3        = {(alphas > 1e-3).sum()} / {len(alphas)}")
print(f"  > 1e-2        = {(alphas > 1e-2).sum()} / {len(alphas)}")

# ============================================================
# PLOT
# ============================================================
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   gridspec_kw={'width_ratios': [1.1, 1.0]})

# ----------------------------------------------------------------
# Panel (a): Per-team alpha histogram (log-x)
# ----------------------------------------------------------------
# Use log scale because alpha spans ~6 orders of magnitude (1e-7 to 1e-3)
# Take log10, with a floor at 1e-8 to avoid -inf
alpha_floor = 1e-8
alphas_safe = np.maximum(alphas, alpha_floor)
log_alphas = np.log10(alphas_safe)

# Build log-spaced bins from min to max
lo, hi = np.floor(log_alphas.min()), np.ceil(log_alphas.max())
bins = np.logspace(lo, hi, 25)

ax_a.hist(alphas_safe, bins=bins, color='#4A7AB8',
          edgecolor='black', alpha=0.78, linewidth=0.6)
ax_a.set_xscale('log')

# Threshold line
ax_a.axvline(ALPHA_THRESHOLD, color='#B22222', linestyle='--', lw=1.2,
             label=fr'Interpretation threshold $\alpha = 10^{{-4}}$')

# Annotate counts
n_above = (alphas > ALPHA_THRESHOLD).sum()
n_below = (alphas <= ALPHA_THRESHOLD).sum()
ax_a.text(0.02, 0.95,
          f"$\\widehat\\alpha \\leq 10^{{-4}}$: {n_below} / {len(alphas)} teams\n"
          f"$\\widehat\\alpha > 10^{{-4}}$: {n_above} / {len(alphas)} teams\n"
          f"max $\\widehat\\alpha$ = {alphas.max():.2e}",
          transform=ax_a.transAxes,
          fontsize=8.5, va='top', ha='left',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='lightgray', lw=0.5))

ax_a.set_xlabel(r'Per-team self-excitation $\widehat{\alpha}$ (M1 fit)')
ax_a.set_ylabel('Number of teams')
ax_a.set_title(f'(a) Distribution of per-team $\\widehat{{\\alpha}}$ across {len(alphas)} NBA franchises')
ax_a.legend(loc='upper right', framealpha=0.95, edgecolor='lightgray')
ax_a.grid(True, alpha=0.25, linestyle='--', which='both')

# ----------------------------------------------------------------
# Panel (b): Model AIC comparison
# ----------------------------------------------------------------
# Detect column names in comp
model_col = comp.columns[0]  # usually 'Model'
aic_col = 'AIC' if 'AIC' in comp.columns else None
if aic_col is None:
    sys.exit(f"[FATAL] No AIC column in {COMP_CSV}")

models = comp[model_col].tolist()
aics = comp[aic_col].values

# Clean up labels: strip "M0:", "M1:" prefixes if present, but keep them clear
short = []
for m in models:
    s = str(m)
    short.append(s)

# Sort by AIC ascending (best at top)
order = np.argsort(aics)
sorted_models = [short[i] for i in order]
sorted_aics = aics[order]

# Color: M3 winner highlighted
colors = []
for m in sorted_models:
    if 'M3' in m or 'Inhom' in m:
        colors.append('#B22222')   # winner red
    else:
        colors.append('#888888')   # gray

bars = ax_b.barh(range(len(sorted_models)), sorted_aics,
                 color=colors, edgecolor='black', alpha=0.85, linewidth=0.6)

ax_b.set_yticks(range(len(sorted_models)))
ax_b.set_yticklabels(sorted_models, fontsize=9.5)
ax_b.invert_yaxis()  # best (lowest AIC) at top

# Annotate AIC values + delta from best
best_aic = sorted_aics[0]
for i, (bar, val) in enumerate(zip(bars, sorted_aics)):
    delta = val - best_aic
    if delta < 0.5:
        label = f' {val:,.0f}  (best)'
    else:
        label = f' {val:,.0f}  (Δ = {delta:+,.0f})'
    ax_b.text(val, i, label, va='center', ha='left', fontsize=8.5)

ax_b.set_xlabel('AIC (lower is better)')
ax_b.set_title('(b) Model comparison by AIC')
ax_b.grid(True, alpha=0.25, linestyle='--', axis='x')

# Add some right-margin so labels don't get cut off
xmax = ax_b.get_xlim()[1]
ax_b.set_xlim(ax_b.get_xlim()[0], xmax * 1.18)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================
plt.savefig(OUTPUT_PDF, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=200)

print(f"\n[Saved] {OUTPUT_PDF}")
print(f"[Saved] {OUTPUT_PNG}")

print("\n" + "=" * 70)
print("SUMMARY (paste into paper §6.3)")
print("=" * 70)
print(f"  Per-team M1 fits across {len(alphas)} NBA teams yield")
print(f"  alpha ranging from {alphas.min():.2e} to {alphas.max():.2e},")
print(f"  with median {np.median(alphas):.2e}; {n_above} of {len(alphas)} teams")
print(f"  exceed the interpretation threshold 1e-4. The inhomogeneous")
print(f"  Poisson M3 dominates all alternatives by AIC, with the next-best")
print(f"  competing model trailing by Delta-AIC = {sorted_aics[1] - sorted_aics[0]:,.0f}.")
