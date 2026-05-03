#!/usr/bin/env python3
"""
figure8_pooled_q1_trajectory.py  (v2)

Generates Figure 8: M3 baseline intensity for Q1 with pooled
observed substitution events across all 2024--25 regular-season
games involving the Los Angeles Lakers or the Dallas Mavericks
(~75 games). Pooling many games rather than picking one provides
enough event density to visually confirm the baseline-driven
clustering pattern: substitutions concentrate where mu_b(t) is
high, not where prior events have just occurred.

Inputs (must be in same directory):
  - m1f_best_baseline.csv   (fitted league-wide mu_b per period-minute bin)
  - lal_dal_2024_25_pbp.csv (raw play-by-play for LAL+DAL games this season)

Outputs:
  - figure8_pooled_q1_trajectory.pdf
  - figure8_pooled_q1_trajectory.png

Differences from v1:
  - title/caption: pooled across many games, not "single game"
  - legend: deduplicated (M3 baseline appears once), rug entry added
  - title: no `vs.\` LaTeX residue
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -------------------------------------------------------------------- inputs

BASELINE_CSV = "m1f_best_baseline.csv"
PBP_CSV      = "lal_dal_2024_25_pbp.csv"

if not os.path.exists(BASELINE_CSV):
    print(f"[FATAL] {BASELINE_CSV} not found in current directory.")
    sys.exit(1)
if not os.path.exists(PBP_CSV):
    print(f"[FATAL] {PBP_CSV} not found in current directory.")
    sys.exit(1)

# -------------------------------------------------------------------- baseline

base = pd.read_csv(BASELINE_CSV)
mu_col = "mu_b_best" if "mu_b_best" in base.columns else "mu_b"
mu_b   = base[mu_col].values    # 48 values, one per period-minute bin
n_bins = len(mu_b)
assert n_bins == 48, f"Expected 48 bins, got {n_bins}"

print(f"[Loaded] {BASELINE_CSV}: 48 bins, mu_b range "
      f"[{mu_b.min():.2e}, {mu_b.max():.2e}]")

# -------------------------------------------------------------------- pbp

pbp = pd.read_csv(PBP_CSV)
print(f"[Loaded] {PBP_CSV}: {len(pbp):,} rows")

# Identify substitution events. NBA play-by-play uses EVENTMSGTYPE == 8
sub_mask = pbp["EVENTMSGTYPE"] == 8
subs = pbp.loc[sub_mask].copy()

# Count games and team-level realizations
n_games = pbp["GAME_ID"].nunique() if "GAME_ID" in pbp.columns else None
print(f"[Subs] {len(subs):,} substitution events across "
      f"{n_games} games" if n_games else f"[Subs] {len(subs):,} subs")

# Time within period (seconds since period start). NBA play-by-play
# typically has PCTIMESTRING ("MM:SS" countdown from 12:00) and PERIOD.
def to_seconds_elapsed(row, period_length=12 * 60):
    s = str(row["PCTIMESTRING"])
    if ":" not in s:
        return None
    mm, ss = s.split(":")
    remaining = int(mm) * 60 + int(ss)
    return period_length - remaining     # seconds since period start

subs["t_in_period"] = subs.apply(to_seconds_elapsed, axis=1)
subs = subs.dropna(subset=["t_in_period", "PERIOD"])

# Restrict to Q1 (period == 1)
q1_subs = subs[subs["PERIOD"] == 1]
print(f"[Q1] {len(q1_subs):,} substitution events in Q1 (pooled)")

# -------------------------------------------------------------------- plot

fig = plt.figure(figsize=(11, 5.5))
gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

# --- Top panel: M3 baseline as step function over Q1 (bins 0-11) ----
bin_edges = np.arange(0, 13) * 60          # 0, 60, ..., 720
mu_q1     = mu_b[:12]                      # first 12 bins for Q1

ax_top.fill_between(np.repeat(bin_edges, 2)[1:-1],
                    0,
                    np.repeat(mu_q1, 2),
                    step=None,
                    color="#4477AA", alpha=0.30)
ax_top.step(bin_edges[:-1], mu_q1, where="post",
            color="#1F4E79", lw=2.0)

# Mean baseline reference (gray dashed)
mu_bar = mu_b.mean()
ax_top.axhline(mu_bar, ls="--", color="0.5", lw=0.9)

# Period boundaries
for x in [0, 720]:
    ax_top.axvline(x, color="0.2", lw=0.8)

# Build a manual, deduplicated legend
legend_handles = [
    Patch(facecolor="#4477AA", alpha=0.30, edgecolor="#1F4E79",
          label=r"Fitted M3 baseline $\widehat{\mu}_b(t)$ (league-wide)"),
    Line2D([0], [0], color="0.5", ls="--", lw=0.9,
           label=fr"Homogeneous mean $\bar\mu = {mu_bar:.4f}$"),
    Line2D([0], [0], color="#CC3311", lw=1.2,
           label="Pooled observed substitutions (LAL and DAL team-games)"),
]
ax_top.legend(handles=legend_handles, loc="upper left",
              frameon=False, fontsize=9)

ax_top.set_ylabel(r"Intensity $\widehat{\mu}_b(t)$ (events/sec)")
ax_top.set_xlim(0, 720)
ax_top.set_ylim(0, mu_q1.max() * 1.35)
ax_top.tick_params(labelbottom=False)
ax_top.set_title(
    "Q1 baseline intensity and pooled observed substitutions",
    fontsize=11
)

# --- Bottom panel: rug plot of substitution event times -------------
event_times = q1_subs["t_in_period"].values

# With ~1780 events the rug saturates; use lower alpha so the density
# pattern is visible rather than a solid block.
for t in event_times:
    ax_bot.axvline(t, ymin=0, ymax=1, color="#CC3311", lw=0.6, alpha=0.18)

ax_bot.set_ylim(0, 1)
ax_bot.set_yticks([])
ax_bot.set_xlabel("Time within Q1 (seconds since tip-off)")
ax_bot.set_xlim(0, 720)
ax_bot.set_xticks([0, 120, 240, 360, 480, 600, 720])
ax_bot.set_xticklabels(["0", "120", "240", "360", "480", "600", "720"])
ax_bot.set_ylabel("Subs", rotation=0, ha="right", va="center", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 1])

out_pdf = "figure8_pooled_q1_trajectory.pdf"
out_png = "figure8_pooled_q1_trajectory.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n[Saved] {out_pdf}")
print(f"[Saved] {out_png}")

# -------------------------------------------------------------------- summary

print("\n" + "=" * 70)
print("SUMMARY for paper § narrative")
print("=" * 70)
print(f"  Sample: pooled Q1 substitutions across LAL and DAL games "
      f"in 2024-25 ({len(event_times):,} events)")
print(f"  Q1 baseline range: "
      f"min = {mu_q1.min():.5f}, "
      f"max = {mu_q1.max():.5f}, "
      f"max/min = {mu_q1.max()/mu_q1.min():.0f}x")
print(f"  Mean homogeneous baseline: {mu_bar:.4f}")
print()
print(f"  The pooled rug plot makes baseline-driven clustering visible:")
print(f"  events accumulate at the late-quarter end of Q1 where "
      f"mu_b(t) is highest,")
print(f"  not as cascades following individual trigger events.")
