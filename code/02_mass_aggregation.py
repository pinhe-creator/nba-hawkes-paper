"""
================================================================================
Script: Phase B — Timeout-Substitution Simultaneity Analysis (FULL LEAGUE)
================================================================================

Purpose:
  Quantify the time gap between each mass-aggregated substitution event and
  the most recent preceding timeout in the SAME game.

  Hypothesis: If a large fraction of subs occur within ~1 second of (the same
  recorded second as) a timeout, this represents institutional simultaneity.
  Hawkes models cannot represent this because they assume lagged causal
  excitation, not same-clock-second co-occurrence.

  This is a mechanistic diagnostic for interpreting why the external-trigger
  Hawkes specification finds α_ext ≈ 0: much of the timeout-substitution
  coupling appears at the same recorded second, whereas the Hawkes kernel
  assigns excitation only to strictly lagged events. This is interpretation,
  not a formal proof, and §4 does not depend on this script.

Inputs:
  filtered_3_seasons.csv.gz  — Full 3-season league play-by-play (3,690 games)

  This file is the SAME source phase4 uses, ensuring substitutions and
  timeouts come from one consistent dataset.

Outputs:
  step_B_simultaneity.png         — Histogram + ECDF + categorical bar
  simul_data.csv                  — Per-sub time gaps (with same-second flag)
  simul_summary.csv               — Categorical breakdown
  simul_key_metrics.csv           — Headline statistics for paper / Phase 4 use

Method:
  For each mass-aggregated substitution, compute:
    delta_prev = t_sub - t_last_preceding_timeout    (if any timeout precedes)

  Note on PCTIMESTRING precision:
    The NBA play-by-play timestamp is second-level. When delta_prev = 0,
    the timeout and substitution are recorded in the SAME second. The
    physical ordering within that second is not resolvable. This is
    captured as 'same_second' rather than 'within_1s_after_timeout'.

  Compare ECDF against a uniform-random null model (single Monte Carlo
  realization, seed=42; the observed/null ratio is one realization of a
  stochastic benchmark, not a tight asymptotic estimate).

Key findings (printed at runtime — read simul_key_metrics.csv for exact
              values before quoting in the paper; do not hard-code these):
  - A large fraction of mass-aggregated subs co-occur with a preceding
    timeout in the same recorded second or within one second.
  - The observed rate is orders of magnitude larger than a uniform-random
    timing benchmark.
  - This indicates that timeout↔sub coupling is institutional simultaneity,
    not lagged causal excitation, and is therefore structurally outside
    the modeling space of standard Hawkes triggering kernels.

Execution time: ~2-5 minutes on full league.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# ---------- Configuration ----------
DATA_FILE = "filtered_3_seasons.csv.gz"
T_HORIZON = 2880.0   # 4 quarters × 720 sec each (regulation time)
RNG_SEED  = 42

# ---------- Header ----------
print("=" * 70)
print("STEP B: SIMULTANEITY ANALYSIS — FULL LEAGUE (timeout vs substitution)")
print("=" * 70)

if not os.path.exists(DATA_FILE):
    sys.exit(f"[FATAL] {DATA_FILE} not found in current directory.")

# ---------- Load + parse t_abs (same logic as phase4) ----------
print(f"\n[Loading] {DATA_FILE}")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"  {len(df):,} rows, {df['GAME_ID'].nunique():,} games")

def pctime_to_sec_left(s):
    if pd.isna(s):
        return np.nan
    try:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return np.nan

def absolute_seconds(period, pctime_str):
    sl = pctime_to_sec_left(pctime_str)
    if pd.isna(sl):
        return np.nan
    period = int(period)
    if period <= 4:
        return 720 * (period - 1) + (720 - sl)
    return 720 * 4 + 300 * (period - 5) + (300 - sl)

print("[Computing t_abs]")
df["t_abs"] = df.apply(lambda r: absolute_seconds(r["PERIOD"], r["PCTIMESTRING"]), axis=1)

# ---------- Mass-aggregate subs (same as phase4) ----------
print("[Mass-aggregating subs]")
subs = df[df["EVENTMSGTYPE"] == 8].dropna(subset=["t_abs", "PLAYER1_TEAM_ID"]).copy()
subs["TEAM_ID"] = subs["PLAYER1_TEAM_ID"].astype(int)
subs["TEAM"] = subs["PLAYER1_TEAM_ABBREVIATION"]
mass_subs = (
    subs.groupby(["GAME_ID", "TEAM_ID", "TEAM", "SEASON", "PERIOD", "t_abs"], as_index=False)
        .agg(n_players=("PLAYER1_ID", "count"))
)
mass_subs = mass_subs.sort_values(["GAME_ID", "TEAM_ID", "t_abs"]).reset_index(drop=True)
mass_reg = mass_subs[mass_subs["PERIOD"] <= 4].copy()
print(f"  Raw subs:         {len(subs):,}")
print(f"  Mass-aggregated:  {len(mass_subs):,}  (regulation: {len(mass_reg):,})")

# ---------- Extract timeouts (same as phase4) ----------
print("[Extracting timeouts]")
timeouts = df[df["EVENTMSGTYPE"] == 9].dropna(subset=["t_abs"])
timeouts_reg = timeouts[timeouts["PERIOD"] <= 4].copy()
print(f"  Total timeouts (regulation): {len(timeouts_reg):,}")

# ---------- Sanity check: mass_subs games must overlap timeouts games ----------
sub_games = set(mass_reg["GAME_ID"].unique())
to_games = set(timeouts_reg["GAME_ID"].unique())
overlap = sub_games & to_games
ms_only = sub_games - to_games
print(f"\n[Coverage check]")
print(f"  Games with subs:               {len(sub_games):,}")
print(f"  Games with timeouts:           {len(to_games):,}")
print(f"  Overlap (full coverage):       {len(overlap):,}")
print(f"  Sub games with no timeout:     {len(ms_only):,}")
if len(ms_only) > 0.05 * len(sub_games):
    print(f"  [WARNING] >5% sub-games lack timeouts — proceed with caution")
else:
    print(f"  [OK] coverage is consistent")

if len(sub_games) < 3000:
    print(f"  [WARNING] Expected ~3,690 games (full 3-season league),")
    print(f"           found {len(sub_games):,}. This is likely a SUBSET, not")
    print(f"           the full league. Paper numbers should reflect this scope.")

# ---------- Pre-build per-game timeout arrays ----------
timeout_per_game = {
    gid: np.sort(g["t_abs"].values)
    for gid, g in timeouts_reg.groupby("GAME_ID")
}

# ---------- For each mass-aggregated sub, compute delta_prev ----------
print(f"\n[Computing delta_prev for {len(mass_reg):,} mass-subs]")
records = []
for _, row in mass_reg.iterrows():
    gid     = row["GAME_ID"]
    t_sub   = row["t_abs"]
    team    = row["TEAM"]
    period  = row["PERIOD"]
    season  = row["SEASON"]
    n_play  = row["n_players"]

    timeouts_in_game = timeout_per_game.get(gid, np.array([]))

    if len(timeouts_in_game) == 0:
        delta_prev = np.nan
        delta_next = np.nan
    else:
        # All timeouts at OR BEFORE this sub second
        prior  = timeouts_in_game[timeouts_in_game <= t_sub]
        delta_prev = (t_sub - prior[-1]) if len(prior) > 0 else np.nan
        future = timeouts_in_game[timeouts_in_game >  t_sub]
        delta_next = (future[0] - t_sub) if len(future) > 0 else np.nan

    records.append({
        "GAME_ID": gid, "TEAM": team, "PERIOD": period, "SEASON": season,
        "t_sub": t_sub, "n_players": n_play,
        "delta_prev": delta_prev, "delta_next": delta_next,
    })

simul = pd.DataFrame(records)
print(f"  Done. {simul['delta_prev'].notna().sum():,} subs have a preceding timeout in-game")

# ---------- Categorical breakdown (with explicit same_second class) ----------
def classify(d):
    """
    Classify the gap to the most recent preceding timeout.

    same_second(0s)         : timeout and sub recorded in the same second.
                              Order within the second is not resolvable from
                              second-level PBP timestamps; this is treated
                              as institutional simultaneity.
    within_1s_after_timeout : delta_prev == 1 (one full second after).
    within_window(1-60s)    : 1 < delta_prev <= 60.
    brief_lag(60-180s)      : 60 < delta_prev <= 180.
    independent(>180s)      : delta_prev > 180.
    no_timeout_in_game      : NaN — no preceding timeout in this game.
    """
    if pd.isna(d):
        return "no_timeout_in_game"
    if d == 0:
        return "same_second(0s)"
    if d <= 1:
        return "within_1s_after_timeout"
    if d <= 60:
        return "within_window(1-60s)"
    if d <= 180:
        return "brief_lag(60-180s)"
    return "independent(>180s)"

simul["category"] = simul["delta_prev"].apply(classify)

CAT_ORDER = [
    "same_second(0s)",
    "within_1s_after_timeout",
    "within_window(1-60s)",
    "brief_lag(60-180s)",
    "independent(>180s)",
    "no_timeout_in_game",
]
summary = simul["category"].value_counts().rename("n").to_frame()
summary["pct"] = 100 * summary["n"] / len(simul)
summary = summary.reindex(CAT_ORDER).fillna(0)
print(f"\n[Categorical breakdown — gap to most recent preceding timeout]")
print(summary.to_string())

# ---------- Quantiles ----------
qs = simul["delta_prev"].dropna().quantile(
    [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
)
print(f"\n[Quantiles of delta_prev (sec, conditional on timeout existing)]")
print(qs.to_string())

# ---------- Headline statistics ----------
delta = simul["delta_prev"].dropna().values  # only with-timeout subs
n_total      = len(simul)
n_with_to    = len(delta)
n_same_sec   = int((delta == 0).sum())
n_le_1s      = int((delta <= 1).sum())   # includes same_second
n_le_60s     = int((delta <= 60).sum())

# Percentages computed two ways:
pct_le_1s_full   = 100 * n_le_1s   / n_total       # over ALL mass-subs
pct_le_1s_cond   = 100 * n_le_1s   / max(n_with_to, 1)  # over subs with a timeout
pct_le_60s_full  = 100 * n_le_60s  / n_total
pct_same_sec     = 100 * n_same_sec / n_total

# ---------- Null benchmark ----------
print(f"\n[Null benchmark: per-game uniform-random sub & timeout times, seed={RNG_SEED}]")
rng = np.random.default_rng(RNG_SEED)
null_deltas = []
null_total_count = 0
null_le_1s_count = 0
for gid, g in mass_reg.groupby("GAME_ID"):
    n_subs = len(g)
    n_to   = len(timeout_per_game.get(gid, []))
    if n_to == 0:
        continue
    fake_subs = rng.uniform(0, T_HORIZON, n_subs)
    fake_to   = np.sort(rng.uniform(0, T_HORIZON, n_to))
    for ts in fake_subs:
        null_total_count += 1
        prior = fake_to[fake_to <= ts]
        if len(prior) > 0:
            d = ts - prior[-1]
            null_deltas.append(d)
            if d <= 1:
                null_le_1s_count += 1

null_deltas = np.array(null_deltas)
# Two denominators (matching observed):
#   _full = same denominator as observed pct_le_1s_full (all fake subs, including
#           those with no prior fake timeout — which contribute 0 to numerator)
#   _cond = denominator restricted to fake subs that had a prior fake timeout
null_pct_le_1s_full = 100 * null_le_1s_count / max(null_total_count, 1)
null_pct_le_1s_cond = (
    100 * (null_deltas <= 1).mean() if len(null_deltas) > 0 else float("nan")
)

# Headline ratio: full-denom on both sides (consistent with observed full-denom)
ratio_obs_null = pct_le_1s_full / max(null_pct_le_1s_full, 1e-9)
# Reference ratio with conditional denom on both sides
ratio_obs_null_cond = pct_le_1s_cond / max(null_pct_le_1s_cond, 1e-9)

print(f"  % of subs within 1s of last timeout — OBSERVED (full):  {pct_le_1s_full:.2f}%")
print(f"  % of subs within 1s of last timeout — OBSERVED (cond):  {pct_le_1s_cond:.2f}%")
print(f"  % of subs within 1s of last timeout — NULL (full):      {null_pct_le_1s_full:.4f}%")
print(f"  % of subs within 1s of last timeout — NULL (cond):      {null_pct_le_1s_cond:.4f}%")
print(f"  Observed/null ratio (full denom on both):               {ratio_obs_null:.1f}x")
print(f"  Observed/null ratio (cond denom on both):               {ratio_obs_null_cond:.1f}x")
print(f"  Same-second co-occurrences (delta_prev == 0):           {pct_same_sec:.2f}%")

# ---------- Plots ----------
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# Histogram of delta_prev (clipped to 600s for readability)
ax = fig.add_subplot(gs[0, 0])
delta_clip = np.clip(delta, 0, 600)
ax.hist(delta_clip, bins=80, color="steelblue", edgecolor="black", alpha=0.75)
ax.set_xlabel("delta_prev = t_sub − t_last_preceding_timeout (sec)")
ax.set_ylabel("Count")
ax.set_title(f"Distribution of gap to preceding timeout (clip 600s, n={len(delta):,})")
ax.axvline(1, color="red", linestyle="--", lw=1.2, label="1s threshold")
ax.legend(fontsize=9)

# Zoomed histogram (0-30s)
ax = fig.add_subplot(gs[0, 1])
delta_zoom = delta[delta <= 30]
ax.hist(delta_zoom, bins=60, color="darkred", edgecolor="black", alpha=0.75)
ax.set_xlabel("delta_prev (0-30s)")
ax.set_ylabel("Count")
ax.set_title(f"Zoomed: subs within 30s of preceding timeout (n={len(delta_zoom):,})")
ax.axvline(1, color="red", linestyle="--", lw=1.2)

# ECDF: observed vs null
ax = fig.add_subplot(gs[1, 0])
sorted_obs  = np.sort(delta)
sorted_null = np.sort(null_deltas)
ax.plot(sorted_obs,  np.arange(1, len(sorted_obs) + 1) / len(sorted_obs),
        color="steelblue", lw=2, label="Observed")
ax.plot(sorted_null, np.arange(1, len(sorted_null) + 1) / len(sorted_null),
        color="orange", lw=2, label="Uniform null (1 MC realization)")
ax.set_xlim(0, 600)
ax.set_xlabel("delta_prev (sec, clipped at 600)")
ax.set_ylabel("ECDF")
ax.set_title("ECDF of gap to preceding timeout — observed vs null")
ax.axvline(1, color="red", linestyle="--", lw=1.0)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Categorical bar chart
ax = fig.add_subplot(gs[1, 1])
vals = summary.loc[CAT_ORDER, "pct"].values
ax.bar(range(len(CAT_ORDER)), vals,
       color=["red", "darkred", "steelblue", "lightblue", "lightgray", "white"],
       edgecolor="black")
ax.set_xticks(range(len(CAT_ORDER)))
ax.set_xticklabels(CAT_ORDER, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("% of mass-aggregated subs")
ax.set_title("Categorical breakdown of timeout-sub gap")
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

plt.savefig("step_B_simultaneity.png", dpi=120, bbox_inches="tight")
print(f"\n[Saved] step_B_simultaneity.png")

# ---------- Save CSVs ----------
simul.to_csv("simul_data.csv", index=False)
print("[Saved] simul_data.csv")

summary.to_csv("simul_summary.csv")
print("[Saved] simul_summary.csv")

# Key metrics for paper / cross-script reference (Phase 4 reads this)
key_metrics = pd.DataFrame({
    "metric": [
        "n_games",
        "n_subs_total",
        "n_subs_with_timeout",
        "n_same_second",
        "n_within_1s",
        "n_within_60s",
        "pct_same_second",
        "pct_within_1s_full",        # Headline number for paper §6
        "pct_within_1s_conditional",
        "pct_within_60s",
        "null_pct_within_1s_full",   # consistent denom with observed
        "null_pct_within_1s_cond",
        "observed_null_ratio_full",
        "observed_null_ratio_cond",
    ],
    "value": [
        len(sub_games),
        n_total,
        n_with_to,
        n_same_sec,
        n_le_1s,
        n_le_60s,
        pct_same_sec,
        pct_le_1s_full,
        pct_le_1s_cond,
        pct_le_60s_full,
        null_pct_le_1s_full,
        null_pct_le_1s_cond,
        ratio_obs_null,
        ratio_obs_null_cond,
    ],
})
key_metrics.to_csv("simul_key_metrics.csv", index=False)
print("[Saved] simul_key_metrics.csv")

# ---------- Verdict ----------
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"  • {pct_same_sec:.2f}% of mass-subs co-occur with a preceding timeout")
print(f"      in the SAME RECORDED SECOND (delta_prev = 0)")
print(f"  • {pct_le_1s_full:.2f}% occur in same second OR within 1 second after")
print(f"  • {pct_le_60s_full:.2f}% occur within 60 seconds after a timeout")
print(f"  • Observed rate is {ratio_obs_null:.0f}x higher than uniform-random null at ≤1s")
if pct_le_1s_full >= 25:
    print("  ✓ Strong simultaneity: timeout-sub coupling is concentrated at Δt ≤ 1s")
    print("  ✓ This supports the interpretation that timeout-sub coupling is")
    print("    institutional simultaneity rather than lagged Hawkes excitation.")
elif pct_le_1s_full >= 10:
    print("  ~ Moderate simultaneity: 10-25% of subs are simultaneous with timeouts")
else:
    print("  ✗ Weak simultaneity: simultaneity argument needs revision")
print()
print("NOTE on PCTIMESTRING precision:")
print("  Same-second co-occurrence does not imply strict causal ordering;")
print("  it indicates institutional bundling of timeouts and substitutions")
print("  at the second-level resolution of NBA play-by-play data.")