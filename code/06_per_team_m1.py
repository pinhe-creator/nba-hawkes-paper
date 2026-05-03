"""
================================================================================
Script 04: Phase 4 — Full-League Self-Hawkes + External-Trigger (DIAGNOSTIC)
================================================================================

Purpose:
  Full-league self-only Hawkes (M1) and Hawkes-with-external-trigger (M2)
  fits, with per-season and per-team breakdowns of M1.

  POSITIONING IN THE PAPER:
    This is a HOMOGENEOUS-BASELINE diagnostic. The paper's headline
    inferential result on self-excitation uses the M1f specification
    (48-bin time-varying baseline + Hawkes self-excitation) fit by
    08_fit_m1f.py with multi-start EM and tested by parametric bootstrap
    in 11_parametric_bootstrap_v3.py.

    The RESULTS to read off this script are:
      - In the homogeneous-baseline specification, both self-excitation
        and positive-lag timeout external excitation collapse to the
        boundary (α ≈ 0).
      - The 48-bin pooled clock-time baseline (M3) dominates M0/M1/M2
        on AIC/BIC, indicating that clock-time structure explains far
        more variation than homogeneous-baseline self- or timeout-
        excitation. The exact magnitude is reported at runtime as
        Delta_AIC and Delta_BIC in phase4_model_comparison.csv.
      - Per-season and per-team M1 estimates also collapse to α ≈ 0,
        ruling out the possibility that league-pooling masks a per-team
        self-excitation effect.

    The conclusions to AVOID writing from this script alone:
      - "timeouts have no effect on substitutions" — this is too strong.
        The correct statement is that timeouts do not generate detectable
        positive-lag Hawkes excitation once same-second co-occurrences
        are excluded (those are the institutional simultaneity diagnosed
        by step_B, which Hawkes kernels structurally cannot represent).
      - "α = 0 across NBA" — say α is below the interpretation threshold
        ALPHA_INTERPRET_THRESHOLD (= 1e-4), not literally zero.

Models:
  M0:    λ(t) = μ                                                        (k = 1)
  M1:    λ(t) = μ + α·β · Σ exp(-β(t-t_j))                                (k = 3)
  M2:    λ(t) = μ + α_s β_s · Σ exp(-β_s(t-t_j))
                  + α_e β_e · Σ exp(-β_e(t-u_k))                          (k = 5)
  M3:    λ(t) = μ_b[bin(t)]                                               (k = 48)
         (POOLED 48-bin: same time-of-game intensity shared across all
          game×team realizations; not team-specific or season-specific.)

Inputs:
  filtered_3_seasons.csv.gz   — Full 3-season league play-by-play (3,690 games)
  simul_key_metrics.csv       — Optional, from step_B; if present, the
                                simultaneity stats are read from there
                                instead of being hardcoded.

Outputs:
  phase4_self_params.csv         — M1 fit: μ, α, β, LL, AIC, BIC
  phase4_ext_params.csv          — M2 fit: best of multi-start
  phase4_per_season.csv          — M1 per-season (also M0 reference)
  phase4_per_team.csv            — M1 per-team (also M0 reference)
  phase4_model_comparison.csv    — M0/M1/M2/M3 comparison table
  phase4_full_league.png         — Diagnostic plots

Fixes vs the original Phase 4:
  [B1] α M-step uses the proper finite-window compensator denominator
       Σ_j (1 - exp(-β(T - t_j))) — applied in fit_self_quick (used by
       per-season, per-team, AND the league-level M1 refit) and in
       fit_ext_em_full (used by M2). Same fix as Phase 2 / Phase 3.
  [B3] No more hardcoded ll_self_full = -755409.0 / alpha_self_full = 0.
       M1 is re-fit at runtime on the same realizations. The model
       comparison table now reflects ACTUAL fitted values.
  [B4] No more hardcoded "30.2% within 1s of timeout (75x null)".
       Reads simul_key_metrics.csv if available (with metric labels
       pct_within_1s_full and observed_null_ratio_full from step_B).
  [I1] M2 uses multi-start EM (4 inits) instead of single-init.
  [I9] per-season and per-team outputs include LogLik, AIC, BIC,
       AND the M0 (homogeneous Poisson) LL on the same subsample.
  [I6] RTC residuals (where computed) include first-event gap.
  [C1] LAM_FLOOR unified to 1e-12.
  [C2] Model names M0 / M1 / M2 / M3 throughout (was "Hawkes" / "B1" / "B2").
  [C5] M3 explicitly labeled "POOLED 48-bin Inhom Poisson" to clarify it
       is not team- or season-specific.
  [Wording] No more "alpha = 0 across NBA" — always say α is below the
            interpretation threshold (negligible) rather than literally zero.
  [Robust] TEAM_ABBREVIATION fallback to TEAM_ID string if missing.
  [Iter]   Per-season / per-team max_iter raised to 200 (was 50–100). At
           100 iters Phase 3 still had α ≈ 5e-4; 200 iters lets per-team
           fits reach α ≈ 1e-5 boundary, avoiding spurious "above threshold"
           classifications driven by incomplete EM convergence.
  [Plot]   Diagnostic plot: log-scale α traces are floored at 1e-8 to
           avoid log(0) issues at the boundary.

EM caveat:
  β_self and β_ext are updated using the standard expected-waiting-time
  ratio (β_new = sum_pS / sum_pS_dt) rather than the strict finite-window
  MLE derivative, which would add a Σ_j (T - t_j) exp(-β(T - t_j))
  correction term. When α is below ALPHA_INTERPRET_THRESHOLD (which is
  the regime we observe empirically), the practical impact of this
  approximation is expected to be limited; β is weakly identified and
  not substantively interpreted in any case.

Boundary numerical artifact:
  Because EM converges asymptotically toward α = 0 from positive initial
  values, the maximized M1 likelihood may be very slightly below the
  exact M0 boundary likelihood (by ~0.01 LL units) — same for M2 vs M1.
  Theoretically LL(M2) ≥ LL(M1) ≥ LL(M0) since these are nested at the
  α = 0 boundary; in practice EM can fail to reach the exact boundary
  by O(0.01) LL units. The model comparison print explicitly flags this.

Execution time: ~30-60 minutes on a laptop.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, time
from collections import defaultdict
from scipy.stats import chi2

np.random.seed(42)

# ---------- Configuration ----------
DATA_FILE = "filtered_3_seasons.csv.gz"
SIMUL_METRICS_FILE = "simul_key_metrics.csv"   # produced by step_B
T_HORIZON = 2880.0
LAM_FLOOR = 1e-12
ALPHA_INTERPRET_THRESHOLD = 1e-4

print("=" * 70)
print("PHASE 4: FULL-LEAGUE Hawkes + External-Trigger (DIAGNOSTIC)")
print("=" * 70)

if not os.path.exists(DATA_FILE):
    sys.exit(f"[FATAL] {DATA_FILE} not found.")

# ============================================================
# Load + parse + mass-aggregate (identical to step_B / phase2 / phase3)
# ============================================================
print(f"\n[Loading] {DATA_FILE}")
df = pd.read_csv(DATA_FILE, low_memory=False)
n_games_loaded = df['GAME_ID'].nunique()
print(f"  {len(df):,} rows, {n_games_loaded:,} games")
if n_games_loaded < 3000:
    print(f"  [WARNING] Expected ~3,690 games, found {n_games_loaded:,}. May be subset.")

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

print("[Mass-aggregating subs]")
subs = df[df["EVENTMSGTYPE"] == 8].dropna(subset=["t_abs", "PLAYER1_TEAM_ID"]).copy()
subs["TEAM_ID"] = subs["PLAYER1_TEAM_ID"].astype(int)
n_missing_team = subs["PLAYER1_TEAM_ABBREVIATION"].isna().sum()
if n_missing_team > 0:
    print(f"  [INFO] {n_missing_team:,} sub rows missing TEAM_ABBREVIATION; falling back to TEAM_ID.")
subs["TEAM"] = subs["PLAYER1_TEAM_ABBREVIATION"].fillna(subs["TEAM_ID"].astype(str))

mass_subs = (
    subs.groupby(["GAME_ID", "TEAM_ID", "TEAM", "SEASON", "PERIOD", "t_abs"], as_index=False)
        .agg(n_players=("PLAYER1_ID", "count"))
)
mass_subs = mass_subs.sort_values(["GAME_ID", "TEAM_ID", "t_abs"]).reset_index(drop=True)
mass_reg = mass_subs[mass_subs["PERIOD"] <= 4].copy()
print(f"  Raw subs: {len(subs):,}  →  Mass-aggregated: {len(mass_subs):,}  "
      f"(regulation: {len(mass_reg):,})")

# Timeouts
print("[Extracting timeouts]")
timeouts_all = df[df["EVENTMSGTYPE"] == 9].dropna(subset=["t_abs"])
timeouts_reg = timeouts_all[timeouts_all["PERIOD"] <= 4].copy()
print(f"  Unique regulation timeouts: {len(timeouts_reg):,}")

timeout_per_game = {
    gid: np.sort(g["t_abs"].values.astype(float))
    for gid, g in timeouts_reg.groupby("GAME_ID")
}

# Build (sub_t, ext_t) realizations
n_at_or_after_T = (mass_reg["t_abs"] >= T_HORIZON).sum()
if n_at_or_after_T > 0:
    print(f"  [INFO] Dropping {n_at_or_after_T:,} regulation events at t_abs >= {T_HORIZON}.")

realizations = []
realization_meta = []
for (gid, tid), g_subs in mass_reg.groupby(["GAME_ID", "TEAM_ID"]):
    sub_t = np.sort(g_subs["t_abs"].values.astype(float))
    sub_t = sub_t[sub_t < T_HORIZON]
    ext_t = timeout_per_game.get(gid, np.array([]))
    ext_t = ext_t[ext_t < T_HORIZON]
    if len(sub_t) > 0:
        realizations.append((sub_t, ext_t))
        realization_meta.append({
            "GAME_ID": gid, "TEAM_ID": tid,
            "TEAM": g_subs["TEAM"].iloc[0],
            "SEASON": g_subs["SEASON"].iloc[0],
        })

n_real = len(realizations)
n_total_sub = sum(len(s) for s, _ in realizations)
print(f"\n[Realizations] {n_real:,}  |  total subs={n_total_sub:,}")

# ============================================================
# Pre-compute caches for M2 (Hawkes-with-External)
# ============================================================
print("\n[Building caches for M2]")
t0 = time.time()
self_dt_cache = []
ext_diff_cache = []
T_minus_sub_cache = []
T_minus_ext_cache = []
for sub_t, ext_t in realizations:
    n = len(sub_t)
    dt_self = np.zeros(n)
    if n > 1:
        dt_self[1:] = np.diff(sub_t)
    self_dt_cache.append(dt_self)
    T_minus_sub_cache.append(T_HORIZON - sub_t)
    T_minus_ext_cache.append(T_HORIZON - ext_t)
    # External kernels are evaluated only for STRICTLY EARLIER timeouts (u_k < t_i).
    # Same-second timeout-substitution pairs are intentionally not counted as
    # positive-lag excitation; they are handled by the step_B simultaneity
    # diagnostic, since Hawkes external kernels structurally cannot represent
    # u_k == t_i co-occurrence. The compensator integral, however, still
    # accumulates over u_k <= T (post-event excitation interval), which is
    # the correct point-process likelihood treatment.
    ext_diff = []
    if len(ext_t) > 0:
        for i in range(n):
            k = np.searchsorted(ext_t, sub_t[i], side='left')
            if k > 0:
                ext_diff.append((i, sub_t[i] - ext_t[:k]))
    ext_diff_cache.append(ext_diff)
print(f"  Done in {time.time() - t0:.1f}s")

# ============================================================
# fit_self_quick — used by league-level M1 + per-season + per-team
# WITH FIX [B1]: α M-step uses finite-window compensator denominator
# ============================================================
def fit_self_quick(reals, T, mu0=0.005, alpha0=0.3, beta0=1/180,
                   max_iter=200, tol=1e-7):
    """
    EM for self-only Hawkes (M1) with the [B1] fix:
      α_new = sum_pS / sum_compensator (NOT / n_total).
    """
    mu, alpha, beta = mu0, alpha0, beta0
    n_r = len(reals)
    for it in range(max_iter):
        sum_pB = 0.0
        sum_pS = 0.0
        sum_pS_dt = 0.0
        sum_compensator = 0.0
        for sub_t, _ in reals:
            n = len(sub_t)
            if n == 0:
                continue
            A = np.zeros(n)
            B = np.zeros(n)
            for i in range(1, n):
                dt = sub_t[i] - sub_t[i - 1]
                e = np.exp(-beta * dt)
                A[i] = e * (1 + A[i - 1])
                B[i] = e * (B[i - 1] + dt * (1 + A[i - 1]))
            lam = np.maximum(mu + alpha * beta * A, LAM_FLOOR)
            pB = mu / lam
            sum_pB += pB.sum()
            sum_pS += (1 - pB).sum()
            sum_pS_dt += (alpha * beta * B / lam).sum()
            sum_compensator += np.sum(1 - np.exp(-beta * (T - sub_t)))
        new_mu = max(sum_pB / (n_r * T), 1e-9)
        if sum_compensator > 1e-12:
            new_alpha = min(sum_pS / sum_compensator, 0.99)
        else:
            new_alpha = 0.0
        if sum_pS > 1e-10 and sum_pS_dt > 1e-12:
            new_beta = max(sum_pS / sum_pS_dt, 1e-7)
        else:
            new_beta = beta
        delta = max(abs(new_mu - mu), abs(new_alpha - alpha), abs(new_beta - beta))
        mu, alpha, beta = new_mu, new_alpha, new_beta
        if delta < tol:
            break
    return mu, alpha, beta

def hawkes_self_ll(reals, mu, alpha, beta, T):
    total = 0.0
    for sub_t, _ in reals:
        n = len(sub_t)
        if n == 0:
            total += -mu * T
            continue
        A = np.zeros(n)
        for i in range(1, n):
            A[i] = np.exp(-beta * (sub_t[i] - sub_t[i - 1])) * (1 + A[i - 1])
        lam = np.maximum(mu + alpha * beta * A, LAM_FLOOR)
        comp = alpha * np.sum(1 - np.exp(-beta * (T - sub_t)))
        total += np.sum(np.log(lam)) - mu * T - comp
    return total

# ============================================================
# fit_ext_em_full / hawkes_ext_ll_full — M2 with FIX [B1] for both α_s and α_e
# NOTE: These two functions intentionally use the GLOBAL `realizations`,
# `self_dt_cache`, `ext_diff_cache`, `T_minus_sub_cache`, `T_minus_ext_cache`,
# and `n_real`. They are full-league only by design — the multi-start EM in
# this script always fits M2 on the entire 3,690-game / 7,380-realization
# sample. Do not reuse these without parameterizing.
# ============================================================
def fit_ext_em_full(mu0, a_s0, b_s0, a_e0, b_e0, T,
                    max_iter=200, tol=1e-7, verbose=False):
    mu, a_s, b_s, a_e, b_e = mu0, a_s0, b_s0, a_e0, b_e0
    history = []
    for it in range(max_iter):
        sum_pB = 0.0
        sum_pS = 0.0
        sum_pE = 0.0
        sum_pS_dt = 0.0
        sum_pE_dt = 0.0
        sum_compensator_self = 0.0
        sum_compensator_ext = 0.0
        for r_idx, (sub_t, ext_t) in enumerate(realizations):
            n = len(sub_t)
            if n == 0:
                continue
            dt_self = self_dt_cache[r_idx]
            A = np.zeros(n)
            B = np.zeros(n)
            for i in range(1, n):
                e_b = np.exp(-b_s * dt_self[i])
                A[i] = e_b * (1 + A[i - 1])
                B[i] = e_b * (B[i - 1] + dt_self[i] * (1 + A[i - 1]))
            E = np.zeros(n)
            F = np.zeros(n)
            for (i, deltas) in ext_diff_cache[r_idx]:
                ee = np.exp(-b_e * deltas)
                E[i] = ee.sum()
                F[i] = (deltas * ee).sum()
            lam = np.maximum(mu + a_s * b_s * A + a_e * b_e * E, LAM_FLOOR)
            pB = mu / lam
            sum_pB += pB.sum()
            sum_pS += (a_s * b_s * A / lam).sum()
            sum_pE += (a_e * b_e * E / lam).sum()
            sum_pS_dt += (a_s * b_s * B / lam).sum()
            sum_pE_dt += (a_e * b_e * F / lam).sum()
            sum_compensator_self += np.sum(1 - np.exp(-b_s * T_minus_sub_cache[r_idx]))
            if len(ext_t) > 0:
                sum_compensator_ext += np.sum(1 - np.exp(-b_e * T_minus_ext_cache[r_idx]))
        new_mu = max(sum_pB / (n_real * T), 1e-9)
        new_a_s = (
            min(sum_pS / sum_compensator_self, 0.99)
            if sum_compensator_self > 1e-12 else 0.0
        )
        # α_e cap = 5.0 is a numerical safeguard; external triggering is
        # exogenous so there is no theoretical stationarity bound on α_e.
        new_a_e = (
            min(sum_pE / sum_compensator_ext, 5.0)
            if sum_compensator_ext > 1e-12 else 0.0
        )
        new_b_s = (
            max(sum_pS / sum_pS_dt, 1e-7)
            if sum_pS > 1e-10 and sum_pS_dt > 1e-12 else b_s
        )
        new_b_e = (
            max(sum_pE / sum_pE_dt, 1e-7)
            if sum_pE > 1e-10 and sum_pE_dt > 1e-12 else b_e
        )
        delta = max(
            abs(new_mu - mu), abs(new_a_s - a_s), abs(new_b_s - b_s),
            abs(new_a_e - a_e), abs(new_b_e - b_e)
        )
        mu, a_s, b_s, a_e, b_e = new_mu, new_a_s, new_b_s, new_a_e, new_b_e
        history.append((it, mu, a_s, b_s, a_e, b_e))
        if verbose and (it % 20 == 0 or it < 3):
            print(f"    iter {it:3d}: mu={mu:.6f} a_s={a_s:.6f} b_s={b_s:.6f} "
                  f"a_e={a_e:.6f} b_e={b_e:.6f}")
        if delta < tol:
            break
    return mu, a_s, b_s, a_e, b_e, history

def hawkes_ext_ll_full(mu, a_s, b_s, a_e, b_e, T):
    """Full-league M2 log-likelihood. Uses global caches by design."""
    total = 0.0
    for r_idx, (sub_t, ext_t) in enumerate(realizations):
        n = len(sub_t)
        if n == 0:
            continue
        A = np.zeros(n)
        dt_self = self_dt_cache[r_idx]
        for i in range(1, n):
            A[i] = np.exp(-b_s * dt_self[i]) * (1 + A[i - 1])
        E = np.zeros(n)
        for (i, deltas) in ext_diff_cache[r_idx]:
            E[i] = np.sum(np.exp(-b_e * deltas))
        lam = np.maximum(mu + a_s * b_s * A + a_e * b_e * E, LAM_FLOOR)
        comp_self = a_s * np.sum(1 - np.exp(-b_s * T_minus_sub_cache[r_idx]))
        comp_ext = (
            a_e * np.sum(1 - np.exp(-b_e * T_minus_ext_cache[r_idx]))
            if len(ext_t) > 0 else 0.0
        )
        total += np.sum(np.log(lam)) - mu * T - comp_self - comp_ext
    return total

# ============================================================
# (1) Re-fit M1 on full league (FIX [B3]: no more hardcoded values!)
# ============================================================
print("\n" + "=" * 70)
print("(1) M1: SELF-ONLY HAWKES — FULL LEAGUE")
print("=" * 70)
print("\n  Fitting M1 on all 7,380 realizations (was hardcoded; now refit)...")
t0 = time.time()
mu_M1, a_M1, b_M1 = fit_self_quick(realizations, T_HORIZON, max_iter=300, tol=1e-7)
ll_M1 = hawkes_self_ll(realizations, mu_M1, a_M1, b_M1, T_HORIZON)
elapsed_M1 = time.time() - t0
print(f"  μ      = {mu_M1:.6f}")
print(f"  α      = {a_M1:.6f}")
print(f"  β      = {b_M1:.6f}")
if a_M1 > ALPHA_INTERPRET_THRESHOLD:
    print(f"           half-life = {np.log(2)/b_M1:.0f}s")
else:
    print(f"           α < {ALPHA_INTERPRET_THRESHOLD:.0e} — β weakly identified, not interpreted")
print(f"  LL     = {ll_M1:.4f}")
print(f"  ({elapsed_M1:.1f}s)")
k_M1 = 3
aic_M1 = 2 * k_M1 - 2 * ll_M1
bic_M1 = k_M1 * np.log(n_total_sub) - 2 * ll_M1

# Save M1 params (FIX: was hardcoded -755409.0 / 0.0 / 0.00578 / 0.00628 before)
m1_df = pd.DataFrame([{
    "n_real": n_real, "n_evt": n_total_sub,
    "mu": mu_M1, "alpha": a_M1, "beta": b_M1,
    "LogLik": ll_M1, "AIC": aic_M1, "BIC": bic_M1,
    "alpha_above_threshold": bool(a_M1 > ALPHA_INTERPRET_THRESHOLD),
}])
m1_df.to_csv("phase4_self_params.csv", index=False)
print("[Saved] phase4_self_params.csv")

# M0
mu_hom = n_total_sub / (n_real * T_HORIZON)
ll_M0 = -mu_hom * n_real * T_HORIZON + n_total_sub * np.log(max(mu_hom, LAM_FLOOR))
k_M0 = 1
aic_M0 = 2 * k_M0 - 2 * ll_M0
bic_M0 = k_M0 * np.log(n_total_sub) - 2 * ll_M0
print(f"\n  [M0 reference] rate={mu_hom:.6f}  LL={ll_M0:.4f}  AIC={aic_M0:.2f}")

# M3
n_bins = 48
bin_w = T_HORIZON / n_bins
all_event_times = np.concatenate([s for s, _ in realizations])
event_bins = np.minimum((all_event_times // bin_w).astype(int), n_bins - 1)
counts = np.bincount(event_bins, minlength=n_bins)
rates_M3 = counts / (n_real * bin_w)
ll_M3 = (np.log(np.maximum(rates_M3[event_bins], LAM_FLOOR))).sum() - (rates_M3 * bin_w * n_real).sum()
k_M3 = n_bins
aic_M3 = 2 * k_M3 - 2 * ll_M3
bic_M3 = k_M3 * np.log(n_total_sub) - 2 * ll_M3
print(f"  [M3 reference] LL={ll_M3:.4f}  AIC={aic_M3:.2f}")

# ============================================================
# (2) M2: Hawkes-with-External — multi-start (FIX [I1])
# ============================================================
print("\n" + "=" * 70)
print("(2) M2: HAWKES-WITH-EXTERNAL — MULTI-START EM (4 inits)")
print("=" * 70)

starts = [
    dict(mu0=0.001, a_s0=0.05, b_s0=1/300, a_e0=0.10, b_e0=1/200),
    dict(mu0=0.003, a_s0=0.20, b_s0=1/180, a_e0=0.50, b_e0=1/120),
    dict(mu0=0.005, a_s0=0.50, b_s0=1/100, a_e0=1.00, b_e0=1/100),
    dict(mu0=0.0001, a_s0=0.80, b_s0=1/60, a_e0=2.00, b_e0=1/60),
]

best = None
all_results = []
for k_start, s in enumerate(starts):
    t0 = time.time()
    print(f"\n  Start {k_start + 1}: {s}")
    mu, a_s, b_s, a_e, b_e, hist = fit_ext_em_full(
        T=T_HORIZON, verbose=(k_start == 0), max_iter=200, **s
    )
    ll = hawkes_ext_ll_full(mu, a_s, b_s, a_e, b_e, T_HORIZON)
    print(f"  -> μ={mu:.6f}  α_s={a_s:.6f}  β_s={b_s:.6f}  "
          f"α_e={a_e:.6f}  β_e={b_e:.6f}  LL={ll:.4f}  ({time.time()-t0:.1f}s)")
    all_results.append((mu, a_s, b_s, a_e, b_e, ll, hist))
    if best is None or ll > best[5]:
        best = (mu, a_s, b_s, a_e, b_e, ll, hist)

mu_M2, a_s_M2, b_s_M2, a_e_M2, b_e_M2, ll_M2, hist_M2 = best
k_M2 = 5
aic_M2 = 2 * k_M2 - 2 * ll_M2
bic_M2 = k_M2 * np.log(n_total_sub) - 2 * ll_M2

print(f"\n[BEST M2 estimate]")
print(f"  μ        = {mu_M2:.6f}")
print(f"  α_self   = {a_s_M2:.6f}")
print(f"  β_self   = {b_s_M2:.6f}")
if a_s_M2 > ALPHA_INTERPRET_THRESHOLD:
    print(f"           half-life = {np.log(2)/b_s_M2:.0f}s")
else:
    print(f"           α_self < {ALPHA_INTERPRET_THRESHOLD:.0e} — β_self weakly identified")
print(f"  α_ext    = {a_e_M2:.6f}")
print(f"  β_ext    = {b_e_M2:.6f}")
if a_e_M2 > ALPHA_INTERPRET_THRESHOLD:
    print(f"           half-life = {np.log(2)/b_e_M2:.0f}s")
else:
    print(f"           α_ext < {ALPHA_INTERPRET_THRESHOLD:.0e} — β_ext weakly identified")
print(f"  LL = {ll_M2:.4f}  AIC = {aic_M2:.2f}  BIC = {bic_M2:.2f}")

# Save M2 params
m2_df = pd.DataFrame([{
    "n_real": n_real, "n_evt": n_total_sub,
    "mu": mu_M2,
    "alpha_self": a_s_M2, "beta_self": b_s_M2,
    "alpha_ext": a_e_M2, "beta_ext": b_e_M2,
    "LogLik": ll_M2, "AIC": aic_M2, "BIC": bic_M2,
    "alpha_self_above_threshold": bool(a_s_M2 > ALPHA_INTERPRET_THRESHOLD),
    "alpha_ext_above_threshold":  bool(a_e_M2 > ALPHA_INTERPRET_THRESHOLD),
}])
m2_df.to_csv("phase4_ext_params.csv", index=False)
print("[Saved] phase4_ext_params.csv")

# ============================================================
# Heuristic LR test M1 → M2 (FIX [B3]: now uses RUNTIME-fit M1, not hardcoded)
# ============================================================
print("\n" + "=" * 70)
print("HEURISTIC LR TEST: H0 (M1, α_ext = 0)  vs  H1 (M2, α_ext free)")
print("=" * 70)
LR_raw = 2 * (ll_M2 - ll_M1)
LR = max(0.0, LR_raw)
p_LR_chi2_2 = 1 - chi2.cdf(LR, df=2)
print(f"  LL(M1)     = {ll_M1:.4f}    [k=3, runtime-fit]")
print(f"  LL(M2)     = {ll_M2:.4f}    [k=5, multi-start best]")
print(f"  LR_raw     = {LR_raw:.6f}")
print(f"  LR_clipped = {LR:.6f}    (clipped to 0 if raw < 0)")
print(f"  p (χ²₂)    = {p_LR_chi2_2:.4g}    [HEURISTIC — see caveats]")
print()
print("  CAVEATS (same as Phase 3):")
print("    - Under H0: α_ext=0, β_ext is unidentified ⇒ χ²₂ is heuristic.")
print(f"    - Negative LR_raw indicates EM did not exactly reach the M1")
print(f"      boundary on the multi-start best M2 fit "
      f"({ll_M2 - ll_M1:+.4f} LL units).")
print("    - The §6 conclusion does not rest on this number alone;")
print("      it rests jointly with the simultaneity diagnostic in step_B.")

# ============================================================
# (3) Per-season M1 — FIX [I9]: include LL/AIC/BIC
# ============================================================
print("\n" + "=" * 70)
print("(3) PER-SEASON M1 ANALYSIS")
print("=" * 70)

season_real = defaultdict(list)
for (sub_t, ext_t), m in zip(realizations, realization_meta):
    season_real[m["SEASON"]].append((sub_t, ext_t))

print()
season_results = []
for season, sea_real in sorted(season_real.items()):
    t0 = time.time()
    mu_s, a_s_, b_s_ = fit_self_quick(sea_real, T_HORIZON, max_iter=200, tol=1e-7)
    ll_s = hawkes_self_ll(sea_real, mu_s, a_s_, b_s_, T_HORIZON)
    n_evt = sum(len(s) for s, _ in sea_real)
    aic_s = 2 * 3 - 2 * ll_s
    bic_s = 3 * np.log(n_evt) - 2 * ll_s
    # M0 reference on the same subsample
    n_real_s = len(sea_real)
    mu0_s = n_evt / (n_real_s * T_HORIZON)
    ll0_s = -mu0_s * n_real_s * T_HORIZON + n_evt * np.log(max(mu0_s, LAM_FLOOR))
    aic0_s = 2 * 1 - 2 * ll0_s
    bic0_s = 1 * np.log(n_evt) - 2 * ll0_s
    print(f"  {season}: n_real={n_real_s:,}  n_evt={n_evt:,}  "
          f"μ={mu_s:.6f}  α={a_s_:.6f}  β={b_s_:.6f}  "
          f"LL(M1)={ll_s:.2f}  ΔLL_M1-M0={ll_s-ll0_s:+.4f}  ({time.time()-t0:.1f}s)")
    season_results.append({
        "season": season, "n_real": n_real_s, "n_evt": n_evt,
        "mu_M1": mu_s, "alpha_M1": a_s_, "beta_M1": b_s_,
        "LogLik_M1": ll_s, "AIC_M1": aic_s, "BIC_M1": bic_s,
        "mu_M0": mu0_s, "LogLik_M0": ll0_s, "AIC_M0": aic0_s, "BIC_M0": bic0_s,
        "Delta_LL_M1_minus_M0": ll_s - ll0_s,
        "alpha_above_threshold": bool(a_s_ > ALPHA_INTERPRET_THRESHOLD),
    })
season_df = pd.DataFrame(season_results)
season_df.to_csv("phase4_per_season.csv", index=False)
print("[Saved] phase4_per_season.csv")

# ============================================================
# (4) Per-team M1 — FIX [I9]: include LL/AIC/BIC
# ============================================================
print("\n" + "=" * 70)
print("(4) PER-TEAM M1 ANALYSIS")
print("=" * 70)

team_real = defaultdict(list)
for (sub_t, ext_t), m in zip(realizations, realization_meta):
    team_real[m["TEAM"]].append((sub_t, ext_t))

print(f"\n  {len(team_real)} teams found. Fitting M1 for each (sorted by team)...")
team_results = []
for k, team in enumerate(sorted(team_real.keys())):
    tr = team_real[team]
    if len(tr) < 30:
        continue
    t0 = time.time()
    mu_t, a_t, b_t = fit_self_quick(tr, T_HORIZON, max_iter=200, tol=1e-7)
    ll_t = hawkes_self_ll(tr, mu_t, a_t, b_t, T_HORIZON)
    n_evt = sum(len(s) for s, _ in tr)
    aic_t = 2 * 3 - 2 * ll_t
    bic_t = 3 * np.log(n_evt) - 2 * ll_t
    # M0 reference on the same subsample
    n_real_t = len(tr)
    mu0_t = n_evt / (n_real_t * T_HORIZON)
    ll0_t = -mu0_t * n_real_t * T_HORIZON + n_evt * np.log(max(mu0_t, LAM_FLOOR))
    aic0_t = 2 * 1 - 2 * ll0_t
    bic0_t = 1 * np.log(n_evt) - 2 * ll0_t
    team_results.append({
        "team": team, "n_real": n_real_t, "n_evt": n_evt,
        "mu_M1": mu_t, "alpha_M1": a_t, "beta_M1": b_t,
        "LogLik_M1": ll_t, "AIC_M1": aic_t, "BIC_M1": bic_t,
        "mu_M0": mu0_t, "LogLik_M0": ll0_t,
        "AIC_M0": aic0_t, "BIC_M0": bic0_t,
        "Delta_LL_M1_minus_M0": ll_t - ll0_t,
        "alpha_above_threshold": bool(a_t > ALPHA_INTERPRET_THRESHOLD),
    })
    if k % 5 == 0:
        print(f"    {team}: n_real={n_real_t:,}  α={a_t:.6f}  μ={mu_t:.6f}  "
              f"ΔLL_M1-M0={ll_t-ll0_t:+.4f}  ({time.time()-t0:.1f}s)")

team_df = pd.DataFrame(team_results).sort_values("alpha_M1", ascending=False).reset_index(drop=True)
team_df.to_csv("phase4_per_team.csv", index=False)
print(f"[Saved] phase4_per_team.csv  ({len(team_df)} teams)")

print(f"\n[Top 5 teams by α]")
print(team_df.head(5).to_string(index=False, float_format='%.5f'))
print(f"\n[Bottom 5 teams by α]")
print(team_df.tail(5).to_string(index=False, float_format='%.5f'))
print(f"\n[Distribution of α across {len(team_df)} teams]")
print(team_df["alpha_M1"].describe().round(6).to_string())
n_above = (team_df["alpha_M1"] > ALPHA_INTERPRET_THRESHOLD).sum()
print(f"\n  Teams with α > {ALPHA_INTERPRET_THRESHOLD:.0e}: {n_above} / {len(team_df)}")
print(f"  Teams with α > 0.01:    {(team_df['alpha_M1'] > 0.01).sum()} / {len(team_df)}")
print(f"  Teams with α > 0.05:    {(team_df['alpha_M1'] > 0.05).sum()} / {len(team_df)}")

# ============================================================
# (5) Final model comparison + read simul_key_metrics (FIX [B4])
# ============================================================
print("\n" + "=" * 70)
print("(5) FINAL MODEL COMPARISON")
print("=" * 70)

results = pd.DataFrame({
    "Model":   ["M0: Hom. Poisson",   "M1: Self-only Hawkes",
                "M2: Hawkes+External", "M3: Pooled 48-bin Inhom. Poisson"],
    "k":       [k_M0,                  k_M1,                  k_M2,                  k_M3],
    "LogLik":  [ll_M0,                 ll_M1,                 ll_M2,                 ll_M3],
    "AIC":     [aic_M0,                aic_M1,                aic_M2,                aic_M3],
    "BIC":     [bic_M0,                bic_M1,                bic_M2,                bic_M3],
})
# Add ΔAIC, ΔBIC relative to the best (lowest) model
results["Delta_AIC"] = results["AIC"] - results["AIC"].min()
results["Delta_BIC"] = results["BIC"] - results["BIC"].min()
print(results.to_string(index=False, float_format='%.4f'))
results.to_csv("phase4_model_comparison.csv", index=False)
print("[Saved] phase4_model_comparison.csv")
print(f"\n  Best model by AIC: {results.loc[results['AIC'].idxmin(), 'Model']}")
print(f"  Best model by BIC: {results.loc[results['BIC'].idxmin(), 'Model']}")

# Boundary numerical artifact diagnostics
if ll_M1 < ll_M0:
    print(f"\n  [NOTE] LL(M1) is {ll_M1 - ll_M0:+.4f} relative to LL(M0).")
    print("         Theoretically LL(M1) >= LL(M0) since M1 nests M0 at α=0;")
    print("         a small negative value reflects EM not exactly reaching")
    print("         the α=0 boundary. Substantively M1 == M0 here.")
if ll_M2 < ll_M1:
    print(f"  [NOTE] LL(M2) is {ll_M2 - ll_M1:+.4f} relative to LL(M1).")
    print("         Theoretically LL(M2) >= LL(M1) since M2 nests M1 at α_ext=0;")
    print("         a small negative value reflects EM not exactly reaching")
    print("         the α_ext=0 boundary. Substantively M2 == M1 here.")

# Read simul_key_metrics if available; otherwise note absence (no hardcoding)
print("\n[Simultaneity stat — read from step_B output, no hardcoding]")
if os.path.exists(SIMUL_METRICS_FILE):
    sk = pd.read_csv(SIMUL_METRICS_FILE)
    sk_d = dict(zip(sk["metric"], sk["value"]))
    required_metrics = [
        "pct_same_second", "pct_within_1s_full", "observed_null_ratio_full",
    ]
    missing = [m for m in required_metrics if m not in sk_d]
    if missing:
        print(f"  [WARNING] Missing metrics in {SIMUL_METRICS_FILE}: {missing}")
        print(f"           Available metrics: {list(sk_d.keys())}")
        print(f"           Re-run 01_step_B_simultaneity_v3.py to regenerate.")
    else:
        pct_1s     = sk_d["pct_within_1s_full"]
        pct_same   = sk_d["pct_same_second"]
        ratio_full = sk_d["observed_null_ratio_full"]
        print(f"  pct_same_second        = {pct_same:.2f}%")
        print(f"  pct_within_1s_full     = {pct_1s:.2f}%")
        print(f"  observed_null_ratio    = {ratio_full:.1f}x  (full denom)")
else:
    print(f"  [INFO] {SIMUL_METRICS_FILE} not found.")
    print(f"        Run 01_step_B_simultaneity_v3.py first to generate it.")
    print(f"        This script does NOT hardcode the simultaneity stats.")

# ============================================================
# Diagnostic plot
# ============================================================
fig = plt.figure(figsize=(14, 11), constrained_layout=True)
gs = gridspec.GridSpec(3, 2, figure=fig)

# (1) M2 multi-start trajectories — show both α_self and α_ext, log-floor at 1e-8
ax = fig.add_subplot(gs[0, 0])
ALPHA_FLOOR_PLOT = 1e-8
for k_start, (mu, a_s, b_s, a_e, b_e, ll, hist) in enumerate(all_results):
    a_s_traj = np.maximum([h[2] for h in hist], ALPHA_FLOOR_PLOT)
    a_e_traj = np.maximum([h[4] for h in hist], ALPHA_FLOOR_PLOT)
    ax.plot(a_s_traj, alpha=0.7, linestyle="-",
            label=f"S{k_start + 1} α_self → {a_s:.2e}")
    ax.plot(a_e_traj, alpha=0.7, linestyle="--",
            label=f"S{k_start + 1} α_ext  → {a_e:.2e}")
ax.set_xlabel("EM iteration")
ax.set_ylabel(rf"$\alpha$ (floored at {ALPHA_FLOOR_PLOT:.0e} for log)")
ax.set_title("M2 multi-start: α trajectories")
ax.set_yscale('log')
ax.legend(fontsize=6, ncol=2)

# (2) Per-team α distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(team_df["alpha_M1"], bins=30, color="steelblue", edgecolor="black", alpha=0.75)
ax.axvline(ALPHA_INTERPRET_THRESHOLD, color="red", linestyle="--",
           label=f"interpret threshold = {ALPHA_INTERPRET_THRESHOLD:.0e}")
ax.set_xlabel(r"$\alpha$ (per-team M1)")
ax.set_ylabel("Number of teams")
ax.set_title(f"Distribution of per-team α (n={len(team_df)} teams)")
ax.legend(fontsize=8)

# (3) Per-season summary
ax = fig.add_subplot(gs[1, 0])
ax.bar(range(len(season_df)), season_df["alpha_M1"],
       color="darkgreen", edgecolor="black")
ax.set_xticks(range(len(season_df)))
ax.set_xticklabels(season_df["season"].astype(str), rotation=20)
ax.axhline(ALPHA_INTERPRET_THRESHOLD, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel(r"$\alpha$ (per-season M1)")
ax.set_title("Per-season M1 α estimates")

# (4) Model comparison bar
ax = fig.add_subplot(gs[1, 1])
mods = results["Model"].tolist()
aics = results["AIC"].tolist()
ax.barh(mods, aics, color="orange", edgecolor="black")
ax.set_xlabel("AIC (lower = better)")
ax.set_title("Model comparison (AIC)")
ax.invert_yaxis()
for i, v in enumerate(aics):
    ax.text(v, i, f"  {v:,.0f}", va="center", fontsize=8)

# (5) Top-team table
ax = fig.add_subplot(gs[2, 0])
ax.axis("off")
top_text = team_df.head(5)[["team", "alpha_M1", "mu_M1", "beta_M1"]].to_string(
    index=False, float_format='%.5f'
)
ax.text(0.0, 0.95, "TOP 5 TEAMS BY α", fontsize=10, fontweight="bold",
        transform=ax.transAxes, va="top", family="monospace")
ax.text(0.0, 0.85, top_text, fontsize=8, transform=ax.transAxes,
        va="top", family="monospace")

# (6) Bottom-team table
ax = fig.add_subplot(gs[2, 1])
ax.axis("off")
bot_text = team_df.tail(5)[["team", "alpha_M1", "mu_M1", "beta_M1"]].to_string(
    index=False, float_format='%.5f'
)
ax.text(0.0, 0.95, "BOTTOM 5 TEAMS BY α", fontsize=10, fontweight="bold",
        transform=ax.transAxes, va="top", family="monospace")
ax.text(0.0, 0.85, bot_text, fontsize=8, transform=ax.transAxes,
        va="top", family="monospace")

plt.savefig("phase4_full_league.png", dpi=120, bbox_inches="tight")
print("\n[Saved] phase4_full_league.png")

print("\n" + "=" * 70)
print("PHASE 4 COMPLETE")
print("=" * 70)
print(f"  M1 league: μ={mu_M1:.6f}, α={a_M1:.6f}, β={b_M1:.6f}")
print(f"  M2 league: α_self={a_s_M2:.6f}, α_ext={a_e_M2:.6f}")
print(f"  Per-season α: min={season_df['alpha_M1'].min():.6f}, max={season_df['alpha_M1'].max():.6f}")
print(f"  Per-team α:   min={team_df['alpha_M1'].min():.6f}, max={team_df['alpha_M1'].max():.6f}")
print(f"  Model AIC ranking: {' < '.join(results.sort_values('AIC')['Model'].tolist())}")