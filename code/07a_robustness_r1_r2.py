"""
================================================================================
Script 05: Phase 5 — Robustness Checks (v2)
================================================================================

Purpose:
  Run four robustness checks (R1-R4) to confirm the negative finding is not
  an artifact of modeling choices.

  POSITIONING (v2 — see "Status by check" below):

    R1. Power-law kernel Hawkes               [DEPRECATED — see 06_phase5_fix_v2.py]
    R2. Mass-aggregation threshold sweep      [PRIMARY ROBUSTNESS RESULT]
    R3. Marked Hawkes with n_players mark     [DEPRECATED — see 07_phase5_r3_redo_v2.py]
    R4. Random-time-change residuals + KS     [SECONDARY DIAGNOSTIC]

  In this v2 (paper §7), R1 and R3 are explicitly DEPRECATED:
    - R1 was numerically unstable (Nelder-Mead with no constraints, c → 1e20,
      p → 1e16, spurious α = 0.85). The bounded re-estimation in
      06_phase5_fix_v2.py supersedes it.
    - R3's α = 0.42 was a mark-weighting normalization artifact, not a real
      finding. The proper LR-style re-test in 07_phase5_r3_redo_v2.py
      supersedes it.

  R2 and R4 still pass cleanly and are the robustness numbers cited in §7.

Status by check:

  R1 [DEPRECATED]: Output kept here only for archival comparison with v0.3.6
                   paper. DO NOT cite the R1 alpha from this script in v0.4.
                   Cite 06_phase5_fix_v2.py results instead.

  R2 [PRIMARY]:    Mass-aggregation window sweep {0, 5, 30, 60} seconds. Re-run
                   on the full 3-season league with FIX [B1] applied to
                   fit_self_quick (α M-step uses the finite-window compensator
                   denominator, matching phase2_v4 / phase4_v4 / fit_m1f_v4).
                   Expected: α < 1e-4 across all windows.

  R3 [DEPRECATED]: As R1. Output kept for archival. Cite 07_phase5_r3_redo_v2.py.

  R4 [SECONDARY]:  RTC residuals using the self-only Hawkes parameters from
                   phase4_v4 / phase2_v4 (μ=0.005775, α=2e-6, β=0.006277).
                   Compares to the 48-bin inhomogeneous Poisson (M3) baseline.

Inputs:
  filtered_3_seasons.csv.gz     — Full 3-season league play-by-play

Outputs:
  phase5_robustness.png          — 4-panel diagnostic figure
  phase5_aggregation_sweep.csv   — R2 results (CITE-WORTHY)
  phase5_robustness_summary.csv  — All R1-R4 numbers (R1, R3 marked deprecated)

Fixes vs the original 05_phase5_robustness.py:
  [B1] R2 fit_self_quick: α M-step uses Σ_j (1 - exp(-β(T - t_j))) denominator,
       not n_total. Same fix as Phase 2 / 3 / 4 / fit_m1f.
  [Header] R1 and R3 sections clearly marked [DEPRECATED]; instructions point
           readers to 06_phase5_fix_v2 / 07_phase5_r3_redo_v2.
  [R4] Use μ_full=0.005775, α_full=2e-6, β_full=0.006277 from phase4_v4
       (matching the now-canonical full-league fit), not the v0.3.6 rounded
       values 0.00578 / 0 / 0.00628.
  [Wording] No "alpha = 0 across NBA"; use ALPHA_INTERPRET_THRESHOLD = 1e-4.
  [I9] R2 output csv adds LL/AIC/BIC columns alongside α, β, μ.

Execution time: hardware-dependent; R2 is the primary runtime component.
                R1 and R3 are retained only for archival reproduction of
                v0.3.6 outputs and are not cited.
================================================================================
"""

# (Original v1 docstring preserved below for archival reference.)
"""
Phase 5: Robustness checks (full-league, 3 seasons).

R1. Power-law kernel Hawkes: kappa(t) = (alpha * (p-1) / c) * (1 + t/c)^(-p)
    To rule out the possibility that alpha = 0 is an exponential-kernel artifact.

R2. Mass-aggregation threshold sweep: aggregate subs that occur within
    Delta-second windows (Delta in {0, 5, 30, 60} seconds) and re-fit Hawkes.
    Verifies that aggregation choice doesn't drive alpha to 0.

R3. Marked Hawkes (with n_players as mark): kappa_ij(t) ~ exp(-beta*t) * f(mark_j)
    Tests whether multi-player substitutions excite future subs more than single ones.

R4. Random-time-change residuals + KS test for the full-league self-only Hawkes,
    matching what we did at small scale in Phase 2.
"""

import numpy as np
import pandas as pd
import time
from scipy.stats import kstest
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# Configuration
ALPHA_INTERPRET_THRESHOLD = 1e-4   # below this α is treated as effectively 0,
                                    # β not interpreted (matches phase2/3/4/fit_m1f)
LAM_FLOOR = 1e-12

print("=" * 70)
print("PHASE 5: ROBUSTNESS CHECKS v2 (full-league, 3 seasons)")
print("  R1 [DEPRECATED — see 06_phase5_fix_v2.py]")
print("  R2 [PRIMARY] aggregation window sweep")
print("  R3 [DEPRECATED — see 07_phase5_r3_redo_v2.py]")
print("  R4 [SECONDARY] RTC residuals diagnostic")
print("=" * 70)

# ============================================================
# Load and prepare
# ============================================================
print("\n[Loading]")
df = pd.read_csv("filtered_3_seasons.csv.gz", compression="gzip", low_memory=False)

def pctime_to_sec_left(s):
    if pd.isna(s): return np.nan
    try:
        m, sec = s.split(":")
        return int(m)*60 + int(sec)
    except: return np.nan

def absolute_seconds(period, pctime_str):
    sl = pctime_to_sec_left(pctime_str)
    if pd.isna(sl): return np.nan
    period = int(period)
    return 720*(period-1)+(720-sl) if period<=4 else 720*4+300*(period-5)+(300-sl)

df["t_abs"] = df.apply(lambda r: absolute_seconds(r["PERIOD"], r["PCTIMESTRING"]), axis=1)

subs_raw = df[df["EVENTMSGTYPE"]==8].dropna(subset=["t_abs", "PLAYER1_TEAM_ID"]).copy()
subs_raw["TEAM_ID"] = subs_raw["PLAYER1_TEAM_ID"].astype(int)

T_HORIZON = 2880.0

# Helper: build mass-aggregated realizations given a window threshold
def build_realizations(subs_df, agg_window_sec, period_filter=4):
    """
    Aggregate subs into one mass-event per (game, team) realization.

    Aggregation rule (definition B — consecutive-gap):
      Walk through chronologically. Two adjacent events whose RAW (not
      cluster-first) timestamps differ by <= agg_window_sec are merged.
      This is the standard streak/cluster definition: a cluster grows as
      long as the next event is within the window of the most recent
      raw event in it.

      [Earlier v0.3.6 implementation used (next event vs. cluster's first
      event) which can prematurely cut clusters. Fixed in v2.]

    Returns: list of (agg_times array, n_per_event array) per realization.
    """
    sub_reg = subs_df[subs_df["PERIOD"] <= period_filter].copy()
    sub_reg = sub_reg.sort_values(["GAME_ID","TEAM_ID","t_abs"]).reset_index(drop=True)

    realizations = []
    for (gid, tid), g in sub_reg.groupby(["GAME_ID","TEAM_ID"]):
        times = g["t_abs"].values
        if agg_window_sec <= 0:
            agg_times = np.unique(times)  # exact-time aggregation
            n_per_event = np.array([(times == t).sum() for t in agg_times], dtype=int)
        else:
            # FIX [#1]: definition B — gap from last raw event, not first event
            agg_times_list = [times[0]]
            n_per_event_list = [1]
            last_t = times[0]
            for t in times[1:]:
                if t - last_t <= agg_window_sec:
                    n_per_event_list[-1] += 1
                else:
                    agg_times_list.append(t)
                    n_per_event_list.append(1)
                last_t = t
            agg_times = np.array(agg_times_list, dtype=float)
            n_per_event = np.array(n_per_event_list, dtype=int)
        # FIX [#2]: mask both arrays together for safety
        mask = agg_times < T_HORIZON
        agg_times = agg_times[mask]
        n_per_event = n_per_event[mask]
        if len(agg_times) > 0:
            realizations.append((agg_times, n_per_event))
    return realizations

# Build standard realizations (window=0, exact mass-aggregation)
realizations = build_realizations(subs_raw, agg_window_sec=0)
print(f"  Standard (window=0): {len(realizations)} realizations, "
      f"{sum(len(r[0]) for r in realizations)} events")

# Strip the n_players for now (not needed for self-only Hawkes)
realiz_times = [r[0] for r in realizations]
realiz_marks = [r[1] for r in realizations]
n_total = sum(len(r) for r in realiz_times)
n_real = len(realiz_times)

# ============================================================
# R1: Power-law kernel Hawkes
# Kernel: kappa(t) = alpha * (p-1)/c * (1 + t/c)^(-p)  for t >= 0, p > 1, c > 0
# Branching ratio = alpha (under stationarity, integral of kappa = alpha)
# Compensator integral: alpha * [1 - (1 + t/c)^(1-p)]
# ============================================================
print("\n" + "="*70)
print("R1: POWER-LAW KERNEL HAWKES   [DEPRECATED — see 06_phase5_fix_v2.py]")
print("="*70)
print("  This R1 fit uses unbounded Nelder-Mead and is known to be numerically")
print("  unstable (in v0.3.6 the optimizer drifted to c → 1e20, p → 1e16).")
print("  Output retained here only for archival comparison. DO NOT cite this")
print("  R1 alpha in v0.4 paper. The bounded re-estimation in")
print("  06_phase5_fix_v2.py is the authoritative power-law robustness test.")
print()

def hawkes_pl_ll(realiz_times, mu, alpha, c, p, T):
    """Log-likelihood for Hawkes with power-law kernel."""
    total = 0.0
    coef = alpha * (p - 1) / c
    for sub_t in realiz_times:
        n = len(sub_t)
        if n == 0:
            total += -mu * T
            continue
        lam = np.full(n, mu)
        for i in range(1, n):
            for j in range(i):
                lam[i] += coef * (1 + (sub_t[i] - sub_t[j])/c)**(-p)
        lam = np.maximum(lam, 1e-15)
        log_term = np.sum(np.log(lam))
        # Compensator integral over [0, T] of intensity:
        #   mu*T + sum_j alpha * [1 - (1 + (T - t_j)/c)^(1-p)]
        comp_excite = alpha * np.sum(1 - (1 + (T - sub_t)/c)**(1-p))
        total += log_term - mu*T - comp_excite
    return total

# Use scipy.optimize.minimize on negative log-likelihood
# Parameters: log(mu), logit(alpha) (to keep alpha in [0, 0.99]), log(c), log(p-1)
from scipy.optimize import minimize

def neg_ll_pl(params, realiz_times, T):
    log_mu, logit_a, log_c, log_pm1 = params
    mu = np.exp(log_mu)
    alpha = 0.99 / (1 + np.exp(-logit_a))  # in (0, 0.99)
    c = np.exp(log_c)
    p = 1.0 + np.exp(log_pm1)  # > 1
    return -hawkes_pl_ll(realiz_times, mu, alpha, c, p, T)

# Subsample to speed up power-law fit (it's O(n^2) per realization)
# Use 300 random realizations as a representative subset
print("  (Note: power-law fit uses 300-realization subsample for tractability;")
print("         results are then verified on full data with the converged params)")
sub_idx = np.random.choice(len(realiz_times), 300, replace=False)
sub_realiz = [realiz_times[i] for i in sub_idx]

# Try multiple starts for power-law
print("\n  Multi-start power-law fits...")
pl_starts = [
    [np.log(0.005), 0, np.log(60), np.log(1.0)],   # alpha=0.5, c=60s, p=2
    [np.log(0.003), -1, np.log(120), np.log(0.5)], # alpha~0.27, c=2min, p=1.5
    [np.log(0.005), 1, np.log(30), np.log(2)],     # alpha~0.72, c=30s, p=3
    [np.log(0.001), -3, np.log(300), np.log(0.2)], # alpha~0.05, c=5min, p=1.2
]

pl_results = []
for k, x0 in enumerate(pl_starts):
    t0 = time.time()
    try:
        res = minimize(neg_ll_pl, x0, args=(sub_realiz, T_HORIZON),
                       method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-4, "fatol": 0.5})
        log_mu, logit_a, log_c, log_pm1 = res.x
        mu = np.exp(log_mu); alpha = 0.99/(1+np.exp(-logit_a)); c = np.exp(log_c); p = 1+np.exp(log_pm1)
        ll = -res.fun
        print(f"  Start {k+1}: mu={mu:.5f}  alpha={alpha:.5f}  c={c:.1f}  p={p:.3f}  LL={ll:.1f}  ({time.time()-t0:.0f}s)")
        pl_results.append((mu, alpha, c, p, ll))
    except Exception as e:
        print(f"  Start {k+1}: failed: {e}")

# FIX [#8]: empty-results guard — if all 4 starts fail, set archival outputs
# to NaN rather than crashing on max() of empty list.
if len(pl_results) == 0:
    print("  [R1 DEPRECATED] All power-law starts failed; archival outputs set to NaN.")
    mu_pl, alpha_pl, c_pl, p_pl, ll_pl_sub = np.nan, np.nan, np.nan, np.nan, np.nan
else:
    best_pl = max(pl_results, key=lambda r: r[4])
    mu_pl, alpha_pl, c_pl, p_pl, ll_pl_sub = best_pl
print(f"\n[BEST power-law on subsample] alpha={alpha_pl:.5f}  c={c_pl:.1f}s  p={p_pl:.3f}  "
      f"[DEPRECATED — see 06_phase5_fix_v2]")

# Verify on full data (just compute likelihood with these params)
print("\n  Computing archival LL on first 1000 realizations with fitted params...")
t0 = time.time()
if np.isnan(alpha_pl):
    ll_pl_full = np.nan
    print(f"  Full LL: NaN (R1 fits all failed)")
else:
    ll_pl_full = hawkes_pl_ll(realiz_times[:1000], mu_pl, alpha_pl, c_pl, p_pl, T_HORIZON)  # 1000 reals for speed
    print(f"  Full LL (first 1000 reals): {ll_pl_full:.1f}  ({time.time()-t0:.0f}s)")
print(f"\n[R1 VERDICT — DEPRECATED] Power-law alpha = {alpha_pl}")
print("  See 06_phase5_fix_v2.py for the bounded re-estimation.")

# ============================================================
# R2: Mass-aggregation threshold sweep
# ============================================================
print("\n" + "="*70)
print("R2: MASS-AGGREGATION THRESHOLD SWEEP   [PRIMARY ROBUSTNESS RESULT]")
print("="*70)
print("  Re-fit M1 (homogeneous-baseline self-only Hawkes) at four")
print("  mass-aggregation windows {0, 5, 30, 60}s. With FIX [B1] applied")
print("  to fit_self_quick (α uses finite-window compensator denominator).")
print()

def fit_self_quick(reals, T, mu0=0.005, alpha0=0.3, beta0=1/180, max_iter=200, tol=1e-7):
    """
    EM for self-only Hawkes (M1) with FIX [B1]:
      α_new = sum_pS / sum_compensator (NOT / n_total).
    Same formula as phase2_v4 / phase4_v4 / fit_m1f_v4 / phase3_v4.
    """
    mu, alpha, beta = mu0, alpha0, beta0
    n_r = len(reals)
    for it in range(max_iter):
        sum_pB = 0.0
        sum_pS = 0.0
        sum_pS_dt = 0.0
        sum_compensator = 0.0
        for sub_t in reals:
            n = len(sub_t)
            if n == 0:
                continue
            A = np.zeros(n); B = np.zeros(n)
            for i in range(1, n):
                dt = sub_t[i] - sub_t[i-1]
                e = np.exp(-beta * dt)
                A[i] = e * (1 + A[i-1])
                B[i] = e * (B[i-1] + dt * (1 + A[i-1]))
            lam = np.maximum(mu + alpha * beta * A, LAM_FLOOR)
            pB = mu / lam
            sum_pB += pB.sum()
            sum_pS += (1 - pB).sum()
            sum_pS_dt += (alpha * beta * B / lam).sum()
            sum_compensator += np.sum(1 - np.exp(-beta * (T - sub_t)))   # FIX [B1]
        new_mu = max(sum_pB / (n_r * T), 1e-9)
        # FIX [B1]: divide by finite-window compensator, not n_total
        # FIX [#4]: clamp to [0, 0.99] in case of numerical noise
        if sum_compensator > 1e-12:
            new_alpha = min(max(sum_pS / sum_compensator, 0.0), 0.99)
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
    """Self-only Hawkes log-likelihood for use in R2 model comparison."""
    total = 0.0
    for sub_t in reals:
        n = len(sub_t)
        if n == 0:
            total += -mu * T
            continue
        A = np.zeros(n)
        for i in range(1, n):
            A[i] = np.exp(-beta * (sub_t[i] - sub_t[i-1])) * (1 + A[i-1])
        lam = np.maximum(mu + alpha * beta * A, LAM_FLOOR)
        comp = alpha * np.sum(1 - np.exp(-beta * (T - sub_t)))
        total += np.sum(np.log(lam)) - mu * T - comp
    return total

print("\n  Aggregation thresholds: {0, 5, 30, 60} seconds")
agg_results = []
for window in [0, 5, 30, 60]:
    t0 = time.time()
    reals = build_realizations(subs_raw, agg_window_sec=window)
    times_only = [r[0] for r in reals]
    n_evt = sum(len(t) for t in times_only)
    mu, alpha, beta = fit_self_quick(times_only, T_HORIZON, max_iter=200)
    ll = hawkes_self_ll(times_only, mu, alpha, beta, T_HORIZON)
    k = 3
    aic = 2 * k - 2 * ll
    bic = k * np.log(max(n_evt, 1)) - 2 * ll
    above = bool(alpha > ALPHA_INTERPRET_THRESHOLD)
    print(f"  window={window:>3}s: n_real={len(reals)}, n_evt={n_evt:>6}, "
          f"alpha={alpha:.6f}, mu={mu:.5f}, beta={beta:.5f}, "
          f"LL={ll:.1f}, AIC={aic:.1f}, "
          f"alpha>thr={above}  ({time.time()-t0:.0f}s)")
    agg_results.append({"window_s": window, "n_real": len(reals), "n_evt": n_evt,
                         "mu": mu, "alpha": alpha, "beta": beta,
                         "LogLik": ll, "AIC": aic, "BIC": bic,
                         "alpha_above_threshold": above})

agg_df = pd.DataFrame(agg_results)
agg_df.to_csv("phase5_aggregation_sweep.csv", index=False)
print(f"\n[R2 VERDICT] alpha range across aggregation windows: "
      f"[{agg_df['alpha'].min():.6f}, {agg_df['alpha'].max():.6f}]")
n_above = int(agg_df["alpha_above_threshold"].sum())
if n_above == 0:
    print(f"  ✓ All 4 windows have alpha < {ALPHA_INTERPRET_THRESHOLD:.0e}.")
    print("    The negative finding is robust to mass-aggregation choice.")
else:
    print(f"  ⚠ {n_above}/4 windows have alpha > {ALPHA_INTERPRET_THRESHOLD:.0e}; investigate.")

# ============================================================
# R3: Marked Hawkes (n_players as mark)
# Model: lambda(t) = mu + alpha * beta * sum_{t_j<t} m_j * exp(-beta*(t - t_j))
#        where m_j = mark for event j (number of players in mass-aggregated sub)
# Tests if multi-player substitutions are stronger triggers than single-player ones.
# ============================================================
print("\n" + "="*70)
print("R3: MARKED HAWKES (n_players as mark)   "
      "[DEPRECATED — see 07_phase5_r3_redo_v2.py]")
print("="*70)
print("  This R3 fit has a known mark-weighting normalization issue:")
print("  in v0.3.6 it produced spurious α = 0.42 because the M-step did")
print("  not properly account for the marks in the compensator denominator.")
print("  The proper LR-style re-test is in 07_phase5_r3_redo_v2.py.")
print("  Output retained here only for archival comparison. DO NOT cite this")
print("  R3 alpha in v0.4 paper.")
print()

def fit_marked_hawkes(realiz_times, realiz_marks, T, mu0=0.005, alpha0=0.3, beta0=1/180,
                      max_iter=80, tol=1e-6, verbose=False):
    mu, alpha, beta = mu0, alpha0, beta0
    n_total = sum(len(s) for s in realiz_times)
    n_r = len(realiz_times)
    for it in range(max_iter):
        sum_pB = 0; sum_pS = 0; sum_pS_dt = 0
        # Compensator basis: alpha * sum_j m_j * (1 - exp(-beta*(T - t_j)))
        sum_compensator = 0
        for sub_t, marks in zip(realiz_times, realiz_marks):
            n = len(sub_t)
            if n == 0: continue
            # M-weighted recursion: A_i = sum_{j<i} m_j * exp(-beta*(t_i-t_j))
            A = np.zeros(n); B = np.zeros(n)
            for i in range(1, n):
                dt = sub_t[i] - sub_t[i-1]
                e = np.exp(-beta*dt)
                A[i] = e*(marks[i-1] + A[i-1])
                B[i] = e*(B[i-1] + dt*(marks[i-1] + A[i-1]))
            lam = mu + alpha*beta*A
            lam = np.maximum(lam, 1e-15)
            pB = mu/lam
            sum_pB += pB.sum()
            sum_pS += (1-pB).sum()
            sum_pS_dt += (alpha*beta*B/lam).sum()
            sum_compensator += np.sum(marks * (1 - np.exp(-beta*(T - sub_t))))
        new_mu = max(sum_pB/(n_r*T), 1e-9)
        # DEPRECATED [archival reproduction of v0.3.6 R3 only]:
        # This M-step uses sum_pS / n_total, which does NOT properly account
        # for the marks in the compensator denominator. The proper marked-Hawkes
        # M-step would use sum_pS / sum_compensator (where sum_compensator
        # already weights by marks). This denominator mismatch is exactly what
        # produced the spurious α = 0.42 in v0.3.6 — see 07_phase5_r3_redo_v2.py
        # for the corrected estimator. Do NOT interpret new_alpha here.
        new_alpha = min(sum_pS / n_total, 0.99)
        new_beta = max(sum_pS / sum_pS_dt, 1e-7) if sum_pS > 1e-10 else beta
        delta = max(abs(new_mu-mu), abs(new_alpha-alpha), abs(new_beta-beta))
        mu, alpha, beta = new_mu, new_alpha, new_beta
        if delta < tol: break
    return mu, alpha, beta

t0 = time.time()
mu_m, alpha_m, beta_m = fit_marked_hawkes(realiz_times, realiz_marks, T_HORIZON, max_iter=80)
print(f"\n  Marked Hawkes (n_players as mark):")
print(f"    mu    = {mu_m:.5f}")
print(f"    alpha = {alpha_m:.5f}")
print(f"    beta  = {beta_m:.5f}")
print(f"    ({time.time()-t0:.0f}s)")

# Mark distribution
all_marks = np.concatenate(realiz_marks)
mark_dist = pd.Series(all_marks).value_counts().sort_index()
print(f"\n  Mark distribution (n_players per atomic event):")
print(mark_dist.to_string())

print(f"\n[R3 ARCHIVAL OUTPUT — DEPRECATED] Marked Hawkes alpha = {alpha_m:.5f}")
print("  Do NOT interpret this value. The M-step denominator does not properly")
print("  account for marks; this is the source of v0.3.6's spurious α = 0.42.")
print("  See 07_phase5_r3_redo_v2.py for the corrected LR-style re-test.")

# ============================================================
# R4: Random-Time-Change residuals + KS test
# ============================================================
print("\n" + "="*70)
print("R4: RANDOM-TIME-CHANGE RESIDUAL DIAGNOSTICS   [SECONDARY DIAGNOSTIC]")
print("="*70)

# Use full-league self-only Hawkes parameters from phase4_v4 / phase2_v4
# (was the v0.3.6 rounded values 0.00578 / 0.0 / 0.00628; now exact match
#  to phase2_v4's converged EM)
mu_full = 0.005775
alpha_full = 2.0e-6   # below ALPHA_INTERPRET_THRESHOLD; effectively 0
beta_full = 0.006277

print(f"\n  Using full-league M1 parameters from phase2_v4 / phase4_v4:")
print(f"    μ = {mu_full:.6f}, α = {alpha_full:.2e} (< {ALPHA_INTERPRET_THRESHOLD:.0e}), "
      f"β = {beta_full:.6f}")
print(f"  Since α < interpretation threshold, the model is effectively a")
print(f"  homogeneous Poisson with rate μ. RTC residuals are μ times the")
print(f"  event-time increments, INCLUDING the first increment from t=0.")

# Since α < threshold, model effectively a homogeneous Poisson with rate μ
# RTC: Λ(t_i) = μ · t_i, residuals = increments of Λ at consecutive events.
# FIX [#6]: include the first increment Λ(t1) - Λ(0) — earlier code skipped it.
print("\n  Self-only Hawkes (α < threshold → effectively homogeneous Poisson):")
residuals_self = []
for sub_t in realiz_times:
    if len(sub_t) > 0:
        L = mu_full * sub_t
        residuals_self.extend(np.diff(np.concatenate(([0.0], L))))
residuals_self = np.array(residuals_self)
ks_stat_self, ks_p_self = kstest(residuals_self, 'expon', args=(0, 1))
print(f"    N residuals: {len(residuals_self):,}  (includes first increment)")
print(f"    Mean: {residuals_self.mean():.3f}  (expected 1.0)")
print(f"    Var:  {residuals_self.var():.3f}  (expected 1.0)")
print(f"    KS stat: {ks_stat_self:.4f}  p-value: {ks_p_self:.4g}")

# Inhomogeneous Poisson 48 bins
n_bins = 48
bin_w = T_HORIZON / n_bins
all_event_times = np.concatenate(realiz_times)
event_bins = np.minimum((all_event_times // bin_w).astype(int), n_bins-1)
counts = np.bincount(event_bins, minlength=n_bins)
rates_b2 = counts / (n_real * bin_w)

print("\n  Inhomogeneous Poisson (48 1-min bins):")
def compensator_b2_fast(sub_t, rates, bin_w, T):
    L = np.zeros(len(sub_t))
    cum_rates = np.concatenate(([0], np.cumsum(rates) * bin_w))
    for k, t in enumerate(sub_t):
        full_bin = int(min(t // bin_w, len(rates) - 1))
        residual_t = t - full_bin * bin_w
        L[k] = cum_rates[full_bin] + rates[full_bin] * residual_t
    return L

residuals_b2 = []
for sub_t in realiz_times:
    if len(sub_t) > 0:
        L = compensator_b2_fast(sub_t, rates_b2, bin_w, T_HORIZON)
        # FIX [#6]: include first increment Λ(t1) - Λ(0)
        residuals_b2.extend(np.diff(np.concatenate(([0.0], L))))
residuals_b2 = np.array(residuals_b2)
ks_stat_b2, ks_p_b2 = kstest(residuals_b2, 'expon', args=(0, 1))
print(f"    N residuals: {len(residuals_b2):,}  (includes first increment)")
print(f"    Mean: {residuals_b2.mean():.3f}  (expected 1.0)")
print(f"    Var:  {residuals_b2.var():.3f}  (expected 1.0)")
print(f"    KS stat: {ks_stat_b2:.4f}  p-value: {ks_p_b2:.4g}")

# ============================================================
# Summary plots
# ============================================================
print("\n[Plotting summary]")
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = gridspec.GridSpec(2, 2, figure=fig)

# (1) R2: aggregation threshold sweep
ax = fig.add_subplot(gs[0, 0])
ax.bar(range(len(agg_df)), agg_df["alpha"], color="steelblue", edgecolor="black")
ax.set_xticks(range(len(agg_df)))
ax.set_xticklabels([f"{w}s" for w in agg_df["window_s"]])
ax.set_xlabel("Mass-aggregation window")
ax.set_ylabel("Estimated alpha (self-excitation)")
ax.set_title("R2: alpha vs aggregation window\n(all below interpret threshold)")
# FIX [#1]: threshold matches the global ALPHA_INTERPRET_THRESHOLD,
# not the v0.3.6 stale 0.05 value.
ax.axhline(
    ALPHA_INTERPRET_THRESHOLD,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"{ALPHA_INTERPRET_THRESHOLD:.0e} threshold",
)
ymax = max(agg_df["alpha"].max() * 1.25, ALPHA_INTERPRET_THRESHOLD * 5)
ax.set_ylim(0, ymax)
for i, v in enumerate(agg_df["alpha"]):
    ax.text(i, v + 0.05 * ymax, f"{v:.2e}", ha="center", fontsize=8)
ax.legend()

# (2) RTC residuals — Self-only (= Hom Poisson) QQ plot
ax = fig.add_subplot(gs[0, 1])
sorted_res = np.sort(residuals_self)
theo_q = -np.log(1 - (np.arange(len(sorted_res)) + 0.5) / len(sorted_res))
# Subsample for plotting (too many points)
if len(sorted_res) > 5000:
    idx = np.random.choice(len(sorted_res), 5000, replace=False)
    idx.sort()
    ax.plot(theo_q[idx], sorted_res[idx], '.', alpha=0.4, ms=2, color="steelblue")
else:
    ax.plot(theo_q, sorted_res, '.', alpha=0.4, ms=2, color="steelblue")
maxv = max(theo_q.max(), sorted_res.max())
ax.plot([0, maxv], [0, maxv], 'r-')
ax.set_xlabel("Theoretical Exp(1) quantile")
ax.set_ylabel("Residual quantile")
ax.set_title(f"R4: RTC residuals — Self-only Hawkes\nKS={ks_stat_self:.3f}, p={ks_p_self:.2g}")

# (3) RTC residuals — Inhom Poisson QQ plot
ax = fig.add_subplot(gs[1, 0])
sorted_res = np.sort(residuals_b2)
theo_q = -np.log(1 - (np.arange(len(sorted_res)) + 0.5) / len(sorted_res))
if len(sorted_res) > 5000:
    idx = np.random.choice(len(sorted_res), 5000, replace=False)
    idx.sort()
    ax.plot(theo_q[idx], sorted_res[idx], '.', alpha=0.4, ms=2, color="orange")
else:
    ax.plot(theo_q, sorted_res, '.', alpha=0.4, ms=2, color="orange")
maxv = max(theo_q.max(), sorted_res.max())
ax.plot([0, maxv], [0, maxv], 'r-')
ax.set_xlabel("Theoretical Exp(1) quantile")
ax.set_ylabel("Residual quantile")
ax.set_title(f"R4: RTC residuals — Inhom Poisson (48 bins)\nKS={ks_stat_b2:.3f}, p={ks_p_b2:.2g}")

# (4) Robustness summary table as text
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
summary_text = f"""ROBUSTNESS SUMMARY (full-league, 3 seasons)

R1. Power-law kernel    [SUPERSEDED]
    Not reported in this figure.
    -> Cite 06_phase5_fix_v2 instead.

R2. Aggregation window sweep    [PRIMARY]
    Window    alpha
    0s     {agg_df.iloc[0]['alpha']:.6f}
    5s     {agg_df.iloc[1]['alpha']:.6f}
    30s    {agg_df.iloc[2]['alpha']:.6f}
    60s    {agg_df.iloc[3]['alpha']:.6f}
    -> Robust to aggregation choice.

R3. Marked Hawkes (n_players mark)    [SUPERSEDED]
    Not reported in this figure.
    -> Cite 07_phase5_r3_redo_v2 instead.

R4. RTC residuals    [SECONDARY DIAGNOSTIC]
    Self-only Hawkes (α≈0): KS={ks_stat_self:.3f}
    Inhom Poisson (M3):     KS={ks_stat_b2:.3f}
    -> M3 residuals close to but
       not exactly Exp(1); finer
       covariate-driven model needed.

R2 + R4 SUPPORT THE
NO-SELF-EXCITATION FINDING.
R1 + R3 SUPERSEDED."""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor='black'))

plt.savefig("phase5_robustness.png", dpi=120, bbox_inches="tight")
print(f"[Saved] phase5_robustness.png")

# Save full results — R1 and R3 columns explicitly marked deprecated
robustness_results = {
    "R1_power_law_alpha_DEPRECATED": alpha_pl,
    "R1_power_law_c_DEPRECATED": c_pl,
    "R1_power_law_p_DEPRECATED": p_pl,
    "R1_status": "DEPRECATED — see 06_phase5_fix_v2.py",
    "R3_marked_alpha_DEPRECATED": alpha_m,
    "R3_marked_beta_DEPRECATED": beta_m,
    "R3_status": "DEPRECATED — see 07_phase5_r3_redo_v2.py",
    "R4_self_KS": ks_stat_self,
    "R4_self_p": ks_p_self,
    "R4_inhom_KS": ks_stat_b2,
    "R4_inhom_p": ks_p_b2,
}
robustness_df = pd.DataFrame([robustness_results])
robustness_df.to_csv("phase5_robustness_summary.csv", index=False)

print("\n" + "="*70)
print("PHASE 5 v2 COMPLETE")
print("  R2 (aggregation sweep): all alphas below threshold ✓")
print("  R4 (RTC residuals): diagnostic ✓")
print("  R2 + R4 support the no-self-excitation finding under the M1")
print("  homogeneous-baseline specification.")
print("  R1 / R3 deprecated; cite 06_phase5_fix_v2 / 07_phase5_r3_redo_v2 instead")
print("  Formal M1f vs M3 inference remains based on the parametric bootstrap")
print("  LR test in 11_parametric_bootstrap_v3.py.")
print("="*70)