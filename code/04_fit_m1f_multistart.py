"""
================================================================================
M1f: Fit Period-Minute Baseline + Hawkes Self-Excitation (PAPER §6 INFERENCE)
================================================================================

Purpose:
  Fit the model M1f:
      lambda(t) = mu_b[bin(t)] + alpha * beta * Σ_{T_i < t} exp(-beta(t - T_i))
  where mu_b is a piecewise-constant 48-bin baseline (1-minute resolution
  within 4 regulation periods).

  This is the paper's PRIMARY test of self-excitation given a flexible
  time-varying baseline, comparing M1f (k=50) against M3 (k=48).

  POSITIONING:
    - Phase 4 (homogeneous baseline) is a diagnostic. THIS is the inference.
    - Inferential statement: alpha and the LR-test against M3 are computed
      under a baseline that already absorbs clock-time rhythm, so any
      remaining excitation (if alpha > 0) is genuine event-to-event
      memory beyond rhythm.

Multi-start: 7 initializations matching paper Methods §5.2 (S1-S7) PLUS
             a M3-anchor candidate (S0) that sets alpha = 0 exactly. This
             guarantees LL_M1f >= LL_M3 (since M1f at alpha = 0 reduces to
             M3 exactly), eliminating the boundary numerical artifact in
             which all 7 EM inits with alpha_0 > 0 land slightly below M3
             due to monotonic decay never reaching the boundary.

Inputs:
  filtered_3_seasons.csv.gz  (full 3-season league play-by-play)

Outputs:
  m1f_results.csv          — fit summary across all inits (alpha, beta, LL)
  m1f_baseline.csv         — 48 fitted M3 bin rates (used as warm start)
  m1f_best_baseline.csv    — best-fit M1f baseline + alpha + beta + LL
                             (CONSUMED BY 10_cluster_robust_ses.py)
  m1f_log.txt              — full convergence log

Fixes vs the original 08_fit_m1f.py:
  [B5] CRITICAL: Add M3-anchor (S0) as 8th candidate. Without it, all 7
       EM inits decay toward alpha = 0 monotonically but never reach it,
       producing LL_M1f very slightly below LL_M3 (LR_raw = -0.005). The
       M3-anchor guarantees the best M1f LL is at least LL_M3, so LR_raw
       >= 0 and the test is statistically meaningful.
  [B1] alpha M-step uses finite-window compensator denominator
       Σ_j (1 - exp(-beta(T - t_j))), not n_total_sub. Same fix as
       Phase 2 / Phase 3 / Phase 4.
  [I9] Save m1f_best_baseline.csv with the 48 baseline rates plus alpha,
       beta, and LL. Consumed downstream by 10_cluster_robust_ses.py.
  [Wording] S4 "moderate fast" → "moderate slow" (beta=0.001 is slow,
            half-life ≈ 690s; S5 with beta=0.01 is the fast one).
  [I10] When best alpha is below ALPHA_INTERPRET_THRESHOLD, log a warning
        that beta is weakly identified (β unidentified at α=0 boundary).
  [I11] Boundary p-value caveat: under H0 alpha=0, beta is not identified;
        the standard chi-bar-square mixture for boundary tests is also
        not the asymptotically correct reference here. Both p-values are
        reported with explicit caveats.

EM caveat:
  beta is updated using the standard expected-waiting-time ratio rather
  than the strict finite-window MLE derivative. When alpha is below
  ALPHA_INTERPRET_THRESHOLD the practical impact of this approximation
  is expected to be limited; beta is weakly identified at the boundary
  in any case.

Execution time: ~10-25 minutes on a laptop (8 inits × 150 iters).
================================================================================
"""

import numpy as np
import pandas as pd
import time
import sys, os
from scipy.stats import chi2

np.random.seed(42)

# ---------- Configuration ----------
T_HORIZON = 2880.0
N_BINS = 48
BIN_W = T_HORIZON / N_BINS  # 60 seconds
LAM_FLOOR = 1e-12
ALPHA_INTERPRET_THRESHOLD = 1e-4

# --- Multi-start initializations: 7 EM inits + 1 M3-anchor (FIX [B5]) ---
# Note on β / decay terminology:
#   β = 0.001  → half-life ≈ 693 s (slow)
#   β = 0.01   → half-life ≈  69 s (medium)
#   β = 0.05   → half-life ≈  14 s (fast)
INITS = [
    # FIX [B5]: M3-anchor — alpha = 0 exactly, no EM. Guarantees LL_M1f >= LL_M3.
    {"name": "S0", "alpha": 0.0,    "beta": 0.01,  "desc": "M3-anchor (alpha=0, no EM)",
     "is_m3_anchor": True},
    # Standard 7 EM inits (S1-S7). FIX wording: medium-decay vs slow distinction.
    {"name": "S1", "alpha": 0.0001, "beta": 0.01,  "desc": "Poisson-like"},
    {"name": "S2", "alpha": 0.001,  "beta": 0.01,  "desc": "small medium-decay"},
    {"name": "S3", "alpha": 0.01,   "beta": 0.01,  "desc": "moderate medium-decay"},
    {"name": "S4", "alpha": 0.10,   "beta": 0.001, "desc": "moderate slow"},
    {"name": "S5", "alpha": 0.05,   "beta": 0.01,  "desc": "substantial medium-decay"},
    {"name": "S6", "alpha": 0.95,   "beta": 0.01,  "desc": "near-critical medium-decay"},
    {"name": "S7", "alpha": 0.50,   "beta": 0.05,  "desc": "high-amp fast"},
]

# ---------- Load and prepare data ----------
print("=" * 70)
print("M1f: 48-BIN BASELINE + HAWKES SELF-EXCITATION (PAPER §6 INFERENCE)")
print("=" * 70)
print("\n[Loading data]")
DATA_FILE = "filtered_3_seasons.csv.gz"
if not os.path.exists(DATA_FILE):
    sys.exit(f"[FATAL] {DATA_FILE} not found in cwd.")

t_load = time.time()
df = pd.read_csv(DATA_FILE, low_memory=False)
n_games_loaded = df['GAME_ID'].nunique()
print(f"  Loaded {len(df):,} rows, {n_games_loaded:,} games in {time.time()-t_load:.1f}s")
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

print("[Computing absolute times]")
df["t_abs"] = df.apply(lambda r: absolute_seconds(r["PERIOD"], r["PCTIMESTRING"]), axis=1)
subs = df[df["EVENTMSGTYPE"] == 8].dropna(subset=["t_abs", "PLAYER1_TEAM_ID"]).copy()
subs["TEAM_ID"] = subs["PLAYER1_TEAM_ID"].astype(int)
n_missing = subs["PLAYER1_TEAM_ABBREVIATION"].isna().sum()
if n_missing > 0:
    print(f"  [INFO] {n_missing:,} sub rows missing TEAM_ABBREVIATION; using TEAM_ID fallback.")
subs["TEAM"] = subs["PLAYER1_TEAM_ABBREVIATION"].fillna(subs["TEAM_ID"].astype(str))

print("[Mass-aggregating substitutions]")
mass_subs = (
    subs.groupby(["GAME_ID", "TEAM_ID", "TEAM", "SEASON", "PERIOD", "t_abs"], as_index=False)
        .agg(n_players=("PLAYER1_ID", "count"))
)
mass_subs = mass_subs.sort_values(["GAME_ID", "TEAM_ID", "t_abs"]).reset_index(drop=True)
mass_reg = mass_subs[mass_subs["PERIOD"] <= 4].copy()

# Drop boundary events
n_at_or_after_T = (mass_reg["t_abs"] >= T_HORIZON).sum()
if n_at_or_after_T > 0:
    print(f"  [INFO] Dropping {n_at_or_after_T:,} regulation events at t_abs >= {T_HORIZON}.")

print("[Building (game, team) realizations]")
realizations = []
for (gid, tid), g_subs in mass_reg.groupby(["GAME_ID", "TEAM_ID"]):
    sub_t = np.sort(g_subs["t_abs"].values.astype(float))
    sub_t = sub_t[sub_t < T_HORIZON]
    if len(sub_t) > 0:
        realizations.append(sub_t)

n_total_sub = sum(len(s) for s in realizations)
n_real = len(realizations)
print(f"  {n_real:,} realizations, {n_total_sub:,} events")

# ---------- Pre-compute bin assignments and exposures ----------
print("\n[Pre-computing bin assignments]")
event_bins_per_real = []
for sub_t in realizations:
    bins = np.minimum((sub_t // BIN_W).astype(int), N_BINS - 1)
    event_bins_per_real.append(bins)

total_exposure = np.full(N_BINS, n_real * BIN_W)
all_event_bins = np.concatenate(event_bins_per_real)
total_events_per_bin = np.bincount(all_event_bins, minlength=N_BINS)

# Pre-compute self-Hawkes structures
print("[Pre-computing self-Hawkes dt cache]")
self_dt_cache = []
T_minus_sub_cache = []
for sub_t in realizations:
    n = len(sub_t)
    dt_self = np.zeros(n)
    if n > 1:
        dt_self[1:] = np.diff(sub_t)
    self_dt_cache.append(dt_self)
    T_minus_sub_cache.append(T_HORIZON - sub_t)

# ---------- M3: closed-form Inhomogeneous Poisson ----------
print("\n[M3: Inhomogeneous Poisson — closed form]")
mu_b_M3 = total_events_per_bin / total_exposure
ll_M3 = (np.log(np.maximum(mu_b_M3[all_event_bins], LAM_FLOOR))).sum() - (mu_b_M3 * total_exposure).sum()
print(f"  M3 LogLik = {ll_M3:.4f}")
print(f"  k = {N_BINS}")
print(f"  Min/Max bin rates: {mu_b_M3.min():.6f} / {mu_b_M3.max():.6f}")

# ============================================================
# fit_m1f_one_iter — EM step with FIX [B1]
# ============================================================
def fit_m1f_one_iter(mu_b, alpha, beta):
    """
    One EM iteration for M1f.

    FIX [B1]: alpha M-step uses Σ_j (1 - exp(-beta(T - t_j))), not n_total_sub.
    """
    sum_pB_per_bin = np.zeros(N_BINS)
    sum_pS = 0.0
    sum_pS_dt = 0.0
    sum_compensator = 0.0   # FIX [B1]: denominator for alpha update

    for r_idx, sub_t in enumerate(realizations):
        n = len(sub_t)
        if n == 0:
            continue
        bins = event_bins_per_real[r_idx]
        dt_self = self_dt_cache[r_idx]

        A = np.zeros(n)
        B = np.zeros(n)
        for i in range(1, n):
            ev = np.exp(-beta * dt_self[i])
            A[i] = ev * (1 + A[i - 1])
            B[i] = ev * (B[i - 1] + dt_self[i] * (1 + A[i - 1]))

        lam = np.maximum(mu_b[bins] + alpha * beta * A, LAM_FLOOR)
        pB = mu_b[bins] / lam
        np.add.at(sum_pB_per_bin, bins, pB)
        pS_term = (alpha * beta * A) / lam
        sum_pS += pS_term.sum()
        sum_pS_dt += (alpha * beta * B / lam).sum()

        # Finite-window compensator contribution
        sum_compensator += np.sum(1 - np.exp(-beta * T_minus_sub_cache[r_idx]))

    # M-step
    new_mu_b = np.maximum(sum_pB_per_bin / total_exposure, 1e-9)
    if sum_compensator > 1e-12:
        new_alpha = min(sum_pS / sum_compensator, 0.99)
    else:
        new_alpha = 0.0
    if sum_pS > 1e-10 and sum_pS_dt > 1e-12:
        new_beta = max(sum_pS / sum_pS_dt, 1e-7)
    else:
        new_beta = beta
    return new_mu_b, new_alpha, new_beta


def m1f_loglik(mu_b, alpha, beta):
    """Compute log-likelihood for M1f."""
    total = 0.0
    for r_idx, sub_t in enumerate(realizations):
        n = len(sub_t)
        if n == 0:
            continue
        bins = event_bins_per_real[r_idx]
        dt_self = self_dt_cache[r_idx]

        A = np.zeros(n)
        for i in range(1, n):
            A[i] = np.exp(-beta * dt_self[i]) * (1 + A[i - 1])

        lam = np.maximum(mu_b[bins] + alpha * beta * A, LAM_FLOOR)
        comp_baseline = (mu_b * BIN_W).sum()
        comp_self = alpha * np.sum(1 - np.exp(-beta * (T_HORIZON - sub_t)))
        total += np.sum(np.log(lam)) - comp_baseline - comp_self
    return total

# ============================================================
# Multi-start: 7 EM inits + 1 M3-anchor (FIX [B5])
# ============================================================
print("\n" + "=" * 70)
print("M1f MULTI-START: 7 EM inits + 1 M3-anchor (8 total)")
print("=" * 70)

m1f_results = []
log_lines = []
best_state = None  # (init_name, mu_b, alpha, beta, LL) for the highest-LL fit

for init in INITS:
    name = init["name"]
    alpha0 = init["alpha"]
    beta0 = init["beta"]
    desc = init["desc"]
    is_anchor = init.get("is_m3_anchor", False)

    print(f"\n[{name}: {desc}] alpha0={alpha0}, beta0={beta0}")
    log_lines.append(f"\n[{name}: {desc}] alpha0={alpha0}, beta0={beta0}")

    if is_anchor:
        # FIX [B5]: M3-anchor — alpha = 0 exactly, mu_b = mu_b_M3, no EM.
        # By construction LL_M3-anchor == LL_M3 (alpha=0 reduces M1f to M3).
        mu_b = mu_b_M3.copy()
        alpha = 0.0
        beta = beta0   # arbitrary (not identified at alpha=0)
        final_ll = m1f_loglik(mu_b, alpha, beta)
        # Sanity check: M3-anchor LL must equal closed-form M3 LL up to
        # numerical precision. Tolerance 1e-6 catches accidental breakage.
        if abs(final_ll - ll_M3) > 1e-6:
            print(f"  [WARNING] M3-anchor LL differs from closed-form M3 LL "
                  f"by {final_ll - ll_M3:+.8f} — investigate.")
        converged_iter = 0
        elapsed = 0.0
        print(f"  [M3-anchor] alpha=0 by construction. LL={final_ll:.4f}")
        print(f"             (matches LL_M3={ll_M3:.4f} up to numerical precision)")
        log_lines.append(f"  [M3-anchor] LL={final_ll:.4f} (=LL_M3 by construction)")
    else:
        # Standard 150-iter EM with M3 warm-start baseline
        mu_b = mu_b_M3.copy()
        alpha = alpha0
        beta = beta0
        prev_ll = -np.inf
        converged_iter = -1
        t_start = time.time()

        for it in range(150):
            mu_b, alpha, beta = fit_m1f_one_iter(mu_b, alpha, beta)
            if it % 5 == 0 or it < 3:
                ll = m1f_loglik(mu_b, alpha, beta)
                elapsed_now = time.time() - t_start
                line = (f"  iter {it:3d}: alpha={alpha:.6f} beta={beta:.6f} "
                        f"LL={ll:.4f} ({elapsed_now:.1f}s)")
                print(line)
                log_lines.append(line)
                if abs(ll - prev_ll) < 0.01:
                    converged_iter = it
                    break
                prev_ll = ll
        final_ll = m1f_loglik(mu_b, alpha, beta)
        elapsed = time.time() - t_start

    final_line = (f"  FINAL: alpha={alpha:.6f}, beta={beta:.6f}, "
                  f"LL={final_ll:.4f}, conv_iter={converged_iter}, time={elapsed:.1f}s")
    print(final_line)
    log_lines.append(final_line)

    m1f_results.append({
        "init": name,
        "init_alpha": alpha0,
        "init_beta": beta0,
        "is_m3_anchor": is_anchor,
        "alpha": alpha,
        "beta": beta,
        "LL": final_ll,
        "converged_iter": converged_iter,
        "elapsed_sec": elapsed,
        "mu_b_min": mu_b.min(),
        "mu_b_max": mu_b.max(),
    })

    if best_state is None or final_ll > best_state["LL"]:
        best_state = {
            "init": name,
            "mu_b": mu_b.copy(),
            "alpha": alpha,
            "beta": beta,
            "LL": final_ll,
        }

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("MULTI-START SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(m1f_results)
print(results_df.to_string(index=False, float_format='%.6f'))

best = results_df.iloc[results_df["LL"].idxmax()]
print(f"\n[BEST]: {best['init']} init, alpha={best['alpha']:.6f}, "
      f"beta={best['beta']:.6f}, LL={best['LL']:.4f}")

if best['init'] == 'S0':
    print("\n  [BEST is M3-anchor (alpha = 0)] — there is no detectable")
    print("  self-excitation beyond the M3 baseline. The best EM init lands")
    print("  slightly below this anchor due to the boundary numerical issue.")

alpha_range_em = (
    results_df.loc[~results_df["is_m3_anchor"], "alpha"].max()
    - results_df.loc[~results_df["is_m3_anchor"], "alpha"].min()
)
ll_range_em = (
    results_df.loc[~results_df["is_m3_anchor"], "LL"].max()
    - results_df.loc[~results_df["is_m3_anchor"], "LL"].min()
)
print(f"\nConvergence diagnostics (across 7 EM inits, excluding M3-anchor):")
print(f"  alpha range: {alpha_range_em:.6f} (max - min)")
print(f"  LL range:    {ll_range_em:.4f}")

# Best EM-only diagnostic: show how close positive-α inits get to S0 (M3-anchor).
# If best_em_LL < ll_M3, every positive-α EM landed below the M3 boundary —
# that is the boundary numerical artifact the M3-anchor was added to fix.
em_df = results_df.loc[~results_df["is_m3_anchor"]].copy().reset_index(drop=True)
best_em_idx = em_df["LL"].idxmax()
best_em = em_df.iloc[best_em_idx]
print(f"  Best EM init (excluding S0): {best_em['init']}, "
      f"alpha={best_em['alpha']:.6f}, LL - LL_M3 = {best_em['LL'] - ll_M3:+.6f}")
if best_em["LL"] < ll_M3:
    print(f"  → All 7 positive-α EM inits land below the M3-anchor LL.")
    print(f"    The M3-anchor S0 fix is what makes LR_obs >= 0 here.")

# ============================================================
# LR Test: M1f vs M3 (now LL(M1f) >= LL(M3) by construction)
# ============================================================
print("\n" + "=" * 70)
print("LR TEST: M1f vs M3 (proper test of self-excitation given baseline)")
print("=" * 70)

ll_M1f = best["LL"]
LR_raw = 2 * (ll_M1f - ll_M3)
LR = max(0.0, LR_raw)   # Defensive; with M3-anchor this should already be >= 0
p_chi2 = 1 - chi2.cdf(LR, df=2)
p_boundary = 0.5 * (1 - chi2.cdf(LR, df=1)) + 0.5 * (1 - chi2.cdf(LR, df=2))

print(f"  LL(M3)     = {ll_M3:.4f}  (k=48)")
print(f"  LL(M1f)    = {ll_M1f:.4f} (k=50, best multi-start)")
print(f"  LR_raw     = 2 * (LL_M1f - LL_M3) = {LR_raw:+.4f}")
print(f"  LR         = max(0, LR_raw) = {LR:.4f}")
print()
print(f"  p (chi2_2):                {p_chi2:.6g}")
print(f"  p (Self-Liang mixture):    {p_boundary:.6g}")
print()
print("  CAVEATS:")
print("    1. Under H0: alpha = 0, beta is unidentified. The standard")
print("       chi2 reference and the Self-Liang chi-bar mixture are both")
print("       heuristic — neither is the strictly correct asymptotic null")
print("       distribution when a nuisance parameter is unidentified at")
print("       the boundary (see Andrews 2001 / Hansen 1996 for the proper")
print("       nonstandard reference). Treat both p-values as diagnostic.")
print("    2. The paper-level inference uses the parametric bootstrap in")
print("       11_parametric_bootstrap_v3.py instead of a closed-form null.")

# AIC comparison
aic_M3 = 2 * 48 - 2 * ll_M3
aic_M1f = 2 * 50 - 2 * ll_M1f
bic_M3 = 48 * np.log(n_total_sub) - 2 * ll_M3
bic_M1f = 50 * np.log(n_total_sub) - 2 * ll_M1f
print(f"\n  AIC(M3)  = {aic_M3:.2f}      BIC(M3)  = {bic_M3:.2f}")
print(f"  AIC(M1f) = {aic_M1f:.2f}      BIC(M1f) = {bic_M1f:.2f}")
print(f"  ΔAIC (M1f - M3) = {aic_M1f - aic_M3:+.2f}")
print(f"  ΔBIC (M1f - M3) = {bic_M1f - bic_M3:+.2f}")

# Boundary alpha interpretation
if best['alpha'] < ALPHA_INTERPRET_THRESHOLD:
    print()
    print(f"  [INTERPRETATION] best alpha = {best['alpha']:.2e} is below")
    print(f"  ALPHA_INTERPRET_THRESHOLD ({ALPHA_INTERPRET_THRESHOLD:.0e}). The")
    print(f"  decay parameter beta = {best['beta']:.6f} is weakly identified")
    print(f"  at the boundary and is NOT substantively interpreted.")

# ============================================================
# Save outputs (FIX [I9]: best_baseline.csv for cluster_robust)
# ============================================================
results_df.to_csv("m1f_results.csv", index=False)
print(f"\n[Saved] m1f_results.csv ({len(m1f_results)} rows)")

pd.DataFrame({"bin": np.arange(N_BINS), "mu_M3": mu_b_M3}).to_csv("m1f_baseline.csv", index=False)
print(f"[Saved] m1f_baseline.csv (M3 reference baseline, 48 bins)")

# Best baseline (consumed by cluster_robust_ses)
# Schema includes alpha, beta, LL_M1f, init replicated on every row so that
# downstream scripts can read either the per-bin baseline OR the model
# parameters from a single file.
best_baseline_df = pd.DataFrame({
    "bin": np.arange(N_BINS),
    "mu_b_best": best_state["mu_b"],
    "alpha": best_state["alpha"],
    "beta": best_state["beta"],
    "LL_M1f": best_state["LL"],
    "init": best_state["init"],
})
best_baseline_df.to_csv("m1f_best_baseline.csv", index=False)
print(f"[Saved] m1f_best_baseline.csv (best M1f baseline + alpha/beta/LL, 48 bins)")

# Auxiliary: best parameters in flat csv
pd.DataFrame([{
    "init": best_state["init"],
    "alpha": best_state["alpha"],
    "beta": best_state["beta"],
    "LL_M1f": best_state["LL"],
    "LL_M3": ll_M3,
    "LR_raw": LR_raw,
    "LR_clipped": LR,
    "AIC_M1f": aic_M1f,
    "AIC_M3": aic_M3,
    "BIC_M1f": bic_M1f,
    "BIC_M3": bic_M3,
    "best_em_init_excluding_S0": best_em["init"],
    "best_em_alpha_excluding_S0": best_em["alpha"],
    "best_em_LL_excluding_S0": best_em["LL"],
    "best_em_LL_minus_M3": best_em["LL"] - ll_M3,
    "alpha_above_threshold": bool(best_state["alpha"] > ALPHA_INTERPRET_THRESHOLD),
    "n_real": n_real,
    "n_total_sub": n_total_sub,
    "ALPHA_INTERPRET_THRESHOLD": ALPHA_INTERPRET_THRESHOLD,
}]).to_csv("m1f_best_params.csv", index=False)
print(f"[Saved] m1f_best_params.csv (best alpha/beta + test stats + EM diagnostics)")

with open("m1f_log.txt", "w") as f:
    f.write("\n".join(log_lines))
    f.write("\n\n" + "=" * 70 + "\n")
    f.write(f"M3 LL = {ll_M3:.4f} (k=48)\n")
    f.write(f"M1f best LL = {ll_M1f:.4f} (k=50)  [init={best_state['init']}]\n")
    f.write(f"LR_raw = {LR_raw:.4f}\n")
    f.write(f"LR (clipped at 0) = {LR:.4f}\n")
    f.write(f"p (chi2_2):    {p_chi2:.6g}\n")
    f.write(f"p (boundary):  {p_boundary:.6g}\n")
    f.write(f"alpha range across EM inits: {alpha_range_em:.6f}\n")
    f.write(f"LL range across EM inits:    {ll_range_em:.4f}\n")
    f.write(f"Best EM init excluding S0: {best_em['init']}\n")
    f.write(f"Best EM alpha excluding S0: {best_em['alpha']:.8g}\n")
    f.write(f"Best EM LL - LL_M3: {best_em['LL'] - ll_M3:+.8f}\n")
print(f"[Saved] m1f_log.txt (full convergence log)")

print("\n" + "=" * 70)
print("M1f COMPLETE")
print("=" * 70)
print(f"  Best init:    {best_state['init']}")
print(f"  Best alpha:   {best_state['alpha']:.6f}")
print(f"  Best beta:    {best_state['beta']:.6f}")
print(f"  Best LL:      {best_state['LL']:.4f}")
print(f"  ΔLL_M1f-M3:   {ll_M1f - ll_M3:+.4f}")
print(f"  Verdict:      {'No detectable self-excitation beyond M3 baseline' if best_state['alpha'] < ALPHA_INTERPRET_THRESHOLD else 'Detectable self-excitation: alpha above threshold'}")