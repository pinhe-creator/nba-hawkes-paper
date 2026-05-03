"""
cluster_robust_ses_v2.py
========================

Cluster-Robust Standard Errors at Game Level for M1f Hawkes Estimation
======================================================================

PURPOSE:
This script computes cluster-robust standard errors for the M1f model
parameters (48 baseline rates + alpha + beta) at the GAME level. This
addresses Limitation L3 in Section 8.5 of the paper, which notes that the
working independence assumption in EM may understate the true sampling
variance because each game contributes 2 (game, team) realizations whose
noise is likely correlated.

USAGE:
    Place in same directory as filtered_3_seasons.csv.gz, m1f_best_baseline.csv,
    and m1f_best_params.csv. Then:
        python cluster_robust_ses_v2.py

DATA PIPELINE (matches step_B / phase4 / fit_m1f exactly):
- Filter raw NBA play-by-play to substitution events (EVENTMSGTYPE == 8)
- Compute absolute seconds from PERIOD + PCTIMESTRING
- Mass-aggregate within (GAME_ID, TEAM_ID, PERIOD, t_abs)
- Restrict to regulation play (PERIOD <= 4)
- Build R = 7,380 realizations from 3,690 games × 2 teams

METHODOLOGY:
Cluster-robust sandwich estimator (Liang & Zeger 1986; Cameron-Miller 2015):

    V_cluster = c · H^{-1} [Σ_g s_g s_g'] H^{-1}

where:
- H = OPG estimator of Fisher information (Σ score outer products per
      realization), asymptotically equivalent to negative Hessian under
      correct specification (Wooldridge 2010 ch.13).
- s_g = sum over realizations r in game g of the score vector ∂ell_r/∂theta
- The cluster is the GAME (each game has 2 realizations: home + away team)
- c = (G / (G-1)) · ((R-1) / (R-p))   small-sample correction

INPUT FILES (must be in same directory):
- filtered_3_seasons.csv.gz  : raw NBA play-by-play
- m1f_best_baseline.csv      : best M1f baseline (48 bins) + alpha + beta + LL
                               [from fit_m1f_v4.py]
- m1f_best_params.csv        : optional flat alpha/beta/LL/LR table
                               [from fit_m1f_v4.py — not required by this
                                script, but useful for cross-checking]

OUTPUT FILES:
- cluster_robust_ses.csv     : parameter, point_estimate, se_independence,
                               se_cluster, ratio, 95% CI under cluster-robust
- cluster_diagnostic.txt     : summary focusing on alpha and beta

EXPECTED RUNTIME: ~10-15 minutes on a modern laptop

================================================================================
FIXES vs original 10_cluster_robust_ses.py:
================================================================================
[B2]   CRITICAL parametrization mismatch with fit_m1f.
       Original used:    λ(t) = mu_b + α · Σ exp(-β(t - t_j))
       fit_m1f uses:     λ(t) = mu_b + α·β · Σ exp(-β(t - t_j))
       These differ by a factor of β. Reading α_hat from fit_m1f and using
       it in the original code's intensity formula would mis-scale α by 250×
       (since β ≈ 0.004). Fixed: this script now uses fit_m1f's parametrization
       throughout — both for log-likelihood and for the score vector.

[B2.1] alpha score formula updated to match the new parametrization:
         ∂λ_i/∂α   = β · R[i]
         ∂L/∂α     = β · Σ_i R[i] / λ_i  −  Σ_j (1 − exp(−β(T − t_j)))
       (Original was: Σ_i R[i]/λ_i − (1/β) Σ_j (...) which was the score for
        the OTHER parametrization.)

[B2.2] Baseline source.
       Original called fit_m3_baseline() and combined it with α_hat from
       m1f_results.csv. But fit_m1f re-estimates the baseline μ_b in EM
       together with α and β; the best-fit μ_b is in m1f_best_baseline.csv.
       Fixed: this script reads μ_b directly from m1f_best_baseline.csv,
       so the variance computation is at the actual M1f MLE, not at
       (M3 baseline + M1f α).

[I12]  beta score by central finite difference (kept), but with safer step
       size and a guard against beta near zero.

[B5 INTERACTION] Best M1f fit at α = 0 (boundary).
       fit_m1f_v4 found that the multi-start best is the M3-anchor (S0)
       with α = 0 exactly. At α = 0:
         - β is unidentified (the score ∂L/∂β = 0 at α = 0 by inspection)
         - The information matrix has a 0 row/column for β
         - The sandwich is degenerate in the (β) direction
       This script handles that case explicitly: when α_hat below
       ALPHA_INTERPRET_THRESHOLD, the SE for β is reported as np.nan with
       a clear caveat. SE for α is reported but interpreted as
       "boundary SE; treat parametric bootstrap as primary inference."

VERSION: v0.4
================================================================================
"""

import numpy as np
import pandas as pd
import time
import os
import sys

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FILE   = "filtered_3_seasons.csv.gz"
BEST_BASELINE_FILE = "m1f_best_baseline.csv"   # FIX [B2.2]
# BEST_PARAMS_FILE = "m1f_best_params.csv"     # not required (m1f_best_baseline
#                                              #   already contains alpha/beta/LL)
OUTPUT_SES  = "cluster_robust_ses.csv"
OUTPUT_LOG  = "cluster_diagnostic.txt"

T_HORIZON = 2880.0
N_BINS    = 48
BIN_W     = T_HORIZON / N_BINS
LAM_FLOOR = 1e-12
ALPHA_INTERPRET_THRESHOLD = 1e-4

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
def load_and_prepare_data():
    """Load raw play-by-play and transform into realizations + game_ids."""
    print(f"[Loading] {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        sys.exit(f"[FATAL] {DATA_FILE} not found.")
    t0 = time.time()
    df = pd.read_csv(DATA_FILE, compression='gzip', low_memory=False)
    n_games = df['GAME_ID'].nunique()
    print(f"  Loaded {len(df):,} rows, {n_games:,} games in {time.time()-t0:.1f}s")
    if n_games < 3000:
        print(f"  [WARNING] Expected ~3,690 games, found {n_games:,}.")

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
    mass_subs = (
        subs.groupby(["GAME_ID", "TEAM_ID", "PERIOD", "t_abs"], as_index=False)
            .agg(n_players=("PLAYER1_ID", "count"))
    )
    mass_subs = mass_subs.sort_values(["GAME_ID", "TEAM_ID", "t_abs"]).reset_index(drop=True)
    mass_reg = mass_subs[mass_subs["PERIOD"] <= 4].copy()

    n_at_or_after_T = (mass_reg["t_abs"] >= T_HORIZON).sum()
    if n_at_or_after_T > 0:
        print(f"  [INFO] Dropping {n_at_or_after_T:,} regulation events at t_abs >= {T_HORIZON}.")

    print("[Building realizations]")
    realizations = []
    game_ids = []
    for (gid, tid), g in mass_reg.groupby(["GAME_ID", "TEAM_ID"]):
        sub_t = np.sort(g["t_abs"].values.astype(float))
        sub_t = sub_t[sub_t < T_HORIZON]
        if len(sub_t) > 0:
            realizations.append(sub_t)
            game_ids.append(gid)
    game_ids = np.array(game_ids)
    n_total = sum(len(r) for r in realizations)
    print(f"  R = {len(realizations):,} realizations, {n_total:,} events, "
          f"{len(np.unique(game_ids)):,} unique games")
    return realizations, game_ids, n_total


# -----------------------------------------------------------------------------
# Score and log-likelihood with FIX [B2] / [B2.1]
# -----------------------------------------------------------------------------
def m1f_score_one_realization(events, mu_b, alpha, beta):
    """
    Compute log-likelihood and score for one realization, using fit_m1f's
    parametrization (FIX [B2]):
        λ(t) = mu_b[bin(t)] + α · β · Σ_{t_j < t} exp(-β (t - t_j))

    Returns:
        ll:    scalar log-likelihood for this realization
        score: array of length N_BINS + 2, gradient w.r.t. (mu_0..mu_47, α, β)

    Score derivations (FIX [B2.1]):
      ∂λ_i/∂α = β · R[i]
      ∂λ_i/∂μ_k = 1 if bin(t_i) == k else 0

      Log-likelihood: L = Σ_i log λ_i − ∫₀ᵀ λ(t) dt

      Compensator:
        ∫₀ᵀ μ_b dt   = Σ_k μ_k · BIN_W
        ∫₀ᵀ α·β·exp(−β(t−t_j))·1{t > t_j} dt = α · (1 − exp(−β(T − t_j)))
        Total excitation comp = α · Σ_j (1 − exp(−β(T − t_j)))

      Scores:
        ∂L/∂μ_k  = Σ_i 1{bin(t_i)=k}/λ_i − BIN_W
        ∂L/∂α   = β · Σ_i R[i]/λ_i − Σ_j (1 − exp(−β(T − t_j)))
        ∂L/∂β   : finite difference (Hawkes recursion derivative is awkward)
    """
    n_events = len(events)
    bin_durations = np.full(N_BINS, BIN_W)

    if n_events == 0:
        # Only baseline compensator contributes
        ll = -np.sum(mu_b * bin_durations)
        score = np.zeros(N_BINS + 2)
        score[:N_BINS] = -bin_durations   # ∂L/∂μ_k = -BIN_W
        return ll, score

    # ---- R[i] = Σ_{j<i} exp(-β (t_i - t_j))  via Hawkes recursion ----
    R = np.zeros(n_events)
    R[0] = 0.0
    for i in range(1, n_events):
        delta = events[i] - events[i - 1]
        R[i] = np.exp(-beta * delta) * (R[i - 1] + 1.0)

    bin_of_event = np.minimum((events / BIN_W).astype(np.int64), N_BINS - 1)
    mu_at_event = mu_b[bin_of_event]

    # FIX [B2]: intensity parametrization is α·β·R, not α·R
    lambdas = mu_at_event + alpha * beta * R
    lambdas_safe = np.maximum(lambdas, LAM_FLOOR)

    # ---- Log-likelihood ----
    log_lam_sum = np.sum(np.log(lambdas_safe))
    baseline_integral = np.sum(mu_b * bin_durations)
    T_minus_t = T_HORIZON - events
    # FIX [B2]: compensator for excitation is α·Σ(1−exp(−β(T−t_j)))
    # (no 1/β factor, because λ has α·β prefactor)
    excitation_integral = alpha * np.sum(1.0 - np.exp(-beta * T_minus_t))
    compensator = baseline_integral + excitation_integral
    ll = log_lam_sum - compensator

    # ---- Score ----
    score = np.zeros(N_BINS + 2)
    inv_lam = 1.0 / lambdas_safe

    # ∂L/∂μ_k
    for k in range(N_BINS):
        mask = (bin_of_event == k)
        score[k] = np.sum(inv_lam[mask]) - bin_durations[k]

    # FIX [B2.1]: ∂L/∂α = β·Σ R[i]/λ_i − Σ_j (1 − exp(−β(T−t_j)))
    score[N_BINS] = beta * np.sum(R * inv_lam) - np.sum(1.0 - np.exp(-beta * T_minus_t))

    # ∂L/∂β by central finite difference.
    # At alpha < ALPHA_INTERPRET_THRESHOLD, β is effectively unidentified —
    # the log-likelihood does not depend on β at α = 0, so the score is
    # exactly zero. Set to 0 to avoid numerical noise contaminating the
    # information matrix in the boundary regime.
    if alpha < ALPHA_INTERPRET_THRESHOLD:
        score[N_BINS + 1] = 0.0
    else:
        h = max(1e-7, abs(beta) * 1e-5)
        if h <= 0 or beta - h <= 0:
            score[N_BINS + 1] = 0.0
        else:
            ll_plus = _ll_only(events, mu_b, alpha, beta + h)
            ll_minus = _ll_only(events, mu_b, alpha, beta - h)
            score[N_BINS + 1] = (ll_plus - ll_minus) / (2 * h)

    return ll, score


def _ll_only(events, mu_b, alpha, beta):
    """Helper: compute log-likelihood only (no gradient). For finite-difference."""
    n_events = len(events)
    bin_durations = np.full(N_BINS, BIN_W)
    if n_events == 0:
        return -np.sum(mu_b * bin_durations)

    R = np.zeros(n_events)
    for i in range(1, n_events):
        delta = events[i] - events[i - 1]
        R[i] = np.exp(-beta * delta) * (R[i - 1] + 1.0)

    bin_of_event = np.minimum((events / BIN_W).astype(np.int64), N_BINS - 1)
    mu_at_event = mu_b[bin_of_event]
    # FIX [B2]: intensity = mu + α·β·R (not mu + α·R)
    lambdas_safe = np.maximum(mu_at_event + alpha * beta * R, LAM_FLOOR)

    log_lam_sum = np.sum(np.log(lambdas_safe))
    baseline_integral = np.sum(mu_b * bin_durations)
    T_minus_t = T_HORIZON - events
    excitation_integral = alpha * np.sum(1.0 - np.exp(-beta * T_minus_t))
    return log_lam_sum - (baseline_integral + excitation_integral)


# -----------------------------------------------------------------------------
# CLUSTER-ROBUST VARIANCE
# -----------------------------------------------------------------------------
def compute_cluster_robust_variance(realizations, game_ids, mu_b, alpha, beta,
                                     is_boundary=False):
    """
    V_cluster = c · H^{-1} · B · H^{-1}
    H = OPG (sum of score outer products per realization)
    B = sum over games of (sum of scores within game) outer-product itself
    c = (G/(G-1)) · ((R-1)/(R-p))

    [v5 BOUNDARY HANDLING] When is_boundary=True (i.e. α at the α=0 boundary
    so β is unidentified and its score is 0 by construction), the full H
    matrix has an all-zero β row/column. Computing H^{-1} via 50×50
    inversion is ill-conditioned and produces RuntimeWarnings (divide by
    zero, overflow). Instead, compute the inverse on the well-conditioned
    49×49 sub-matrix (excluding β), then pad the β row/column with NaN.

    Mathematically: when score_β ≡ 0 across all realizations, the (μ_b, α)
    block of H is decoupled from β, and the sub-matrix inverse yields the
    correct sub-Hessian inverse for the identified parameters.
    """
    n_params = N_BINS + 2
    R = len(realizations)
    H = np.zeros((n_params, n_params))
    score_per_realization = np.zeros((R, n_params))

    print(f"\n[Computing scores for {R} realizations]")
    t0 = time.time()
    for idx, events in enumerate(realizations):
        ll, score = m1f_score_one_realization(events, mu_b, alpha, beta)
        score_per_realization[idx] = score
        H += np.outer(score, score)
        if (idx + 1) % 1000 == 0 or (idx + 1) == R:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (R - idx - 1) / rate if rate > 0 else 0
            print(f"  {idx+1}/{R} ({elapsed:.1f}s, eta {eta:.1f}s)")
    print(f"  Done in {time.time() - t0:.1f}s")

    # Aggregate cluster meat matrix
    unique_games = np.unique(game_ids)
    G = len(unique_games)
    print(f"\n[Aggregating scores by game cluster ({G} games)]")
    B = np.zeros((n_params, n_params))
    for gid in unique_games:
        mask = (game_ids == gid)
        s_g = score_per_realization[mask].sum(axis=0)
        B += np.outer(s_g, s_g)

    # Effective parameter count (FIX [#8]): in the boundary case β is
    # excluded from the identified parameters used in the sandwich.
    p_eff = (N_BINS + 1) if is_boundary else (N_BINS + 2)
    cluster_correction = (G / (G - 1.0)) * ((R - 1.0) / max(R - p_eff, 1.0))

    if is_boundary:
        # The boundary case makes the full information matrix singular
        # because β is unidentified. Compute the sandwich only for the
        # identified block (μ_b, α). The OPG matrix on this block can
        # still be ill-conditioned because μ_b Fisher info (~5e9) and α
        # Fisher info (~2e5) span ~4 orders of magnitude. Diagnose the
        # condition number, fall back to pseudo-inverse if ill-conditioned,
        # symmetrize, and verify the result is finite.
        n_sub = N_BINS + 1   # μ_0..μ_47, α
        H_sub = H[:n_sub, :n_sub]
        B_sub = B[:n_sub, :n_sub]
        H_sub_reg = H_sub + 1e-8 * np.eye(n_sub)   # FIX [#4]: stronger ridge

        # FIX [#1]: condition-number diagnostic
        cond_H_sub = np.linalg.cond(H_sub_reg)
        print(f"  [Diagnostic] cond(H_sub_reg) = {cond_H_sub:.3e}")
        if (not np.isfinite(cond_H_sub)) or cond_H_sub > 1e12:
            # FIX [#2]: pseudo-inverse fallback for very ill-conditioned H
            print("  [Warning] H_sub is ill-conditioned; using pseudo-inverse "
                  "with rcond=1e-10")
            H_sub_inv = np.linalg.pinv(H_sub_reg, rcond=1e-10)
        else:
            try:
                H_sub_inv = np.linalg.inv(H_sub_reg)
            except np.linalg.LinAlgError:
                print("  [Warning] H_sub not invertible; using pseudo-inverse")
                H_sub_inv = np.linalg.pinv(H_sub_reg, rcond=1e-10)

        V_indep_sub = H_sub_inv
        V_cluster_sub = cluster_correction * (H_sub_inv @ B_sub @ H_sub_inv)

        # FIX [#5]: symmetrize to remove small numerical asymmetry
        V_indep_sub   = 0.5 * (V_indep_sub   + V_indep_sub.T)
        V_cluster_sub = 0.5 * (V_cluster_sub + V_cluster_sub.T)

        # FIX [#3]: verify finite values; retry with stronger pseudo-inverse
        # rcond if non-finite values appear.
        if not np.all(np.isfinite(V_cluster_sub)):
            print("  [Warning] V_cluster_sub contains non-finite values; "
                  "retrying with rcond=1e-8")
            H_sub_inv = np.linalg.pinv(H_sub_reg, rcond=1e-8)
            V_indep_sub = H_sub_inv
            V_cluster_sub = cluster_correction * (H_sub_inv @ B_sub @ H_sub_inv)
            V_indep_sub   = 0.5 * (V_indep_sub   + V_indep_sub.T)
            V_cluster_sub = 0.5 * (V_cluster_sub + V_cluster_sub.T)
        if not np.all(np.isfinite(V_cluster_sub)):
            print("  [ERROR] V_cluster_sub still contains non-finite values.")
            print("          Cluster-robust SEs may be unreliable.")

        # Pad to 50×50, β row/col = NaN
        V_indep   = np.full((n_params, n_params), np.nan)
        V_cluster = np.full((n_params, n_params), np.nan)
        V_indep[:n_sub, :n_sub]   = V_indep_sub
        V_cluster[:n_sub, :n_sub] = V_cluster_sub
        print(f"  [BOUNDARY] computed V on 49×49 identified block; "
              f"β row/col = NaN (p_eff = {p_eff})")
    else:
        # Standard 50×50 inversion
        H_reg = H + 1e-10 * np.eye(n_params)
        cond_H = np.linalg.cond(H_reg)
        print(f"  [Diagnostic] cond(H_reg) = {cond_H:.3e}")
        if (not np.isfinite(cond_H)) or cond_H > 1e12:
            print("  [Warning] H is ill-conditioned; using pseudo-inverse")
            H_inv = np.linalg.pinv(H_reg, rcond=1e-10)
        else:
            try:
                H_inv = np.linalg.inv(H_reg)
            except np.linalg.LinAlgError:
                print("  [Warning] H not invertible; using pseudo-inverse")
                H_inv = np.linalg.pinv(H_reg, rcond=1e-10)
        V_indep = H_inv
        V_cluster = cluster_correction * (H_inv @ B @ H_inv)
        # Symmetrize
        V_indep   = 0.5 * (V_indep   + V_indep.T)
        V_cluster = 0.5 * (V_cluster + V_cluster.T)
        if not np.all(np.isfinite(V_cluster)):
            print("  [ERROR] V_cluster contains non-finite values.")

    return V_cluster, V_indep, G, R


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CLUSTER-ROBUST STANDARD ERRORS FOR M1f")
    print("Clustering at GAME level — λ = μ_b + α·β·R parametrization (FIX B2)")
    print("=" * 70)

    # ---- 1. Load raw data and prepare realizations ----
    realizations, game_ids, n_total = load_and_prepare_data()

    # ---- 2. Read M1f best fit (FIX [B2.2]: use best_baseline, not refit M3) ----
    print(f"\n[Reading M1f best fit]")
    if not os.path.exists(BEST_BASELINE_FILE):
        sys.exit(f"[FATAL] {BEST_BASELINE_FILE} not found. Run fit_m1f_v4.py first.")
    bb = pd.read_csv(BEST_BASELINE_FILE)
    expected_cols = {"bin", "mu_b_best", "alpha", "beta", "LL_M1f", "init"}
    missing = expected_cols - set(bb.columns)
    if missing:
        sys.exit(f"[FATAL] {BEST_BASELINE_FILE} missing columns {missing}. "
                 f"Re-run fit_m1f_v4.py to regenerate.")
    if len(bb) != N_BINS:
        sys.exit(f"[FATAL] {BEST_BASELINE_FILE} has {len(bb)} rows, expected {N_BINS}.")
    bb = bb.sort_values("bin").reset_index(drop=True)
    mu_b = bb["mu_b_best"].values.astype(float)
    alpha_hat = float(bb["alpha"].iloc[0])
    beta_hat = float(bb["beta"].iloc[0])
    ll_M1f = float(bb["LL_M1f"].iloc[0])
    init_name = str(bb["init"].iloc[0])
    print(f"  Best init:    {init_name}")
    print(f"  alpha_hat:    {alpha_hat:.6e}")
    print(f"  beta_hat:     {beta_hat:.6e}")
    print(f"  LL_M1f:       {ll_M1f:.4f}")
    print(f"  μ_b range:    [{mu_b.min():.6f}, {mu_b.max():.6f}]")

    # Sanity check
    if alpha_hat < 0 or beta_hat <= 0:
        print(f"  [WARNING] Unusual parameter values: alpha={alpha_hat}, beta={beta_hat}")

    # Boundary case detection (FIX [B5 INTERACTION])
    is_boundary = alpha_hat < ALPHA_INTERPRET_THRESHOLD
    if is_boundary:
        print(f"\n  [BOUNDARY] alpha_hat < {ALPHA_INTERPRET_THRESHOLD:.0e}.")
        print(f"             β is weakly identified at the α=0 boundary;")
        print(f"             the Fisher information for β collapses, and the")
        print(f"             cluster-robust SE for β will be reported as NaN")
        print(f"             because β is not identified when α is at the boundary.")

    # ---- 3. Compute cluster-robust variance ----
    V_cluster, V_indep, G, R = compute_cluster_robust_variance(
        realizations, game_ids, mu_b, alpha_hat, beta_hat,
        is_boundary=is_boundary,
    )

    # ---- 4. Extract SEs ----
    diag_cluster = np.diag(V_cluster)
    diag_indep = np.diag(V_indep)
    se_cluster = np.sqrt(np.maximum(diag_cluster, 0.0))
    se_indep   = np.sqrt(np.maximum(diag_indep, 0.0))

    param_names = [f'mu_{k}' for k in range(N_BINS)] + ['alpha', 'beta']
    point_estimates = list(mu_b) + [alpha_hat, beta_hat]

    rows = []
    for i, (name, est) in enumerate(zip(param_names, point_estimates)):
        ratio = se_cluster[i] / se_indep[i] if se_indep[i] > 0 else np.nan
        ci_lo = est - 1.96 * se_cluster[i]
        ci_hi = est + 1.96 * se_cluster[i]
        # Wald CI is valid for interior parameters (μ_b bins). For α and β at
        # the boundary, conventional Wald inference is not asymptotically valid
        # (Andrews 2001) and these SEs are diagnostic only.
        if name == "alpha":
            valid_wald = (not is_boundary)
        elif name == "beta":
            valid_wald = (not is_boundary)
        else:
            valid_wald = True
        rows.append({
            'parameter': name,
            'point_estimate': est,
            'se_independence': se_indep[i],
            'se_cluster': se_cluster[i],
            'ratio_cluster_over_indep': ratio,
            'ci95_lo_cluster': ci_lo,
            'ci95_hi_cluster': ci_hi,
            'valid_wald_inference': valid_wald,
        })
    df_out = pd.DataFrame(rows)

    # FIX [#1]: At the α = 0 boundary, β is unidentified; the OPG sandwich
    # gives a numerically-determined β SE that is not statistically meaningful.
    # Setting these to NaN matches the behavior promised in the docstring and
    # prevents downstream tables from displaying spurious β SEs.
    if is_boundary:
        beta_mask = df_out["parameter"] == "beta"
        df_out.loc[beta_mask, [
            "se_independence",
            "se_cluster",
            "ratio_cluster_over_indep",
            "ci95_lo_cluster",
            "ci95_hi_cluster",
        ]] = np.nan

    df_out.to_csv(OUTPUT_SES, index=False)
    print(f"\n[Saved] {OUTPUT_SES}")

    # ---- 5. Diagnostic summary ----
    alpha_row = df_out[df_out['parameter'] == 'alpha'].iloc[0]
    beta_row  = df_out[df_out['parameter'] == 'beta'].iloc[0]
    mu_rows = df_out[df_out['parameter'].str.startswith('mu_')]
    mu_ratio_mean = mu_rows['ratio_cluster_over_indep'].mean()
    mu_ratio_med  = mu_rows['ratio_cluster_over_indep'].median()
    mu_ratio_max  = mu_rows['ratio_cluster_over_indep'].max()

    log_lines = [
        "=" * 70,
        "CLUSTER-ROBUST SE DIAGNOSTIC SUMMARY (v7 — boundary β excluded)",
        "=" * 70,
        "",
        f"Number of clusters (games): G = {G}",
        f"Number of realizations:     R = {R}",
        f"Realizations per cluster:   {R/G:.2f} (expected ~2: home + away team)",
        f"Total parameters:           p_total = {N_BINS + 2}",
        f"Identified parameters used: p_eff = "
        f"{(N_BINS + 1) if is_boundary else (N_BINS + 2)}"
        f"  ({'boundary: β excluded' if is_boundary else 'all params identified'})",
        f"Small-sample correction:    "
        f"{(G/(G-1.0)) * ((R-1.0) / max(R - ((N_BINS + 1) if is_boundary else (N_BINS + 2)), 1.0)):.6f}",
        "",
        f"Parametrization:            λ = μ_b + α·β·R   (matches fit_m1f)",
        f"Best M1f init:              {init_name}",
        f"alpha_hat:                  {alpha_hat:.6e}",
        f"beta_hat:                   {beta_hat:.6e}",
        f"M1f LL at this point:       {ll_M1f:.4f}",
        f"Boundary case (α < {ALPHA_INTERPRET_THRESHOLD:.0e}): {is_boundary}",
        "",
        "FOCUS: ALPHA (self-excitation magnitude)",
        "-" * 70,
        f"  Point estimate:            alpha_hat = {alpha_hat:.6e}",
        f"  SE (working-independence): {alpha_row['se_independence']:.6e}",
        f"  SE (cluster-robust):       {alpha_row['se_cluster']:.6e}",
        f"  Ratio (cluster/indep):     {alpha_row['ratio_cluster_over_indep']:.4f}",
        f"  95% CI (cluster-robust):   [{alpha_row['ci95_lo_cluster']:.4e}, "
        f"{alpha_row['ci95_hi_cluster']:.4e}]",
        "",
        "FOCUS: BETA (decay rate)",
        "-" * 70,
        f"  Point estimate:            beta_hat = {beta_hat:.6e}",
        f"  SE (working-independence): {beta_row['se_independence']:.6e}",
        f"  SE (cluster-robust):       {beta_row['se_cluster']:.6e}",
        f"  Ratio (cluster/indep):     {beta_row['ratio_cluster_over_indep']:.4f}",
        "",
        "BASELINE PARAMETERS (mu_0 ... mu_47):",
        "-" * 70,
        f"  Mean ratio (cluster/indep):   {mu_ratio_mean:.4f}",
        f"  Median ratio (cluster/indep): {mu_ratio_med:.4f}",
        f"  Max ratio (cluster/indep):    {mu_ratio_max:.4f}",
        "",
        "INTERPRETATION FOR M1f vs M3 NESTED COMPARISON",
        "-" * 70,
        "",
        "The cluster-robust SE for alpha is the more honest measure of",
        "sampling uncertainty, accounting for the fact that home and away",
        "teams within the same game share common noise (referee tendencies,",
        "score-margin dynamics, time-out timing, broadcast-cue effects).",
        "",
    ]

    if is_boundary:
        log_lines += [
            "BOUNDARY CASE: alpha_hat is at or near zero",
            "-" * 70,
            "",
            "The maximum-likelihood M1f fit is the M3-anchor: alpha = 0, with",
            "beta unidentified. The cluster-robust SE for alpha is the SE at",
            "the boundary; conventional Wald inference (point ± 1.96·SE) is",
            "not asymptotically valid in this regime. The 95% CI shown is a",
            "naive interval, useful only as a diagnostic.",
            "",
            "The PRIMARY inferential reference is the parametric bootstrap in",
            "11_parametric_bootstrap_v3.py, which simulates the null",
            "distribution under M3 and compares the observed LR to it.",
            "",
            "The cluster-robust SE for beta is reported as NaN because β is",
            "unidentified at α = 0; it should not be substantively interpreted.",
            "",
            "Note on conditional dependence on β_hat:",
            "Although β is unidentified when α = 0, the α score itself depends",
            "on β through the formula β·Σ R[i]/λ_i − Σ_j (1 − exp(−β(T − t_j))).",
            "The reported cluster-robust SE for α is therefore conditional on",
            "the chosen / anchored β value (here β_hat = 0.01 from the M3-anchor",
            "init). Different anchor values of β would produce slightly different",
            "α SEs at α = 0. Because β is unidentified under H0, formal inference",
            "is the parametric bootstrap LR test rather than this Wald SE.",
            "",
        ]
    else:
        n_se_alpha = alpha_hat / max(alpha_row['se_cluster'], 1e-300)
        log_lines += [
            f"Wald-style diagnostic (NOT primary inference):",
            f"  alpha_hat / SE_cluster = {n_se_alpha:.2f}",
            "",
        ]

    log_lines += [
        "Note on the OPG approximation:",
        "We use the outer-product-of-gradients estimator for the information",
        "matrix (sum of score outer products), asymptotically equivalent to",
        "the negative Hessian under correct specification (Wooldridge 2010",
        "ch.13). An exact-Hessian alternative is feasible but adds significant",
        "code complexity for limited additional precision in this large-N",
        "regime (R = 7,380 realizations).",
        "",
        "Note on parametrization (FIX [B2]):",
        "v2 uses λ(t) = μ_b[bin(t)] + α·β·Σ exp(−β(t − t_j)) (Hawkes",
        "branching-ratio parametrization, matching fit_m1f_v4). Original",
        "v1 used λ = μ_b + α·R, which was a different parametrization. The",
        "alpha point estimate and its score formula here are consistent with",
        "fit_m1f_v4's M1f model.",
        "",
        "=" * 70,
    ]

    log_text = "\n".join(log_lines)
    with open(OUTPUT_LOG, "w") as f:
        f.write(log_text)
    print(f"[Saved] {OUTPUT_LOG}")
    print()
    print(log_text)


if __name__ == "__main__":
    main()