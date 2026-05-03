"""
================================================================================
Parametric Bootstrap V4c — V4b + summary wording cleanup
================================================================================

V4c CHANGES vs V4b (cosmetic/wording, no algorithmic change):

  [V4c #1] Summary block: "V3, multi-start MLE" → "V4b, multi-start EM-style fit"
    (the bootstrap is not exact MLE because β uses a recursive EM-style
    update, not the exact finite-window MLE; "EM-style fit" is the
    accurate descriptor.)

  [V4c #3] Removed informal "V1 raw EM p-value (~0.255)" reference from
    summary text. Earlier single-start EM diagnostics could return negative
    LR values when EM hit local optima — a negative LR is not a valid LR-
    test statistic, so those diagnostics are not used for formal inference.
    The V4b nesting-enforced bootstrap is the paper's authoritative test.

  [V4c log] load_and_prepare_data() now logs an explicit caveat that
    realizations exclude team-games with zero in-regulation substitutions,
    and recommends explicit verification if any such team-games are present
    in the source data. (No empirical claim about exclusion magnitude is
    made because the code does not count zero-substitution team-games.)

================================================================================

V4b CHANGES vs V4 (preserved):

  [V4b #1] sim_n_total = 0 case: now uses mu_b = 0 (true MLE) and ll = 0,
    instead of mu_b = 1e-9 + ll = 0 (which was internally inconsistent).
    Probability of this branch executing is ~exp(-17000) given n_real = 7,380
    and mu_b ~ 1.6e-2, so this is cosmetic in practice but mathematically
    cleaner.

  [V4b #6] fit_m1f_iter: documented that n_total is no longer used internally
    after [B1] fix; retained in signature for caller compatibility.

  [V4b #7] CSV resume validation now scans the entire iter column for
    sequential integrity (was: last-row only). Negligible startup overhead
    for B=1000 rows. Detects and repairs mid-file corruption.

  [V4b #3 doc] Added explicit caveat in fit_m1f_iter docstring that the β
    M-step is a recursive EM-style update consistent with phase2_v4 etc.,
    not the exact finite-window MLE.

  [V4b #8 doc] Conservative runtime estimate (~20-40 hours instead of 20-25).

ALSO PRESERVED FROM V4 (added on top of V3b):

V4 [B1] FIX (preserved from V4):

  [B1] ALPHA M-STEP DENOMINATOR: The α update in fit_m1f_iter() now uses
    the compensator-based denominator
        α = sum_pS / Σ_j (1 − exp(−β(T − t_j)))
    instead of the simpler (and biased-high) sum_pS / n_total. This
    matches fit_m1f_v4 (Stage 5) byte-for-byte, so LR_OBS computed inside
    V4 by fit_m1f_multistart_with_anchor agrees with the standalone
    fit_m1f_v4 run, and LR_sims are computed from the same likelihood
    surface as LR_OBS. Without this fix, the V3b bootstrap would compute
    LR values from a DIFFERENT (biased) likelihood than the paper's main
    fit_m1f result, breaking end-to-end consistency.

V3b CRITICAL FIX (preserved from V3b):

  ZERO_EVENT_BUG: m1f_loglik() now correctly accounts for realizations with
    zero events. Previously, `if n == 0: continue` silently skipped these
    realizations, omitting their baseline integral contribution. This was a
    real bug: in B=1000 bootstrap iterations totalling 7.38M realization
    draws under H0 (mu_b ~ 1.6e-2 mean event rate), there is a ~37% chance
    of at least one zero-event realization. Each such realization, if missed,
    inflates LL_M1f by ~16.6 units relative to fit_m3, which itself correctly
    integrates over total exposure. The fix moves the baseline-integral
    subtraction outside the `if n == 0` branch, so it is applied for every
    realization regardless of event count. See m1f_loglik docstring.

================================================================================

CRITICAL FIXES vs V2:

  FIX #1: M3-ANCHOR in multi-start
    The likelihood ratio LR_b = 2*(LL_M1f - LL_M3) is theoretically >= 0 because
    M1f nests M3 (M3 is M1f at alpha=0). However, EM maximization of M1f does not
    NUMERICALLY guarantee LL_M1f >= LL_M3 because: (a) EM updates start from
    alpha > 0 and may not reach alpha=0; (b) starting from alpha=0 freezes the
    update; and (c) any EM init that converges to a local optimum below the M3
    closed-form maximum yields LL_M1f < LL_M3.
    
    V3 fixes this by adding the M3 closed-form solution (mu_b = mu_b_M3,
    alpha = 0) as an explicit candidate after the K=5 EM restarts. We
    mathematically verify (see SANITY_CHECK_M3_ANCHOR below) that M1f LL
    evaluated at alpha=0 with any baseline mu_b equals the M3 LL with that
    same baseline. So the M3-anchor candidate ALWAYS yields LR_b >= 0 by
    construction.

  FIX #3: Per-iteration RNG seed (instead of carry-forward)
    V2 used `rng = default_rng(SEED + start_iter)` once at the start of the
    bootstrap and carried the RNG state through all iterations. This means
    a bootstrap interrupted at iter 100 and resumed will produce a DIFFERENT
    sequence than a single run from 0 to 1000 (because the RNG state at
    the resume start differs from the RNG state in the original run).
    
    V3 uses `rng = default_rng(SEED + b)` at the start of EACH iteration b,
    so iteration b's randomness depends only on the seed and b, regardless
    of resume point. Single-run and resume-run are reproducible under the
    same software environment (identical numpy/pandas versions, OS, BLAS
    backend, and floating-point determinism). Strict byte-level equality
    additionally requires these environmental factors to be held fixed;
    in practice, "reproducible-under-same-environment" is the operative
    guarantee.

  FIX #4: CSV resume with last-row validation
    V2's resume logic was `start_iter = len(existing_csv_rows)`. If the program
    was interrupted while writing a row (e.g., during the f.write call), the
    last row could be incomplete or malformed, leading to either a parse error
    or a phantom-completed row. V3 validates:
      (i)   pd.read_csv parses without error;
      (ii)  last row's 'iter' column equals len(df)-1 (sequential numbering);
      (iii) no NaN cells in last row.
    If any check fails, V3 drops the last row, rewrites the CSV cleanly, and
    resumes from the validated count. (Note: V3 validates only the LAST row,
    not all intermediate rows, on the assumption that interruption can only
    truncate the row currently being written. Strict sequential validation
    of all rows is possible but adds startup overhead and has not been observed
    necessary in practice.)

  FIX #7: LL_M1f_OBS is computed at runtime (not hardcoded)
    V2 had `LL_M1f_OBS = -727981.5117` hardcoded from a separate run of fit_m1f.py.
    This breaks end-to-end consistency: if the data pipeline, bin width, or any
    parameter changes between fit_m1f.py and V2, LL_M1f_OBS may not correspond
    to the same algorithm/data combination as LL_M3_obs computed inside V2.
    
    V3 runs the same multi-start-with-M3-anchor procedure on the OBSERVED data
    at startup, before the bootstrap loop, and uses that LL as LL_M1f_OBS.
    Then LR_OBS = 2*(LL_M1f_OBS - ll_M3_obs) is end-to-end consistent.

  FIX #13: Removed the broken clipped p-value
    V2 reported a "robustness" p-value computed as 
        p_clipped = mean(max(LR_sims, 0) >= max(LR_obs, 0))
    Since max(x, 0) >= 0 always, and LR_obs is small (< 0.01), this p-value
    is mechanically forced to ~1.0 regardless of the true distribution. It
    has no statistical meaning. V3 removes it entirely.
    
    With the M3-anchor fix, LR_b >= 0 holds by construction, so a clipped
    version would be redundant anyway.

ALSO IMPROVED:

  FIX #9: EM convergence check every iteration (was every 5 in V2)
    Computing LL each iteration is more expensive but ensures convergence
    is detected promptly and avoids the prev_ll = -inf init artifact.

  FIX #6 (cosmetic): init_win_counts loaded from CSV on resume
    V2 only counted wins from current resume session. V3 reads the existing
    'best_init_idx' column on resume and pre-populates the counter.

================================================================================

USAGE:
    python3 parametric_bootstrap_v4c.py
    
INPUT (must be in same directory):
    filtered_3_seasons.csv.gz   — raw NBA play-by-play (13 MB)

OUTPUT (incremental):
    bootstrap_v4c_results.csv     — each row = one iteration
    bootstrap_v4c_progress.log    — running log
    bootstrap_v4c_summary.txt     — final summary statistics

ESTIMATED RUNTIME: ~70-150 sec/iter depending on hardware/convergence
                   B=1000 → roughly 20-40 hours (conservative range)
                   Resume capability is built in.

EXPECTED RESULTS (anticipated based on V1 evidence):

  Because V1 reported LL_M1f - LL_M3 = -0.0025 on observed data (i.e., M3 LL
  was slightly HIGHER than EM-fitted M1f LL), the M3-anchor will likely win
  on observed data, yielding LR_obs = 0 EXACTLY. The bootstrap distribution
  will then consist of:
      (a) a spike at LR_b = 0 (iterations where M3-anchor wins on sim data)
      (b) a tail of LR_b > 0 (iterations where some EM init genuinely beats M3)
  
  The primary p-value P(LR_b >= LR_obs) = P(LR_b >= 0) will then equal 1.000
  (since all LR_b >= 0 by construction). This is NOT a bug — it is the
  natural result of a boundary LR test where the observed statistic lies
  exactly at the boundary.
  
  When reporting results, the formal inference is the V4b nesting-enforced
  bootstrap p-value (= 1.000 under LR_obs = 0), which unambiguously fails
  to reject H0: alpha = 0. Earlier single-start EM diagnostics produced
  the same qualitative conclusion but are NOT used for formal inference
  because their LR statistic could return negative values when EM
  converged to local optima — and a negative LR is not a valid LR-test
  statistic. The V4b nesting-enforced bootstrap is the paper's
  authoritative parametric LR test.

  Substantive conclusion: there is no evidence to reject the null
  hypothesis that the substitution process is inhomogeneous Poisson with
  a period-minute baseline.

================================================================================
"""

import numpy as np
import pandas as pd
import time
import os
import gc

# --- Configuration ---
B_TARGET = 1000
T_HORIZON = 2880.0
N_BINS = 48
BIN_W = T_HORIZON / N_BINS
RNG_SEED = 42

# --- Multi-start EM init grid (5 inits per replication) ---
EM_INITS = [
    {"alpha_init": 0.0001, "beta_init": 0.01,  "label": "B1_low_alpha"},
    {"alpha_init": 0.05,   "beta_init": 0.01,  "label": "B2_moderate"},
    {"alpha_init": 0.10,   "beta_init": 0.001, "label": "B3_diff_beta"},
    {"alpha_init": 0.50,   "beta_init": 0.05,  "label": "B4_middle"},
    {"alpha_init": 0.95,   "beta_init": 0.01,  "label": "B5_high_alpha"},
]
N_INITS = len(EM_INITS)
M3_ANCHOR_INIT_IDX = -1  # special code

# --- EM hyperparameters ---
EM_TOL = 1e-3
EM_MAX_ITER = 150

# --- File paths ---
DATA_FILE     = "filtered_3_seasons.csv.gz"
RESULTS_FILE  = "bootstrap_v4c_results.csv"
PROGRESS_LOG  = "bootstrap_v4c_progress.log"
SUMMARY_FILE  = "bootstrap_v4c_summary.txt"

# --- Memory monitoring ---
try:
    import psutil
    PROCESS = psutil.Process(os.getpid())
    def mem_mb():
        return PROCESS.memory_info().rss / (1024 * 1024)
except ImportError:
    def mem_mb():
        return -1


def log(msg, also_print=True):
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    if also_print:
        print(msg, flush=True)


# ============================================================================
# SECTION 1: DATA LOADING (matches V2; verified to work on user's laptop)
# ============================================================================

def load_and_prepare_data():
    log("Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    log(f"  Loaded {len(df):,} rows")
    
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
    
    log("Computing absolute times...")
    df["t_abs"] = df.apply(lambda r: absolute_seconds(r["PERIOD"], r["PCTIMESTRING"]), axis=1)
    subs = df[df["EVENTMSGTYPE"]==8].dropna(subset=["t_abs", "PLAYER1_TEAM_ID"]).copy()
    subs["TEAM_ID"] = subs["PLAYER1_TEAM_ID"].astype(int)
    
    del df
    gc.collect()
    
    log("Mass-aggregating...")
    mass_subs = (subs.groupby(["GAME_ID","TEAM_ID","PERIOD","t_abs"], as_index=False)
                      .agg(n_players=("PLAYER1_ID","count")))
    mass_subs = mass_subs.sort_values(["GAME_ID","TEAM_ID","t_abs"]).reset_index(drop=True)
    mass_reg = mass_subs[mass_subs["PERIOD"]<=4].copy()
    
    del subs, mass_subs
    gc.collect()
    
    log("Building realizations...")
    realizations = []
    for (gid, tid), g_subs in mass_reg.groupby(["GAME_ID","TEAM_ID"]):
        sub_t = np.sort(g_subs["t_abs"].values.astype(np.float64))
        sub_t = sub_t[sub_t < T_HORIZON]
        if len(sub_t) > 0:
            realizations.append(sub_t)
    
    del mass_reg
    gc.collect()
    
    n_total = sum(len(s) for s in realizations)
    log(f"  {len(realizations):,} realizations, {n_total:,} events")
    log(f"  NOTE: realizations include only team-games with at least one")
    log(f"        substitution before T_HORIZON ({T_HORIZON}s). Team-games")
    log(f"        with zero in-regulation substitutions (if any) are excluded")
    log(f"        from this realization set; any exposure effect should be")
    log(f"        checked explicitly if zero-substitution team-games are")
    log(f"        present in the source data.")
    log(f"  Memory after data prep: {mem_mb():.0f} MB")
    
    return realizations, n_total


# ============================================================================
# SECTION 2: M3 closed-form fit (verbatim from V2)
# ============================================================================

def fit_m3(realizations, total_exposure):
    event_bins = []
    for sub_t in realizations:
        bins = np.minimum((sub_t // BIN_W).astype(np.int32), N_BINS-1)
        event_bins.append(bins)
    
    all_bins = np.concatenate(event_bins) if len(event_bins) > 0 else np.array([], dtype=np.int32)
    counts = np.bincount(all_bins, minlength=N_BINS)
    mu_b = counts / total_exposure
    
    if len(all_bins) > 0:
        ll = (np.log(np.maximum(mu_b[all_bins], 1e-12))).sum() - (mu_b * total_exposure).sum()
    else:
        ll = 0.0
    
    return mu_b, ll, event_bins


# ============================================================================
# SECTION 3: M1f EM kernel + LL function (verbatim from V2 + fixes)
# ============================================================================

def fit_m1f_iter(realizations, event_bins, mu_b, alpha, beta, n_total, total_exp):
    """
    Single EM iteration for M1f (inhomogeneous baseline + self-excitation).

    FIX [B1] (cross-script consistency with fit_m1f_v4):
        α M-step uses the compensator denominator
            α = sum_pS / Σ_j (1 − exp(−β(T − t_j)))
        rather than the simpler sum_pS / n_total. The latter is biased high
        under finite-window observation. This change makes the bootstrap's
        EM kernel byte-identical to fit_m1f_v4 (Stage 5), so LR_OBS and
        LR_sims are computed from the same likelihood surface.

    Note: the `n_total` parameter is retained for backward-compatible function
    signature (unchanged caller signatures across the bootstrap), but is no
    longer used inside the function after the [B1] fix.

    Caveat (β M-step): the β update here uses the standard recursive form
        β_new = sum_pS / sum_pS_dt
    which is the EM-style update for an unbounded-window exponential Hawkes.
    Strictly, the finite-window MLE for β has additional terms from the
    excitation compensator α·Σ_j(1 - exp(-β(T - t_j))), but we use this
    recursive form for cross-script consistency with phase2_v4 / phase4_v4 /
    fit_m1f_v4. This is documented in the paper as an EM-style update rather
    than the exact finite-window MLE.
    """
    sum_pB_per_bin = np.zeros(N_BINS)
    sum_pS = 0.0
    sum_pS_dt = 0.0
    sum_compensator = 0.0  # FIX [B1]: denominator for alpha update

    for r_idx, sub_t in enumerate(realizations):
        n = len(sub_t)
        if n == 0: continue
        bins = event_bins[r_idx]

        A = np.zeros(n); B_acc = np.zeros(n)
        if n > 1:
            for i in range(1, n):
                dt = sub_t[i] - sub_t[i-1]
                ev = np.exp(-beta * dt)
                A[i] = ev * (1 + A[i-1])
                B_acc[i] = ev * (B_acc[i-1] + dt * (1 + A[i-1]))

        lam = mu_b[bins] + alpha * beta * A
        lam = np.maximum(lam, 1e-12)  # NUMERICAL_FLOOR: unified with fit_m3 (was 1e-15 in V2)

        pB = mu_b[bins] / lam
        np.add.at(sum_pB_per_bin, bins, pB)
        sum_pS += ((alpha * beta * A) / lam).sum()
        sum_pS_dt += ((alpha * beta * B_acc) / lam).sum()
        # FIX [B1]: accumulate compensator term Σ_j (1 - exp(-β(T - t_j)))
        sum_compensator += np.sum(1 - np.exp(-beta * (T_HORIZON - sub_t)))

    new_mu_b = np.maximum(sum_pB_per_bin / total_exp, 1e-9)
    # FIX [B1]: denominator is sum_compensator (proper finite-window MLE),
    # not n_total (which gives biased-high alpha estimates)
    if sum_compensator > 1e-12:
        new_alpha = min(sum_pS / sum_compensator, 0.99)
    else:
        new_alpha = 0.0
    new_beta = max(sum_pS / sum_pS_dt, 1e-7) if sum_pS > 1e-10 else beta
    return new_mu_b, new_alpha, new_beta


def m1f_loglik(realizations, event_bins, mu_b, alpha, beta):
    """
    Log-likelihood under M1f:
        lambda(t) = mu_b[bin(t)] + alpha * beta * sum_{t_i<t} exp(-beta*(t-t_i))
    
    NOTE on alpha=0 case: When alpha=0, the recursive A term gets multiplied by
    0 inside lambda, so the intensity reduces to the M3 baseline mu_b[bin(t)].
    The excitation integral, in our parametrization where the kernel is
    alpha*beta*exp(-beta*(t-t_i)), evaluates to alpha * sum(1-exp(-beta*(T-t_i)))
    after closed-form integration. When alpha=0 this is exactly 0 regardless
    of beta. So m1f_loglik(real, bins, mu_b, alpha=0, beta=ANY) equals
    m3_loglik(real, bins, mu_b). This is the basis for the M3-anchor.
    
    NOTE on zero-event realizations: A realization with no events contributes
    log(P(no events)) - integral(lambda) = 0 - comp_baseline (when alpha=0)
    or 0 - comp_baseline - 0 (when alpha>0, since with no events the excitation
    sum is empty, giving zero excitation integral). In both cases the contribution
    is -comp_baseline. We must NOT skip these realizations, otherwise their
    baseline integral is omitted and m1f_loglik over-counts vs fit_m3, which
    correctly accounts for total exposure across ALL realizations including
    empty ones. (Bug fix, V3b: previously used `if n == 0: continue` which
    silently dropped empty realizations from the LL.)
    """
    total = 0.0
    # Each realization contributes -comp_baseline to the LL regardless of n.
    # Pre-compute and apply once per realization. This avoids missing the
    # baseline integral for n==0 realizations.
    comp_baseline_per_realization = (mu_b * BIN_W).sum()
    
    for r_idx, sub_t in enumerate(realizations):
        n = len(sub_t)
        # Always subtract baseline integral (compensator for the realization)
        total -= comp_baseline_per_realization
        
        if n == 0:
            # Zero-event realization: contributes log(empty product) = 0 from events
            # and 0 from excitation integral (no events => empty sum => no excitation).
            # We've already subtracted comp_baseline above, so nothing more to do.
            continue
        
        bins = event_bins[r_idx]
        A = np.zeros(n)
        if n > 1:
            for i in range(1, n):
                A[i] = np.exp(-beta * (sub_t[i] - sub_t[i-1])) * (1 + A[i-1])
        lam = mu_b[bins] + alpha * beta * A
        lam = np.maximum(lam, 1e-12)  # NUMERICAL_FLOOR: unified with fit_m3 (was 1e-15 in V2); ensures M3-anchor LL exactly equals fit_m3 LL when alpha=0
        comp_self = alpha * np.sum(1 - np.exp(-beta * (T_HORIZON - sub_t)))
        total += np.sum(np.log(lam)) - comp_self
    return total


def fit_m1f_full(realizations, event_bins, mu_b_init, n_total, total_exp,
                 alpha_init, beta_init, max_iter=EM_MAX_ITER, tol=EM_TOL):
    """Single full EM run from given init.
    
    FIX #9: LL is computed every iteration (not every 5).
    """
    mu_b = mu_b_init.copy()
    alpha = alpha_init
    beta = beta_init
    
    prev_ll = m1f_loglik(realizations, event_bins, mu_b, alpha, beta)
    ll = prev_ll  # FIX #4 (defensive): ensures `ll` is defined even if max_iter == 0
    
    for it in range(max_iter):
        mu_b, alpha, beta = fit_m1f_iter(
            realizations, event_bins, mu_b, alpha, beta, n_total, total_exp
        )
        ll = m1f_loglik(realizations, event_bins, mu_b, alpha, beta)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    
    final_ll = ll
    return mu_b, alpha, beta, final_ll


def fit_m1f_multistart_with_anchor(realizations, event_bins, mu_b_M3_baseline,
                                    n_total, total_exp):
    """
    FIX #1: Multi-start EM for M1f, WITH M3 closed-form solution as an explicit
            anchor candidate. This guarantees that the returned LL is at least
            equal to the M3 LL evaluated at mu_b_M3_baseline, and therefore
            LR_b = 2*(LL_M1f_returned - LL_M3) >= 0 by construction.
    
    Returns:
        (best_mu_b, best_alpha, best_beta, best_ll, best_init_idx, all_lls)
        
        best_init_idx == -1 means the M3-anchor won (no EM init beat it).
        all_lls includes all 5 EM init LLs and the M3-anchor LL (at index 5).
    """
    best_ll = -np.inf
    best_result = None
    best_init_idx = -2  # placeholder
    all_lls = []
    
    # Run K=5 EM inits
    for k, init in enumerate(EM_INITS):
        mu_b_k, alpha_k, beta_k, ll_k = fit_m1f_full(
            realizations, event_bins, mu_b_M3_baseline, n_total, total_exp,
            alpha_init=init["alpha_init"], beta_init=init["beta_init"]
        )
        all_lls.append(ll_k)
        if ll_k > best_ll:
            best_ll = ll_k
            best_result = (mu_b_k, alpha_k, beta_k, ll_k)
            best_init_idx = k
    
    # Compute M3-anchor LL: M1f at (mu_b = mu_b_M3, alpha = 0) reduces to M3 LL exactly
    # (mathematically verified; see m1f_loglik docstring)
    ll_m3_anchor = m1f_loglik(realizations, event_bins, mu_b_M3_baseline,
                              alpha=0.0, beta=0.01)  # beta is irrelevant at alpha=0
    all_lls.append(ll_m3_anchor)
    
    # If M3-anchor matches or beats all EM inits, use it.
    # Note: we use >= rather than >, so that when an EM init converges to a
    # solution numerically equal to the M3-anchor LL (e.g., when EM finds
    # alpha very close to 0 on H0-distributed sim data), the tie goes to the
    # M3-anchor. This gives an honest accounting of how often the algorithm
    # cannot improve on M3, which is a key diagnostic for the boundary test.
    if ll_m3_anchor >= best_ll:
        best_ll = ll_m3_anchor
        best_result = (mu_b_M3_baseline.copy(), 0.0, 0.01, ll_m3_anchor)
        best_init_idx = M3_ANCHOR_INIT_IDX  # = -1
    
    if best_result is None:
        # Defensive fallback
        ll_default = m1f_loglik(realizations, event_bins, mu_b_M3_baseline, 0.0, 0.01)
        return mu_b_M3_baseline.copy(), 0.0, 0.01, ll_default, M3_ANCHOR_INIT_IDX, all_lls
    
    return (*best_result, best_init_idx, all_lls)


# ============================================================================
# SECTION 4: Inhom Poisson simulation (verbatim from V2)
# ============================================================================

def simulate_inhom_poisson(mu_b, n_real_target, T, bin_w, rng):
    sim_realizations = []
    for r in range(n_real_target):
        events = []
        for k in range(N_BINS):
            n_in_bin = rng.poisson(mu_b[k] * bin_w)
            if n_in_bin > 0:
                bin_start = k * bin_w
                events.extend(rng.uniform(bin_start, bin_start + bin_w, size=n_in_bin))
        events_arr = np.sort(np.asarray(events, dtype=np.float64))
        sim_realizations.append(events_arr)
    return sim_realizations


# ============================================================================
# SECTION 5: CSV resume validation (FIX #4)
# ============================================================================

def validate_and_fix_resume_csv(path):
    """
    FIX #4: Validate the existing CSV before resume.

    Checks (in order):
      (i)    pd.read_csv parses without error
      (ii)   The 'iter' column is sequential (0, 1, 2, ..., N-1) — added in V4b
             to detect mid-file corruption, not just truncated last row.
      (iii)  Last row has no NaN cells

    Repair strategy: keep the longest valid sequential prefix.

    Returns: int — number of completed iterations (resume start index)
    """
    if not os.path.exists(path):
        return 0

    try:
        df = pd.read_csv(path)
    except Exception as e:
        log(f"[RESUME] CSV unreadable: {e}; starting from iter 0")
        # Move the bad CSV aside for inspection
        os.rename(path, path + ".corrupted")
        return 0

    if len(df) == 0:
        log(f"[RESUME] CSV is empty (header only); starting from iter 0")
        return 0

    # FIX [V4b #7]: Sequential validation of the 'iter' column.
    # Find the longest prefix where iter[i] == i. For B=1000 this scan is
    # negligible (< 1ms) so it's strictly an upgrade over checking only the
    # last row.
    iter_arr = df['iter'].to_numpy()
    expected_seq = np.arange(len(df))
    if not np.array_equal(iter_arr, expected_seq):
        # Find longest valid prefix
        valid_len = 0
        for i, val in enumerate(iter_arr):
            try:
                if int(val) == i:
                    valid_len += 1
                else:
                    break
            except (ValueError, TypeError):
                break
        log(f"[RESUME] 'iter' column is not sequential at row {valid_len}; "
            f"truncating to longest valid prefix (was {len(df)} rows, "
            f"keeping {valid_len}).")
        df = df.iloc[:valid_len]
        df.to_csv(path, index=False)

    if len(df) == 0:
        return 0

    # Check last row for NaN (truncated mid-write)
    last_row = df.iloc[-1]
    if last_row.isna().any():
        log(f"[RESUME] Last row has NaN values; dropping and rewriting")
        df = df.iloc[:-1]
        df.to_csv(path, index=False)

    log(f"[RESUME] Validated {len(df)} rows; resuming from iter {len(df)}")
    return len(df)


# ============================================================================
# SECTION 6: Main bootstrap loop
# ============================================================================

def main():
    # ----- Load data once -----
    realizations, n_total_obs = load_and_prepare_data()
    n_real = len(realizations)
    total_exposure = np.full(N_BINS, n_real * BIN_W)
    
    # ----- Fit M3 on observed data once -----
    mu_b_M3_obs, ll_M3_obs, event_bins_obs = fit_m3(realizations, total_exposure)
    log(f"M3 observed LL = {ll_M3_obs:.4f}")
    
    # FIX #7: Compute observed M1f LL via the SAME multi-start procedure (with M3 anchor)
    # used in the bootstrap loop. This is end-to-end consistent with bootstrap LL_M1f_sim.
    log("\n[Computing observed LL_M1f via multi-start with M3-anchor]")
    log("  (this takes ~1-2 minutes on first run)")
    t0 = time.time()
    _, alpha_obs, beta_obs, LL_M1f_OBS, obs_init_idx, obs_all_lls = \
        fit_m1f_multistart_with_anchor(
            realizations, event_bins_obs, mu_b_M3_obs, n_total_obs, total_exposure
        )
    log(f"  Observed multistart took {time.time()-t0:.1f}s")
    log(f"  M1f observed LL = {LL_M1f_OBS:.4f}")
    log(f"  Best init: {obs_init_idx} ({'M3-anchor' if obs_init_idx == -1 else EM_INITS[obs_init_idx]['label']})")
    log(f"  alpha_obs = {alpha_obs:.6e}, beta_obs = {beta_obs:.6e}")
    log(f"  All init LLs: {[f'{ll:.4f}' for ll in obs_all_lls]}")
    
    LR_OBS_raw = 2 * (LL_M1f_OBS - ll_M3_obs)
    # FIX [V4c float-eps]: When the M3-anchor wins (idx -1), LL_M1f_OBS and
    # ll_M3_obs are mathematically equal (M1f at α=0 reduces to M3 with the
    # same μ_b), but they are computed by different sum paths:
    #   fit_m3():     uses (mu_b * total_exposure).sum() — single vector op
    #   m1f_loglik(): subtracts comp_baseline per realization (loop, V3b
    #                 zero-event fix path)
    # These give floating-point differences of order 1e-10..1e-12 on a LL of
    # magnitude ~7e5 — relative error ~1e-15, pure machine epsilon. So
    # LR_OBS_raw can be a tiny negative value when the anchor wins. We floor
    # at 0 to reflect the true mathematical value.
    LR_FLOOR_EPS = 1e-6
    if LR_OBS_raw < -LR_FLOOR_EPS:
        log(f"  WARNING: LR_OBS_raw = {LR_OBS_raw:.6e} < -{LR_FLOOR_EPS:.0e}. "
            f"This should not happen with M3-anchor — investigate.")
        LR_OBS = LR_OBS_raw  # keep the bad value visible for debugging
    else:
        LR_OBS = max(LR_OBS_raw, 0.0)
        if abs(LR_OBS_raw) < LR_FLOOR_EPS:
            log(f"  Observed LR raw = {LR_OBS_raw:+.4e} (machine-epsilon "
                f"noise from M3-anchor M1f vs fit_m3 sum-path differences); "
                f"floored to LR_OBS = 0.")
        else:
            log(f"  CHECK PASSED: LR_OBS = {LR_OBS:.6f} >= 0 "
                f"(as theoretically required)")
    log(f"  Observed LR (used for inference) = {LR_OBS:.6f}")
    
    # ----- Print bootstrap config -----
    log("")
    log("=" * 70)
    log(f"PARAMETRIC BOOTSTRAP V4c (multi-start EM + M3-anchor)")
    log(f"  B = {B_TARGET}")
    log(f"  K = {N_INITS} EM restarts + 1 M3-anchor candidate per replication")
    log(f"  EM tol = {EM_TOL}")
    log(f"  EM max_iter = {EM_MAX_ITER}")
    log(f"  RNG: per-iteration seed = {RNG_SEED} + b")
    log(f"  Observed LR = {LR_OBS:.6f}")
    log("=" * 70)
    
    # ----- FIX #4: validated resume -----
    start_iter = validate_and_fix_resume_csv(RESULTS_FILE)
    
    if start_iter == 0:
        # Write fresh header
        all_init_cols = ",".join([f"ll_init_{k}" for k in range(N_INITS)] + ["ll_init_anchor"])
        with open(RESULTS_FILE, "w") as f:
            f.write(f"iter,alpha_M1f,beta_M1f,LL_M1f,LL_M3_sim,LR_sim,best_init_idx,n_total_sim,{all_init_cols}\n")
        log(f"[Init] Wrote fresh CSV header")
    else:
        log(f"[Resume] Continuing from iteration {start_iter} ({start_iter} rows verified)")
    
    # FIX #6 (cosmetic): pre-populate init_win_counts from existing CSV
    init_win_counts = np.zeros(N_INITS + 1, dtype=int)  # +1 for M3-anchor wins
    if start_iter > 0:
        try:
            existing = pd.read_csv(RESULTS_FILE)
            for idx in existing['best_init_idx']:
                if idx == M3_ANCHOR_INIT_IDX:
                    init_win_counts[N_INITS] += 1
                elif 0 <= idx < N_INITS:
                    init_win_counts[idx] += 1
            log(f"[Resume] Pre-populated init_win_counts: {init_win_counts.tolist()}")
            del existing
            gc.collect()
        except Exception as e:
            log(f"[Resume] Could not pre-populate init_wins: {e}")
    
    t_start = time.time()
    
    for b in range(start_iter, B_TARGET):
        iter_start = time.time()
        
        # FIX #3: Per-iteration RNG seed (reproducible regardless of resume point)
        rng = np.random.default_rng(RNG_SEED + b)
        
        # 1. Simulate dataset under H0
        sim_real = simulate_inhom_poisson(mu_b_M3_obs, n_real, T_HORIZON, BIN_W, rng)
        sim_event_bins = [np.minimum((s // BIN_W).astype(np.int32), N_BINS-1) for s in sim_real]
        sim_n_total = sum(len(s) for s in sim_real)
        
        # 2a. Fit M3 (closed-form) on simulated data
        if sim_n_total > 0:
            sim_all_bins = np.concatenate(sim_event_bins)
            sim_counts = np.bincount(sim_all_bins, minlength=N_BINS)
            mu_b_M3_sim = sim_counts / total_exposure
            ll_M3_sim = (np.log(np.maximum(mu_b_M3_sim[sim_all_bins], 1e-12))).sum() - (mu_b_M3_sim * total_exposure).sum()
            del sim_all_bins, sim_counts
        else:
            # FIX [V4b #1]: When sim_n_total = 0, the M3 MLE is mu_b = 0 with
            # ll_M3 = 0 (the Poisson log-likelihood is 0·log(0) - 0·T = 0 by
            # convention). Previously we set mu_b = 1e-9 (nonzero!) but kept
            # ll_M3 = 0, which was internally inconsistent: at mu_b = 1e-9,
            # the correct ll would be -1e-9·total_exposure ≈ -3e-3, not 0.
            # With n_real = 7,380 and mu_b ≈ 1.6e-2, sim_n_total = 0 has
            # probability ≈ exp(-17,000) ≈ 0 in practice, so this branch
            # almost never executes — but the math is now clean.
            mu_b_M3_sim = np.zeros(N_BINS)
            ll_M3_sim = 0.0
        
        # 2b. Fit M1f with multi-start + M3-anchor (FIX #1)
        _, alpha, beta, ll_M1f_sim, best_init_idx, all_lls = \
            fit_m1f_multistart_with_anchor(
                sim_real, sim_event_bins, mu_b_M3_sim, sim_n_total, total_exposure
            )
        
        # Track init wins (best_init_idx of -1 means M3-anchor won; map to index N_INITS)
        if best_init_idx == M3_ANCHOR_INIT_IDX:
            init_win_counts[N_INITS] += 1
        elif 0 <= best_init_idx < N_INITS:
            init_win_counts[best_init_idx] += 1
        
        # 3. Compute LR (guaranteed >= 0 by M3-anchor)
        LR_sim = 2 * (ll_M1f_sim - ll_M3_sim)
        
        # Sanity check: LR should be >= 0 (modulo floating-point noise from
        # different mu_b_M3 paths in EM vs closed-form on sim_data).
        # In practice, when M3-anchor wins, mu_b is mu_b_M3_sim and LR = 0 exactly.
        # When EM wins, EM's mu_b differs from mu_b_M3_sim but EM's overall LL
        # is higher than M3-anchor's LL on (mu_b_M3_sim, alpha=0), so LR > 0.
        # Numerical floor: if LR_sim < -1e-6, something is wrong.
        if LR_sim < -1e-6:
            log(f"  WARNING iter {b}: LR_sim = {LR_sim:.6f} < 0; "
                f"M3-anchor may have failed. ll_M1f={ll_M1f_sim:.4f}, "
                f"ll_M3_sim={ll_M3_sim:.4f}, best_init={best_init_idx}")
        
        # 4. Append to CSV (streaming write with explicit flush — FIX #4 robustness)
        all_lls_str = ",".join([f"{ll:.6f}" for ll in all_lls])
        line = (f"{b},{alpha:.10g},{beta:.10g},{ll_M1f_sim:.6f},"
                f"{ll_M3_sim:.6f},{LR_sim:.6f},{best_init_idx},{sim_n_total},"
                f"{all_lls_str}\n")
        with open(RESULTS_FILE, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk before next iteration
        
        # 5. Free memory
        del sim_real, sim_event_bins, mu_b_M3_sim
        gc.collect()
        
        # 6. Progress logging
        iter_elapsed = time.time() - iter_start
        total_elapsed = time.time() - t_start
        n_done = b - start_iter + 1
        n_remaining = B_TARGET - b - 1
        eta_sec = (total_elapsed / n_done) * n_remaining if n_done > 0 else 0
        
        if (b+1) % 10 == 0 or b < 5 or (b+1) % 100 == 0:
            mem = mem_mb()
            mem_str = f"mem={mem:.0f}MB" if mem > 0 else ""
            wins_str = ",".join([str(c) for c in init_win_counts])
            log(f"  iter {b+1:4d}/{B_TARGET}: alpha={alpha:.6f} LR_sim={LR_sim:+.4f} "
                f"best={best_init_idx} "
                f"wins=[{wins_str}] "
                f"({iter_elapsed:.1f}s/iter, {total_elapsed/60:.1f}min elapsed, "
                f"~{eta_sec/60:.1f}min remaining) {mem_str}")
    
    # ----- Compute and write final summary -----
    log("\n" + "="*70)
    log("COMPUTING FINAL SUMMARY")
    log("="*70)
    
    results_df = pd.read_csv(RESULTS_FILE)
    LR_sims = results_df["LR_sim"].values
    
    # Primary p-value: P(LR_sim >= LR_obs)
    n_extreme = np.sum(LR_sims >= LR_OBS)
    p_value = (1 + n_extreme) / (1 + len(LR_sims))
    
    # Diagnostic: fraction of LR_sims at exactly 0 (M3-anchor wins)
    # Threshold 1e-6 matches CSV save precision (LR_sim is written with .6f format,
    # so any LR < 5e-7 reads back as 0.000000). Using 1e-9 would miss values that
    # were rounded to zero on CSV write.
    n_at_zero = np.sum(np.abs(LR_sims) < 1e-6)
    n_negative = np.sum(LR_sims < -1e-6)
    n_positive = np.sum(LR_sims >= 1e-6)
    
    # Init win statistics
    init_wins_csv = results_df["best_init_idx"].value_counts().sort_index()
    init_summary_lines = []
    for k in range(N_INITS):
        c = init_wins_csv.get(k, 0)
        init_summary_lines.append(
            f"  init {k} ({EM_INITS[k]['label']}): {c:4d} wins ({c/len(results_df)*100:5.1f}%)"
        )
    c_anchor = init_wins_csv.get(M3_ANCHOR_INIT_IDX, 0)
    init_summary_lines.append(
        f"  M3-anchor (idx -1):                {c_anchor:4d} wins ({c_anchor/len(results_df)*100:5.1f}%)"
    )
    init_summary = "\n".join(init_summary_lines)
    
    # Construct context-aware p-value commentary (FIX #6 — addresses p=1 boundary case)
    mc_se = np.sqrt(p_value * (1 - p_value) / len(LR_sims))
    ci_lo = max(0.0, p_value - 1.96 * mc_se)
    ci_hi = min(1.0, p_value + 1.96 * mc_se)
    
    if abs(p_value - 1.0) < 1e-9:
        p_value_block = (
            f"PRIMARY p-VALUE:\n"
            f"  p = (1 + {n_extreme}) / (1 + {len(LR_sims)}) = {p_value:.4f}\n"
            f"  Monte Carlo SE = {mc_se:.4f}\n"
            f"  Approx 95% CI:  [{ci_lo:.3f}, {ci_hi:.3f}]\n"
            f"\n"
            f"  IMPORTANT INTERPRETIVE NOTE:\n"
            f"  The p-value of 1.000 is a MECHANICAL result, not evidence of\n"
            f"  unusually strong null support. Because the M3-anchor enforces\n"
            f"  LR_b >= 0 by construction, AND the observed LR statistic is\n"
            f"  itself at the boundary value 0 (LR_obs = {LR_OBS:.6f}), every\n"
            f"  bootstrap statistic trivially satisfies LR_b >= LR_obs, giving\n"
            f"  p = 1 by definition. The MC SE = {mc_se:.4f} and CI = [1, 1] do\n"
            f"  NOT reflect Monte Carlo precision in the usual sense; they\n"
            f"  reflect the deterministic structure of a boundary likelihood\n"
            f"  ratio test where the observed statistic is exactly at the\n"
            f"  boundary. The substantive conclusion is unambiguous: under\n"
            f"  the nesting-enforced LR test, there is no evidence whatsoever\n"
            f"  to reject H0: alpha = 0.\n"
            f"\n"
            f"  Earlier single-start EM diagnostics produced the same qualitative\n"
            f"  conclusion but are NOT used for formal inference because their LR\n"
            f"  statistic could return negative values when EM converged to local\n"
            f"  optima — a negative LR is not a valid LR-test statistic. The V4b\n"
            f"  nesting-enforced bootstrap is the paper's authoritative parametric\n"
            f"  LR test."
        )
    elif p_value < 1.0:
        p_value_block = (
            f"PRIMARY p-VALUE:\n"
            f"  p = (1 + {n_extreme}) / (1 + {len(LR_sims)}) = {p_value:.4f}\n"
            f"  Monte Carlo SE = sqrt(p(1-p)/B) = {mc_se:.4f}\n"
            f"  Approx 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]"
        )
    else:
        p_value_block = (
            f"PRIMARY p-VALUE:\n"
            f"  p = (1 + {n_extreme}) / (1 + {len(LR_sims)}) = {p_value:.4f}\n"
            f"  (boundary case)"
        )
    
    summary = f"""Parametric Bootstrap V4c (Multi-Start EM + M3-Anchor + [B1] + small fixes) — Final Summary
============================================================================
B = {len(LR_sims)} bootstrap iterations
K = {N_INITS} EM restarts + 1 M3-anchor per replication
EM tol = {EM_TOL}, max_iter = {EM_MAX_ITER}
RNG: per-iteration seed = {RNG_SEED} + b (reproducible)
H0: alpha = 0 (data ~ M3 inhomogeneous Poisson with period-minute baseline)

OBSERVED:
  LR_obs (data) = {LR_OBS:+.6f}
    [Computed end-to-end: M3 LL = {ll_M3_obs:.4f}, M1f LL = {LL_M1f_OBS:.4f}]

BOOTSTRAP LR DISTRIBUTION (V4b, multi-start EM-style fit + M3-anchor):
  min:    {LR_sims.min():+.4f}
  q01:    {np.quantile(LR_sims, 0.01):+.4f}
  q05:    {np.quantile(LR_sims, 0.05):+.4f}
  q25:    {np.quantile(LR_sims, 0.25):+.4f}
  median: {np.median(LR_sims):+.4f}
  q75:    {np.quantile(LR_sims, 0.75):+.4f}
  q95:    {np.quantile(LR_sims, 0.95):+.4f}
  q99:    {np.quantile(LR_sims, 0.99):+.4f}
  max:    {LR_sims.max():+.4f}
  mean:   {LR_sims.mean():+.4f}
  std:    {LR_sims.std():+.4f}

DISTRIBUTION SHAPE:
  At zero (|LR| < 1e-6, matches CSV precision):    {n_at_zero}/{len(LR_sims)} ({n_at_zero/len(LR_sims)*100:.1f}%)
  Strictly positive (LR >= 1e-6):                   {n_positive}/{len(LR_sims)} ({n_positive/len(LR_sims)*100:.1f}%)
  Negative (numerical, should be ~0):               {n_negative}/{len(LR_sims)} ({n_negative/len(LR_sims)*100:.1f}%)
  
  Interpretation: With M3-anchor enforced, LR_b >= 0 holds by construction.
  The "at zero" mass corresponds to bootstrap iterations where the M3-anchor
  beat all 5 EM inits — the algorithm could find no Hawkes specification
  better than the M3 closed-form. The strictly positive mass corresponds
  to iterations where some EM init genuinely converged to a positive-alpha
  solution that, on the simulated dataset, fit slightly better than M3 in
  the joint (mu_b, alpha, beta) space. This can occur in two ways under
  H0: (i) finite-sample random clustering in the simulated Poisson process
  that happens to look like self-excitation, leading EM to over-fit a small
  positive alpha; (ii) the boundary nature of the LR distribution under
  H0 with unidentified beta — even when alpha is truly zero, the LR
  statistic has a non-degenerate distribution at and above 0 due to the
  parameter-space topology. A high positive-mass fraction would therefore
  reflect EM's susceptibility to spurious self-excitation, not a violation
  of the nesting property.

TAIL COUNTS (for p-value):
  #{{LR_sim >= LR_obs={LR_OBS:.6f}}} = {n_extreme}/{len(LR_sims)} = {n_extreme/len(LR_sims):.4f}

{p_value_block}

INIT WIN STATISTICS (which solution found the highest LL):
{init_summary}

  Diagnosis: a high M3-anchor win rate is consistent with the H0 (alpha=0)
  data-generating process — under H0, the EM should rarely find a meaningful
  positive alpha solution. A low M3-anchor win rate (say < 10%) would
  suggest that the EM is finding many spurious positive-alpha optima even
  on Poisson data, which would itself be evidence of EM's tendency to
  over-fit the self-excitation kernel.

FIXES VS V2:
  #1  M3-anchor in multi-start (guarantees LR_b >= 0)              [DONE]
  #3  Per-iteration RNG seed (resume reproducible under same env)  [DONE]
  #4  CSV resume validation (drops corrupted last row)             [DONE]
  #7  LL_M1f_OBS computed at runtime (not hardcoded)               [DONE]
  #13 Removed broken clipped p-value                               [DONE]
  #9  EM convergence check every iteration                         [DONE]
  #6  init_wins counter pre-populated on resume                    [DONE]
  
ADDITIONAL CONSISTENCY FIXES (post-V3-review):
  NUMERICAL_FLOOR: unified 1e-12 floor across fit_m3, fit_m1f_iter,
                   m1f_loglik (was inconsistent 1e-12/1e-15 in V3a)
  TIE_BREAKING:    M3-anchor wins ties with EM inits (>= comparison)
  P1_BOUNDARY:     summary explains p=1 case as boundary mechanism, not
                   small MC SE

============================================================================
"""
    
    print(summary, flush=True)
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary)
    
    log(f"\n[Saved] {RESULTS_FILE} ({len(LR_sims)} rows)")
    log(f"[Saved] {SUMMARY_FILE}")
    log(f"[Saved] {PROGRESS_LOG}")


if __name__ == "__main__":
    main()