"""
================================================================================
Script 07: Phase 5 R3 FIX — Marked Hawkes via EM + Profiled M-step (v3)
================================================================================

Purpose:
  Provide the canonical R3 robustness test: does augmenting the Hawkes
  process with a mark function over n_players (number of players in each
  mass-aggregated event) provide additional explanatory power?

  POSITIONING:
    This script is the PAPER §7 authoritative R3 test, superseding the
    L-BFGS-B-based R3 result in 06_phase5_fix_v4.py (which finds a
    degenerate slow-decay saddle, not the true MLE) and the marked
    Hawkes archival output in 05_phase5_robustness_v5.py (which used
    a known-buggy n_total denominator, producing spurious α = 0.42).

Marked Hawkes model:
  λ(t) = μ + α · β · Σ_{j: t_j < t} (m_j/m_avg)^γ · exp(-β·(t - t_j))

  m_j   = number of players in event j (1, 2, 3, 4, 5, or 6)
  m_avg = mean mark = 1.415 across full league

Hypotheses:
  H0: γ = 0  (no mark effect; reduces to standard self-only Hawkes)
  H1: γ free (mark may amplify or suppress excitation)

Test: Likelihood ratio, df = 1, χ² distribution under H0.
      (Caveat: at the α = 0 boundary the standard χ² approximation may not
       be exact; if α > threshold the LR test is more interpretable.)

Estimation: Full marked-Hawkes EM derivation:
  - E-step: 3-component responsibility assignment with mark weights w_j
  - M-step: μ closed-form. β uses the standard recursive update consistent
            with phase2_v4 / phase4_v4 / fit_m1f_v4 (not the exact finite-
            window MLE, but kept for cross-script comparability). γ is
            updated by bounded scalar optimization on the PROFILED Q-function
            Q_profile(γ) = γ·S_logm − S·log C(γ); α is then set to its
            closed-form profiled MLE α*(γ) = S / C(γ).
  - Multi-start (4 seeds) for both H0 and H1 to avoid slow-decay saddles.

Inputs:
  filtered_3_seasons.csv.gz     — Full 3-season league play-by-play

Outputs:
  phase5_r3_redo.csv             — H0 vs H1 fit summary + LR test result

Fixes vs the original 07_phase5_r3_redo.py:
  [B7] CRITICAL: γ M-step.
       Original used a heuristic gradient step:
         grad_γ = sum_pS_logm - α · sum_compensator_logm
         γ_new  = γ + 0.05 · sign(grad) · min(|grad|/n_total, 0.3)
       This is NOT a proper EM M-step — it's an ad hoc 1-step gradient
       update with hardcoded step size 0.05 and is not guaranteed to
       converge to the γ MLE.

       The correct M-step uses the PROFILED Q-function in γ. After the
       E-step, the joint Q in (α, γ) is:
         Q(α, γ) = S · log α + γ · S_logm − α · C(γ) + const
       where S = sum_pS, S_logm = Σ p^S_ij·log(m_j/m_avg), and
       C(γ) = Σ_j (m_j/m_avg)^γ · K_j with K_j = (1 − exp(−β(T−t_j))).
       For any γ, the α-maximizer is α*(γ) = S / C(γ); substituting:
         Q_profile(γ) = γ · S_logm − S · log C(γ) + const
       We maximize Q_profile over γ ∈ [-3, 3] via scipy.optimize.
       minimize_scalar (Brent's method, bounded), then set α = S / C(γ_new).
       This gives the correct profiled α/γ update conditional on β.

       Note: β here uses the standard recursive update (β = sum_pS /
       sum_pS_dt) rather than the exact finite-window MLE, for cross-script
       consistency with phase2/4. Because of this β approximation, the
       full cycle is not a strict full-EM monotone iteration; we rely on
       multi-start consistency and the final likelihood to assess
       convergence.

  [H0 multi-start] H0 (γ=0) is now also multi-start (4 seeds) to guard
       against slow-decay saddles, ensuring the LR test does not exaggerate
       Δ-LL by under-fitting H0.

  [Wording] Use ALPHA_INTERPRET_THRESHOLD = 1e-4 (consistent with phase 2/4/5).
  [Cross-check] R3 H0 fit should match phase2_v4 (α ≈ 0, γ irrelevant) up to
                subsample noise.

Why EM (not L-BFGS-B) for marked Hawkes:
  L-BFGS-B finds a saddle at (α ≈ 0.3, β ≈ 0.0003) which represents a
  degenerate "constant excitation" solution, not the true MLE. EM has
  monotone LL increase and converges to the correct optimum.

Execution time: ~5-10 minutes
================================================================================
"""

# (Original v1 docstring preserved below for archival reference.)
"""
Phase 5 R3 REDO: Marked Hawkes via EM (not L-BFGS-B).

Standard EM derivation for marked Hawkes with kernel:
  kappa(t, m) = alpha * beta * (m/m_avg)^gamma * exp(-beta * t)

E-step responsibilities for event i:
  p^B_i = mu / lam_i  (background)
  p^S_ij = alpha*beta*(m_j/m_avg)^gamma * exp(-beta*(t_i - t_j)) / lam_i  (excited by event j<i)

M-step:
  mu = sum_i p^B_i / (n_real * T)
  alpha = sum_i sum_{j<i} p^S_ij / sum_j (m_j/m_avg)^gamma * (1 - exp(-beta*(T - t_j)))
  beta:   solve recursion (similar to standard)
  gamma:  numerical 1D update on profiled likelihood (or fixed grid search)

For LR test, fix kernel/baseline structure, only flip gamma between 0 (H0) and free (H1).
"""

import numpy as np
import pandas as pd
import time
from scipy.stats import chi2
from scipy.optimize import minimize_scalar   # FIX [B7]

# Configuration
ALPHA_INTERPRET_THRESHOLD = 1e-4
LAM_FLOOR = 1e-12

print("=" * 70)
print("PHASE 5 R3 REDO v3: MARKED HAWKES via EM + LR test")
print("  [PAPER §7 marked-Hawkes robustness — authoritative]")
print("=" * 70)

# ---------- Load (reuse) ----------
df = pd.read_csv("filtered_3_seasons.csv.gz", compression="gzip", low_memory=False)
n_games_loaded = df['GAME_ID'].nunique()
print(f"  Loaded {len(df):,} rows, {n_games_loaded:,} games")

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
sub_reg = subs_raw[subs_raw["PERIOD"] <= 4].copy().sort_values(["GAME_ID","TEAM_ID","t_abs"])

realiz_times = []
realiz_marks = []
for (gid, tid), g in sub_reg.groupby(["GAME_ID","TEAM_ID"]):
    times = g["t_abs"].values
    unique_times, counts = np.unique(times, return_counts=True)
    keep = unique_times < T_HORIZON
    if keep.sum() > 0:
        realiz_times.append(unique_times[keep])
        realiz_marks.append(counts[keep].astype(float))

n_real = len(realiz_times)
n_total = sum(len(t) for t in realiz_times)
m_avg = np.concatenate(realiz_marks).mean()
print(f"\n  {n_real} realizations, {n_total} events, mean mark = {m_avg:.3f}")

# ============================================================
# Marked Hawkes EM
# Kernel: kappa(t, m) = alpha * beta * (m/m_avg)^gamma * exp(-beta * t)
# ============================================================
def fit_marked_em(realiz_times, realiz_marks, T, m_avg, gamma_fixed=None,
                  mu0=0.005, alpha0=0.3, beta0=1/180, gamma0=0.0,
                  max_iter=200, tol=1e-7, verbose=False):
    """
    Marked Hawkes EM fitter with FIX [B7]: profiled γ M-step.

    Kernel: κ(t, m) = α · β · (m/m_avg)^γ · exp(-β · t)

    After the E-step, the joint Q-function in (α, γ) is:
        Q(α, γ) = S · log α + γ · S_logm − α · C(γ) + const
    where:
        S      = Σ_i Σ_{j<i} p^S_ij                       [= sum_pS]
        S_logm = Σ_i Σ_{j<i} p^S_ij · log(m_j/m_avg)      [= sum_pS_logm]
        C(γ)   = Σ_j (m_j/m_avg)^γ · K_j
        K_j    = (1 − exp(−β(T − t_j)))                   [β fixed in M-step]

    For any γ, α*(γ) = S / C(γ). Substituting gives the profiled Q:
        Q_profile(γ) = γ · S_logm − S · log C(γ) + const

    We maximize Q_profile(γ) over γ ∈ [-3, 3] via scipy.optimize.
    minimize_scalar (Brent's method, bounded), then set α = S / C(γ_new).
    This is the proper profiled α/γ update conditional on the current β.

    Note: β here uses the standard recursive update (β = sum_pS / sum_pS_dt)
    rather than the exact finite-window MLE, for cross-script consistency
    with phase2/4. As a result, the full cycle is not a strict full-EM
    monotone iteration; we rely on multi-start consistency and the final
    likelihood to assess convergence.
    """
    mu, alpha, beta = mu0, alpha0, beta0
    gamma = gamma0 if gamma_fixed is None else gamma_fixed
    n_real = len(realiz_times)

    for it in range(max_iter):
        # E-step
        sum_pB = 0.0
        sum_pS = 0.0
        sum_pS_dt = 0.0
        sum_pS_logm = 0.0  # sum over (i,j<i) of p^S_ij · log(m_j/m_avg)

        # Per-realization cache for the γ M-step (β-dependent K_j is
        # rebuilt after the β update below).
        per_real_logm = []

        for sub_t, marks in zip(realiz_times, realiz_marks):
            n = len(sub_t)
            if n == 0:
                continue
            w = (marks / m_avg) ** gamma
            log_m = np.log(marks / m_avg)

            # Recursions:
            #   A[i] = Σ_{j<i} w_j · exp(-β(t_i - t_j))
            #   B[i] = Σ_{j<i} w_j · (t_i - t_j) · exp(-β(t_i - t_j))
            #   C[i] = Σ_{j<i} w_j · log(m_j/m_avg) · exp(-β(t_i - t_j))
            A = np.zeros(n); Bmat = np.zeros(n); C = np.zeros(n)
            for i in range(1, n):
                dt = sub_t[i] - sub_t[i-1]
                e = np.exp(-beta * dt)
                A[i] = e * (w[i-1] + A[i-1])
                Bmat[i] = e * (Bmat[i-1] + dt * (w[i-1] + A[i-1]))
                C[i] = e * (w[i-1] * log_m[i-1] + C[i-1])

            lam = np.maximum(mu + alpha * beta * A, LAM_FLOOR)
            pB = mu / lam

            sum_pB += pB.sum()
            sum_pS += (alpha * beta * A / lam).sum()
            sum_pS_dt += (alpha * beta * Bmat / lam).sum()
            sum_pS_logm += (alpha * beta * C / lam).sum()

            per_real_logm.append(log_m)

        # M-step: μ closed-form
        new_mu = max(sum_pB / (n_real * T), 1e-9)
        # M-step: β recursive update (same as phase2/4 — note: not exact
        # finite-window MLE, but consistent across all our EM scripts)
        new_beta = max(sum_pS / sum_pS_dt, 1e-7) if (sum_pS > 1e-10 and sum_pS_dt > 1e-12) else beta

        # Recompute K_j caches at the updated β before α/γ joint update
        # (β changed so the compensator basis K_j = 1-exp(-β(T-t_j)) changes)
        per_real_K_newbeta = []
        for sub_t in realiz_times:
            per_real_K_newbeta.append(1 - np.exp(-new_beta * (T - sub_t)))

        def comp_at_gamma(g):
            """C(γ) = Σ_j (m_j/m_avg)^γ · K_j (sum over all realizations)."""
            return sum(
                np.sum(np.exp(g * lm) * K)
                for lm, K in zip(per_real_logm, per_real_K_newbeta)
            )

        # M-step: γ via FIX [B7] PROFILED α — bounded Brent maximization.
        # Maximizing Q_profile(γ) = γ · S_logm − S · log C(γ) over γ ∈ [-3, 3],
        # then setting α = S/C(γ_new), is the proper profiled α/γ update
        # conditional on the current β. (See function docstring.)
        if gamma_fixed is None:
            def neg_profile_Q_gamma(g):
                Cg = max(comp_at_gamma(g), 1e-12)
                Q_prof = g * sum_pS_logm - sum_pS * np.log(Cg)
                return -Q_prof
            try:
                res_g = minimize_scalar(
                    neg_profile_Q_gamma,
                    bounds=(-3.0, 3.0),
                    method='bounded',
                    options={'xatol': 1e-5}
                )
                new_gamma = float(np.clip(res_g.x, -3.0, 3.0))
            except Exception:
                new_gamma = gamma
        else:
            new_gamma = gamma_fixed

        # M-step: α at the chosen γ (profiled MLE)
        C_new = max(comp_at_gamma(new_gamma), 1e-12)
        new_alpha = min(max(sum_pS / C_new, 0.0), 0.99)

        delta = max(abs(new_mu - mu), abs(new_alpha - alpha),
                    abs(new_beta - beta), abs(new_gamma - gamma))
        mu, alpha, beta, gamma = new_mu, new_alpha, new_beta, new_gamma

        if verbose and (it % 20 == 0 or it < 5):
            print(f"    iter {it:3d}: mu={mu:.5f} alpha={alpha:.6e} beta={beta:.5f} gamma={gamma:.4f}")

        if delta < tol:
            break

    return mu, alpha, beta, gamma, it + 1


def marked_ll(realiz_times, realiz_marks, T, m_avg, mu, alpha, beta, gamma):
    """Marked Hawkes log-likelihood for LR test."""
    total = 0.0
    for sub_t, marks in zip(realiz_times, realiz_marks):
        n = len(sub_t)
        if n == 0:
            total += -mu * T
            continue
        w = (marks / m_avg) ** gamma
        A = np.zeros(n)
        for i in range(1, n):
            A[i] = np.exp(-beta * (sub_t[i] - sub_t[i-1])) * (w[i-1] + A[i-1])
        lam = np.maximum(mu + alpha * beta * A, LAM_FLOOR)
        log_term = np.sum(np.log(lam))
        comp_excite = alpha * np.sum(w * (1 - np.exp(-beta * (T - sub_t))))
        total += log_term - mu * T - comp_excite
    return total

# Use a 1500-realization subsample so LR is stable
np.random.seed(7)
mark_idx = np.random.choice(n_real, 1500, replace=False)
mark_t = [realiz_times[i] for i in mark_idx]
mark_m = [realiz_marks[i] for i in mark_idx]
print(f"  Using {len(mark_t)}-realization subsample ({sum(len(t) for t in mark_t)} events)")

# H0: gamma = 0 (unmarked, equivalent to standard Hawkes since w=1 for all)
# FIX [#3]: H0 multi-start to guard against slow-decay saddles.
print("\n  Fitting H0 (gamma = 0) via EM, multi-start...")
h0_starts = [
    dict(mu0=0.005, alpha0=0.3,  beta0=1/180),
    dict(mu0=0.005, alpha0=0.01, beta0=1/180),
    dict(mu0=0.005, alpha0=0.3,  beta0=1/60),
    dict(mu0=0.005, alpha0=0.3,  beta0=1/600),
]
best_h0 = None
for k, s in enumerate(h0_starts):
    t0 = time.time()
    mu0_k, a0_k, b0_k, _, it0_k = fit_marked_em(
        mark_t, mark_m, T_HORIZON, m_avg,
        gamma_fixed=0.0, max_iter=200,
        verbose=False, **s
    )
    ll0_k = marked_ll(mark_t, mark_m, T_HORIZON, m_avg, mu0_k, a0_k, b0_k, 0.0)
    print(f"    start {k+1} (α0={s['alpha0']:.2f}, β0={s['beta0']:.4f}): "
          f"mu={mu0_k:.5f} alpha={a0_k:.6e} beta={b0_k:.5f} LL={ll0_k:.1f} "
          f"(iter {it0_k}, {time.time()-t0:.1f}s)")
    if best_h0 is None or ll0_k > best_h0[3]:
        best_h0 = (mu0_k, a0_k, b0_k, ll0_k, it0_k)
mu0, a0, b0, ll0, it0 = best_h0
print(f"\n  BEST H0: mu={mu0:.5f}  alpha={a0:.6e}  beta={b0:.5f}  LL={ll0:.1f}")

# H1: gamma free — multi-start EM
print("\n  Fitting H1 (gamma free) via EM, multi-start...")
h1_starts = [
    dict(mu0=0.005, alpha0=0.3,  beta0=1/180, gamma0=0.0),
    dict(mu0=0.005, alpha0=0.01, beta0=1/180, gamma0=0.0),   # FIX [#5]: low α start
    dict(mu0=0.005, alpha0=0.1,  beta0=1/300, gamma0=0.5),
    dict(mu0=0.003, alpha0=0.5,  beta0=1/120, gamma0=-0.5),
    dict(mu0=0.005, alpha0=0.3,  beta0=1/600, gamma0=1.0),
]
best_h1 = None
for k, s in enumerate(h1_starts):
    t0 = time.time()
    mu1, a1, b1, g1, it1 = fit_marked_em(mark_t, mark_m, T_HORIZON, m_avg, max_iter=200, **s)
    ll1 = marked_ll(mark_t, mark_m, T_HORIZON, m_avg, mu1, a1, b1, g1)
    print(f"    start {k+1} (gamma0={s['gamma0']:.1f}): "
          f"mu={mu1:.5f} alpha={a1:.6e} beta={b1:.5f} gamma={g1:.4f} LL={ll1:.1f} "
          f"(iter {it1}, {time.time()-t0:.1f}s)")
    if best_h1 is None or ll1 > best_h1[4]:
        best_h1 = (mu1, a1, b1, g1, ll1)

mu1, a1, b1, g1, ll1 = best_h1
print(f"\n  BEST H1: mu={mu1:.5f}  alpha={a1:.6e}  beta={b1:.5f}  gamma={g1:.4f}  LL={ll1:.1f}")

# LR test
LR = 2 * max(ll1 - ll0, 0)
p_LR = 1 - chi2.cdf(LR, df=1) if LR > 0 else 1.0

print("\n" + "="*70)
print("R3 LR TEST RESULT   [PAPER §7]")
print("="*70)
print(f"  H0 (gamma=0):    mu={mu0:.5f}  alpha={a0:.6e}  beta={b0:.5f}  LL={ll0:.1f}")
print(f"  H1 (gamma free): mu={mu1:.5f}  alpha={a1:.6e}  beta={b1:.5f}  gamma={g1:.4f}  LL={ll1:.1f}")
print(f"  Delta LL = {ll1 - ll0:.4f}")
print(f"  LR stat  = {LR:.4f}")
print(f"  p-value  = {p_LR:.4g} (chi^2_1)")
print(f"  Decision: {'REJECT H0' if p_LR < 0.05 else 'FAIL TO REJECT H0'}")

print("\n[FINAL R3 VERDICT]")
a0_below = a0 < ALPHA_INTERPRET_THRESHOLD
a1_below = a1 < ALPHA_INTERPRET_THRESHOLD
if a0_below and a1_below:
    print(f"  ✓ Both H0 (α={a0:.2e}) and H1 (α={a1:.2e}) are below the")
    print(f"    interpretation threshold ({ALPHA_INTERPRET_THRESHOLD:.0e}). Mark provides")
    print(f"    no rescue for self-excitation; the no-self-excitation finding holds")
    print(f"    even with a player-count mark.")
elif p_LR > 0.05:
    print("  ✓ Mark provides no significant additional information (FAIL TO REJECT H0).")
    print(f"    H0 α = {a0:.2e}, H1 α = {a1:.2e}.")
elif p_LR < 0.05 and a1_below:
    print("  ~ Mark statistically significant but H1 α is below interpretation threshold.")
    print(f"    γ = {g1:.3f} but H1 α = {a1:.2e}. Mark detects a weak modulation that")
    print(f"    does not lift α into the interpretable regime.")
else:
    print(f"  ⚠ Mark significant AND H1 α = {a1:.4e} is above threshold.")
    print(f"    γ = {g1:.3f}. This warrants further investigation.")

print()
print("  Caveats:")
print(f"   - LR test based on χ²_1 approximation; if α is at the boundary the")
print(f"     test may not be exact (Andrews 2001).")
print(f"   - This test is a complementary diagnostic. The primary M1f vs M3")
print(f"     inference is in the parametric bootstrap LR test.")

# Save
results_r3 = pd.DataFrame([{
    "H0_mu": mu0, "H0_alpha": a0, "H0_beta": b0, "H0_LL": ll0,
    "H1_mu": mu1, "H1_alpha": a1, "H1_beta": b1, "H1_gamma": g1, "H1_LL": ll1,
    "LR": LR, "p_value": p_LR,
    "n_realizations": len(mark_t),
    "n_events": sum(len(t) for t in mark_t),
    "alpha_threshold": ALPHA_INTERPRET_THRESHOLD,
    "H0_alpha_below_threshold": bool(a0_below),
    "H1_alpha_below_threshold": bool(a1_below),
    "status": "PAPER §7 — marked-Hawkes authoritative (γ via scipy.optimize)",
}])
results_r3.to_csv("phase5_r3_redo.csv", index=False)
print("\n[Saved] phase5_r3_redo.csv")

print("\n" + "="*70)
print("PHASE 5 R3 REDO v3 COMPLETE")
print(f"  H0 alpha = {a0:.6e}")
print(f"  H1 alpha = {a1:.6e},  gamma = {g1:.4f}")
print(f"  LR p-value = {p_LR:.4g}  ({'REJECT H0' if p_LR < 0.05 else 'FAIL TO REJECT H0'})")
print("  Formal M1f vs M3 inference remains based on the parametric bootstrap")
print("  LR test in 11_parametric_bootstrap_v3.py.")
print("="*70)