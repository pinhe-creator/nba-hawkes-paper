"""
================================================================================
Script 02: Phase 2 — Self-only Hawkes Process (FULL LEAGUE diagnostic)
================================================================================

Purpose:
  Fit an exponential-kernel self-only Hawkes process with a HOMOGENEOUS
  baseline (M1) to mass-aggregated substitution events from the full
  3-season NBA league, and compare against M0 and M3.

  POSITIONING IN THE PAPER:
    This script is a full-league diagnostic / sanity check, NOT the §6
    final inferential result. The paper's headline result on self-excitation
    uses the M1f specification (48-bin baseline + Hawkes self-excitation)
    fit by 08_fit_m1f.py with multi-start EM, and tested by parametric
    bootstrap in 11_parametric_bootstrap_v3.py.

    The M1 estimates here are also reproduced in 04_phase4_full_league.py
    via a different code path (`fit_self_quick`); the two implementations
    should agree up to numerical tolerance because they fit the SAME
    homogeneous-baseline self-only Hawkes model. They are NOT expected to
    agree with the M1f estimates in 08_fit_m1f.py, which fit a different
    model (different baseline structure).

Models compared:
  M0: λ(t) = μ                                                  (k = 1)
  M1: λ(t) = μ + α·β · Σ_{j: t_j < t} exp(-β·(t - t_j))         (k = 3)
  M3: λ(t) = μ_b[bin(t)]                                        (k = 48)

  M1f (which IS the §6 main specification) is fit elsewhere in 08_fit_m1f.py:
       λ(t) = μ_b[bin(t)] + α·β · Σ exp(-β·(t - t_j))           (k = 50)

Inputs:
  filtered_3_seasons.csv.gz   — Full 3-season league play-by-play (3,690 games)

  This is the SAME source step_B and phase4 use, ensuring consistency.

Outputs:
  phase2_model_comparison.csv  — M0, M1, M3 LL/AIC/BIC + KS diagnostic
  phase2_diagnostics.png       — Convergence + RTC + kernel plots
  residuals_hawkes.npy         — RTC residuals under M1 (kept name for back-compat)
  residuals_b2.npy             — RTC residuals under M3
  mass_subs_phase2.csv         — Mass-aggregated subs (named distinctly so it
                                 does not collide with downstream phase outputs)

Estimation: EM algorithm (Veen-Schoenberg 2008 form), single initialization.
  Final self-excitation inference uses multi-start EM in 08_fit_m1f.py,
  which estimates M1f rather than this homogeneous-baseline M1. The
  single-init estimate here is a rough sanity check; do not cite it as
  the paper's primary self-excitation estimate.

Fixes vs the original Phase 2:
  [B1] α M-step uses the proper finite-window compensator denominator
       Σ_j (1 - exp(-β(T - t_j))), not the event count.
  [I6] Random-time-change residuals now include the gap from t=0 to the
       first event (np.concatenate([[0], L])).
  [Data] Reads filtered_3_seasons.csv.gz directly (full 3-season league).
       Previously read subs_clean.csv (a LAL+DAL pilot subset) while the
       docstring claimed mass_subs.csv as input.
  [Robust] TEAM_ABBREVIATION fallback to TEAM_ID string if missing, so
           groupby does not silently drop rows with NaN team labels.

Execution time: ~6-12 minutes on a laptop (was ~3 min on the LAL+DAL pilot).
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
from scipy.stats import kstest

np.random.seed(42)

# ---------- Configuration ----------
DATA_FILE = "filtered_3_seasons.csv.gz"
T_HORIZON = 2880.0   # 4 quarters × 720 sec each (regulation only)
LAM_FLOOR = 1e-12

print("=" * 70)
print("PHASE 2: SELF-ONLY HAWKES (M1) on FULL 3-SEASON LEAGUE")
print("=" * 70)

if not os.path.exists(DATA_FILE):
    sys.exit(f"[FATAL] {DATA_FILE} not found in current directory.")

# ---------- Load + parse t_abs (identical to phase4 / step_B) ----------
print(f"\n[Loading] {DATA_FILE}")
df = pd.read_csv(DATA_FILE, low_memory=False)
n_games_loaded = df['GAME_ID'].nunique()
print(f"  {len(df):,} rows, {n_games_loaded:,} games")
if n_games_loaded < 3000:
    print(f"  [WARNING] Expected ~3,690 games for the full 3-season league,")
    print(f"            but found only {n_games_loaded:,}. This may be a subset.")

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

# ---------- Mass-aggregate subs (identical to phase4) ----------
print("[Mass-aggregating subs]")
subs = df[df["EVENTMSGTYPE"] == 8].dropna(subset=["t_abs", "PLAYER1_TEAM_ID"]).copy()
subs["TEAM_ID"] = subs["PLAYER1_TEAM_ID"].astype(int)

# TEAM abbreviation: use TEAM_ID string as fallback if abbreviation is missing.
# Without this, groupby on TEAM would silently drop rows with NaN team labels.
n_missing_team = subs["PLAYER1_TEAM_ABBREVIATION"].isna().sum()
if n_missing_team > 0:
    print(f"  [INFO] {n_missing_team:,} sub rows have missing TEAM_ABBREVIATION; "
          f"falling back to TEAM_ID string.")
subs["TEAM"] = subs["PLAYER1_TEAM_ABBREVIATION"].fillna(subs["TEAM_ID"].astype(str))

mass_subs = (
    subs.groupby(["GAME_ID", "TEAM_ID", "TEAM", "SEASON", "PERIOD", "t_abs"], as_index=False)
        .agg(n_players=("PLAYER1_ID", "count"))
)
mass_subs = mass_subs.sort_values(["GAME_ID", "TEAM_ID", "t_abs"]).reset_index(drop=True)
mass_reg = mass_subs[mass_subs["PERIOD"] <= 4].copy()
print(f"  Raw subs:         {len(subs):,}")
print(f"  Mass-aggregated:  {len(mass_subs):,}  (regulation: {len(mass_reg):,})")

# ---------- Build per-realization sequences ----------
# Note: events at t_abs >= T_HORIZON are dropped (regulation-end-buzzer subs)
n_at_or_after_T = (mass_reg["t_abs"] >= T_HORIZON).sum()
if n_at_or_after_T > 0:
    print(f"  [INFO] Dropping {n_at_or_after_T:,} regulation events at t_abs >= {T_HORIZON} "
          f"(buzzer-instant subs, no excitation interval after).")

realizations = []
for (gid, tid), g in mass_reg.groupby(["GAME_ID", "TEAM_ID"]):
    times = np.sort(g["t_abs"].values.astype(float))
    times = times[times < T_HORIZON]
    if len(times) > 0:
        realizations.append(times)

n_real = len(realizations)
n_total_events = sum(len(r) for r in realizations)
print(f"\n[Realizations] {n_real:,} (game × team) sequences")
print(f"  Total events = {n_total_events:,}")
print(f"  Mean events / realization = {np.mean([len(r) for r in realizations]):.2f}")

# ============================================================
# M1: Self-only Hawkes — log-likelihood and EM
# ============================================================
def hawkes_loglik_single(times, mu, alpha, beta, T):
    """Log-likelihood for one realization of exponential-kernel Hawkes on [0, T]."""
    n = len(times)
    if n == 0:
        return -mu * T
    A = np.zeros(n)
    for i in range(1, n):
        A[i] = np.exp(-beta * (times[i] - times[i - 1])) * (1 + A[i - 1])
    lam = np.maximum(mu + alpha * beta * A, LAM_FLOOR)
    log_term = np.sum(np.log(lam))
    compensator_excite = alpha * np.sum(1 - np.exp(-beta * (T - times)))
    return log_term - mu * T - compensator_excite


def hawkes_loglik_pooled(realizations, mu, alpha, beta, T):
    return sum(hawkes_loglik_single(r, mu, alpha, beta, T) for r in realizations)


def fit_hawkes_em(realizations, T, max_iter=300, tol=1e-7, verbose=True):
    """
    EM for exponential-kernel univariate Hawkes pooled across i.i.d. realizations.

    M-step formulas (with proper finite-window compensator denominator for α):
      μ_new   = (Σ p_back) / (n_real · T)
      α_new   = (Σ p_off ) / Σ_j [1 − exp(−β·(T − t_j))]      ← FIX [B1]
      β_new   = (Σ p_off ) / Σ p_off · (t_i − t_j)

    Reference: Veen & Schoenberg (2008), Lewis & Mohler (2011).
    """
    total_events = sum(len(r) for r in realizations)
    n_real_loc = len(realizations)
    mu = total_events / (n_real_loc * T) * 0.7
    alpha = 0.3
    beta = 1.0 / 180.0

    history = []
    for it in range(max_iter):
        sum_p_back = 0.0
        sum_p_off = 0.0
        sum_p_off_times_dt = 0.0
        sum_compensator_excite = 0.0   # finite-window excitation compensator (denominator for α)

        for times in realizations:
            n = len(times)
            if n == 0:
                continue
            A = np.zeros(n)
            B = np.zeros(n)
            for i in range(1, n):
                dt = times[i] - times[i - 1]
                e = np.exp(-beta * dt)
                A[i] = e * (1 + A[i - 1])
                B[i] = e * (B[i - 1] + dt * (1 + A[i - 1]))
            lam = np.maximum(mu + alpha * beta * A, LAM_FLOOR)
            p_back_i = mu / lam
            sum_p_back += p_back_i.sum()
            sum_p_off += (1 - p_back_i).sum()
            sum_p_off_times_dt += (alpha * beta * B / lam).sum()
            sum_compensator_excite += np.sum(1 - np.exp(-beta * (T - times)))

        # ----- M-step -----
        new_mu = max(sum_p_back / (n_real_loc * T), 1e-9)
        # FIX [B1]: divide by finite-window compensator, not total events
        if sum_compensator_excite > 1e-12:
            new_alpha = min(sum_p_off / sum_compensator_excite, 0.99)
        else:
            new_alpha = 0.0
        if sum_p_off > 1e-10 and sum_p_off_times_dt > 1e-12:
            new_beta = max(sum_p_off / sum_p_off_times_dt, 1e-7)
        else:
            new_beta = beta

        ll = hawkes_loglik_pooled(realizations, new_mu, new_alpha, new_beta, T)
        history.append((it, new_mu, new_alpha, new_beta, ll))

        delta = max(abs(new_mu - mu), abs(new_alpha - alpha), abs(new_beta - beta))
        mu, alpha, beta = new_mu, new_alpha, new_beta

        if verbose and (it % 10 == 0 or it < 5):
            print(f"  iter {it:3d}: mu={mu:.6f}  alpha={alpha:.6f}  beta={beta:.6f}  ll={ll:.2f}")

        if delta < tol:
            if verbose:
                print(f"  Converged at iter {it}")
            break
    return mu, alpha, beta, history


# ---------- Fit M1 ----------
print("\n[Fitting M1: Self-only Hawkes via EM (single init, full league)]")
print("  Note: This is a diagnostic single-init fit. The paper's headline")
print("        self-excitation result is from 08_fit_m1f.py (multi-start, M1f).")
mu_hat, alpha_hat, beta_hat, history = fit_hawkes_em(
    realizations, T_HORIZON, max_iter=300, tol=1e-7, verbose=True
)

print(f"\n[M1 estimates]")
print(f"  μ (baseline rate)         = {mu_hat:.6f} events/sec")
print(f"  α (branching ratio)       = {alpha_hat:.6f}")
print(f"  β (decay rate)            = {beta_hat:.6f} /sec")
if alpha_hat > 1e-6:
    print(f"  1/β  (mean kernel duration) = {1/beta_hat:.1f} sec")
    print(f"  half-life                   = {np.log(2)/beta_hat:.1f} sec")
else:
    print(f"  α ≈ 0 — β is weakly identified and not substantively interpreted")

ll_M1 = hawkes_loglik_pooled(realizations, mu_hat, alpha_hat, beta_hat, T_HORIZON)
print(f"  log-likelihood = {ll_M1:.2f}")

k_M1 = 3
aic_M1 = 2 * k_M1 - 2 * ll_M1
bic_M1 = k_M1 * np.log(n_total_events) - 2 * ll_M1
print(f"  AIC = {aic_M1:.2f}  |  BIC = {bic_M1:.2f}")

# ============================================================
# M0: Homogeneous Poisson
# ============================================================
print("\n[M0: Homogeneous Poisson]")
mu_pois = n_total_events / (n_real * T_HORIZON)
ll_M0 = -mu_pois * n_real * T_HORIZON + n_total_events * np.log(mu_pois)
k_M0 = 1
aic_M0 = 2 * k_M0 - 2 * ll_M0
bic_M0 = k_M0 * np.log(n_total_events) - 2 * ll_M0
print(f"  rate            = {mu_pois:.6f}  ({mu_pois*720:.3f} per quarter)")
print(f"  log-likelihood  = {ll_M0:.2f}")
print(f"  AIC = {aic_M0:.2f}  |  BIC = {bic_M0:.2f}")

# ============================================================
# M3: 48-bin inhomogeneous Poisson
# ============================================================
print("\n[M3: 48-bin inhomogeneous Poisson]")
n_bins_per_period = 12
total_bins = 4 * n_bins_per_period   # 48
bin_width = 720.0 / n_bins_per_period   # 60 sec

all_event_times = np.concatenate(realizations) if realizations else np.array([])
event_bins = np.minimum((all_event_times // bin_width).astype(int), total_bins - 1)
counts = np.bincount(event_bins, minlength=total_bins)
rates_M3 = counts / (n_real * bin_width)

log_rate_at_event = np.log(np.maximum(rates_M3[event_bins], LAM_FLOOR))
ll_M3 = log_rate_at_event.sum() - (rates_M3 * bin_width * n_real).sum()
k_M3 = total_bins
aic_M3 = 2 * k_M3 - 2 * ll_M3
bic_M3 = k_M3 * np.log(n_total_events) - 2 * ll_M3
print(f"  bins             = {total_bins} (each {bin_width:.0f}s)")
print(f"  log-likelihood   = {ll_M3:.2f}")
print(f"  AIC = {aic_M3:.2f}  |  BIC = {bic_M3:.2f}")

# ============================================================
# Random-time-change residuals (with first-event gap fix)
# ============================================================
print("\n[Random-Time-Change residual diagnostic]")

def compute_compensator_M1(times, mu, alpha, beta):
    """Cumulative compensator Λ(t_i) at each event under M1."""
    n = len(times)
    Lambda = np.zeros(n)
    for i in range(n):
        if i == 0:
            excite = 0.0
        else:
            excite = alpha * np.sum(1 - np.exp(-beta * (times[i] - times[:i])))
        Lambda[i] = mu * times[i] + excite
    return Lambda

def compute_compensator_M3(times, rates, bin_width_):
    """Cumulative Λ(t) for piecewise-constant rate model."""
    n_bins_loc = len(rates)
    L = np.zeros(len(times))
    for k, t in enumerate(times):
        full_bin = min(int(t // bin_width_), n_bins_loc - 1)
        full_part = rates[:full_bin].sum() * bin_width_
        residual_t = t - full_bin * bin_width_
        L[k] = full_part + rates[full_bin] * residual_t
    return L

# FIX [I6]: include gap from 0 to first event
residuals_M1 = []
for r in realizations:
    if len(r) > 0:
        L = compute_compensator_M1(r, mu_hat, alpha_hat, beta_hat)
        residuals_M1.extend(np.diff(np.concatenate([[0.0], L])))
residuals_M1 = np.array(residuals_M1)
ks_stat_M1, ks_p_M1 = kstest(residuals_M1, 'expon', args=(0, 1))
print(f"  M1: KS stat = {ks_stat_M1:.4f}  p = {ks_p_M1:.4g}  (n={len(residuals_M1):,})")
print(f"      mean = {residuals_M1.mean():.3f} (target 1.0)  var = {residuals_M1.var():.3f}")

residuals_M0 = []
for r in realizations:
    if len(r) > 0:
        L = mu_pois * r
        residuals_M0.extend(np.diff(np.concatenate([[0.0], L])))
residuals_M0 = np.array(residuals_M0)
ks_stat_M0, ks_p_M0 = kstest(residuals_M0, 'expon', args=(0, 1))
print(f"  M0: KS stat = {ks_stat_M0:.4f}  p = {ks_p_M0:.4g}")

residuals_M3 = []
for r in realizations:
    if len(r) > 0:
        L = compute_compensator_M3(r, rates_M3, bin_width)
        residuals_M3.extend(np.diff(np.concatenate([[0.0], L])))
residuals_M3 = np.array(residuals_M3)
ks_stat_M3, ks_p_M3 = kstest(residuals_M3, 'expon', args=(0, 1))
print(f"  M3: KS stat = {ks_stat_M3:.4f}  p = {ks_p_M3:.4g}")
print(f"      mean = {residuals_M3.mean():.3f}  var = {residuals_M3.var():.3f}")

# ============================================================
# Comparison summary
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY (lower AIC/BIC = better)")
print("=" * 70)
results = pd.DataFrame({
    "Model":   ["M0: Hom. Poisson",   "M1: Self-only Hawkes", "M3: Inhom. Poisson"],
    "k":       [k_M0,                  k_M1,                   k_M3],
    "LogLik":  [ll_M0,                 ll_M1,                  ll_M3],
    "AIC":     [aic_M0,                aic_M1,                 aic_M3],
    "BIC":     [bic_M0,                bic_M1,                 bic_M3],
    "KS_stat": [ks_stat_M0,            ks_stat_M1,             ks_stat_M3],
    "KS_p":    [ks_p_M0,               ks_p_M1,                ks_p_M3],
})
print(results.to_string(index=False, float_format='%.4f'))
results.to_csv("phase2_model_comparison.csv", index=False)
print("\n[Saved] phase2_model_comparison.csv")

# ============================================================
# Diagnostic plots
# ============================================================
fig = plt.figure(figsize=(14, 11), constrained_layout=True)
gs = gridspec.GridSpec(3, 2, figure=fig)

# (1) EM convergence
ax = fig.add_subplot(gs[0, 0])
hist = pd.DataFrame(history, columns=["iter", "mu", "alpha", "beta", "ll"])
ax2 = ax.twinx()
ax.plot(hist["iter"],  hist["ll"],         'b-',  label="log-lik")
ax2.plot(hist["iter"], hist["alpha"],      'r--', label="α")
ax2.plot(hist["iter"], hist["beta"]*100,   'g:',  label="β × 100")
ax.set_xlabel("EM iteration")
ax.set_ylabel("Log-likelihood (blue)")
ax2.set_ylabel("α (red), β·100 (green)")
ax.set_title("EM convergence")
ax.legend(loc="lower right")
ax2.legend(loc="center right")

# (2) Empirical vs model rates
ax = fig.add_subplot(gs[0, 1])
hist_counts, hist_edges = np.histogram(all_event_times, bins=72, range=(0, T_HORIZON))
emp_rate = hist_counts / (n_real * (hist_edges[1] - hist_edges[0]))
emp_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
ax.plot(emp_centers, emp_rate, 'k-', lw=1.5, label="Empirical rate")
ax.axhline(mu_pois, color='blue', linestyle='--', lw=1, label=f"M0 const = {mu_pois:.4f}")
ax.step((np.arange(total_bins) + 0.5) * bin_width, rates_M3, where="mid",
        color='orange', lw=1, label="M3 piecewise")
ax.axhline(mu_hat, color='red', linestyle=':', lw=1, label=f"M1 baseline μ={mu_hat:.4f}")
for boundary in [720, 1440, 2160]:
    ax.axvline(boundary, color='gray', alpha=0.3)
ax.set_xlabel("Seconds in game"); ax.set_ylabel("Rate (events/sec)")
ax.set_title("Empirical event rate vs model baselines")
ax.legend(loc="upper right", fontsize=8)

# (3) M1 RTC QQ
ax = fig.add_subplot(gs[1, 0])
sr = np.sort(residuals_M1)
tq = -np.log(1 - (np.arange(len(sr)) + 0.5) / len(sr))
ax.plot(tq, sr, '.', alpha=0.4, ms=2, label="M1 residuals")
mx = max(tq.max(), sr.max())
ax.plot([0, mx], [0, mx], 'r-', label="y=x (Exp(1))")
ax.set_xlabel("Theoretical Exp(1) quantile"); ax.set_ylabel("Residual quantile")
ax.set_title(f"QQ: M1 RTC residuals\nKS={ks_stat_M1:.3f}, p={ks_p_M1:.3g}")
ax.legend()

# (4) M3 RTC QQ
ax = fig.add_subplot(gs[1, 1])
sr = np.sort(residuals_M3)
tq = -np.log(1 - (np.arange(len(sr)) + 0.5) / len(sr))
ax.plot(tq, sr, '.', alpha=0.4, ms=2, color="orange", label="M3 residuals")
mx = max(tq.max(), sr.max())
ax.plot([0, mx], [0, mx], 'r-', label="y=x")
ax.set_xlabel("Theoretical Exp(1) quantile"); ax.set_ylabel("Residual quantile")
ax.set_title(f"QQ: M3 RTC residuals\nKS={ks_stat_M3:.3f}, p={ks_p_M3:.3g}")
ax.legend()

# (5) Residual histograms
ax = fig.add_subplot(gs[2, 0])
ax.hist(residuals_M1, bins=60, density=True, alpha=0.6, color="steelblue",
        edgecolor="black", label="M1 residuals")
xx = np.linspace(0, residuals_M1.max(), 200)
ax.plot(xx, np.exp(-xx), 'r-', lw=2, label="Exp(1) density")
ax.set_xlim(0, np.percentile(residuals_M1, 99))
ax.set_xlabel("Residual"); ax.set_ylabel("Density")
ax.set_title("M1 residual distribution vs Exp(1)")
ax.legend()

# (6) Hawkes kernel
ax = fig.add_subplot(gs[2, 1])
t_grid = np.linspace(0, 600, 300)
kernel_vals = alpha_hat * beta_hat * np.exp(-beta_hat * t_grid)
ax.plot(t_grid, kernel_vals, 'b-', lw=2)
ax.fill_between(t_grid, 0, kernel_vals, alpha=0.3)
ax.set_xlabel("t since previous event (s)"); ax.set_ylabel("Kernel value")
hl_str = f"{np.log(2)/beta_hat:.1f}s" if beta_hat > 1e-6 else "n/a"
ax.set_title(f"M1 kernel: α={alpha_hat:.4f}, β={beta_hat:.6f}, half-life={hl_str}")
ax.grid(True, alpha=0.3)

plt.savefig("phase2_diagnostics.png", dpi=120, bbox_inches="tight")
print("[Saved] phase2_diagnostics.png")

# ---------- Persist residuals + a renamed mass-subs file ----------
np.save("residuals_hawkes.npy", residuals_M1)
np.save("residuals_b2.npy", residuals_M3)
mass_subs.to_csv("mass_subs_phase2.csv", index=False)
print("[Saved] residuals_hawkes.npy, residuals_b2.npy, mass_subs_phase2.csv")

print("\n" + "=" * 70)
print("Phase 2 complete.")
print("=" * 70)