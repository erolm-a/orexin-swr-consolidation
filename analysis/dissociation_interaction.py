"""
Formal test of the encoding/consolidation dissociation.

Reads:  results/agent_model_results.pkl

Fits DI = b0 + b1*x + b2*x^2 + b3*R + b4*R*x + b5*R*x^2 jointly across
both regimes, where x is the mean-centred OX2R level and R the regime
indicator. b5 is the dissociation. Reported three ways: raw DI,
within-regime standardised DI, and a rank-based paired delta rho.

Paths are resolved by orexin_paths; nothing here is machine-specific.
"""

import sys
from pathlib import Path

def _repo_root() -> Path:
    """Find the repository root whether this runs as a file or pasted in a cell."""
    start = (Path(globals()["__file__"]).resolve().parent
             if "__file__" in globals() else Path.cwd().resolve())
    for d in (start, *start.parents):
        if (d / "orexin_paths.py").exists():
            return d
    raise ImportError(
        "orexin_paths.py not found above " + str(start) + ".\n"
        "Running from a notebook? cd into the repository first, e.g.\n"
        "    %cd /path/to/orexin-swr-consolidation")


sys.path.insert(0, str(_repo_root()))

from orexin_paths import FILES, ensure_dirs  # noqa: E402

ensure_dirs()

"""
Formal test of the encoding/consolidation dissociation.
=======================================================

Replaces the informal "encoding rho > 0.5 AND consolidation curvature < 0"
criterion with a single fitted contrast.

Model, fitted jointly across both regimes:

    DI = b0 + b1*x + b2*x^2 + b3*R + b4*R*x + b5*R*x^2

    x = OX2R level, mean-centred
    R = regime indicator (0 = encoding-dominant, 1 = consolidation-dominant)

    b5 is the dissociation: the difference in quadratic coefficient between
    regimes. b2 and b5 are invariant to translation of x, so the centring
    affects only the interpretation of b1 and b4.

Inference is by paired replicate bootstrap. Both regimes are generated with
seed = rep * 1000 and share the memory pair and agent initialisation at each
replicate index, so a resampled replicate index is applied to BOTH regimes
within a draw. This preserves the common-random-number structure.

Three framings are reported. They are deliberately redundant:

  (1) b5, raw DI            -- amplitude and shape together
  (2) b5, DI standardised
      within regime         -- shape only; immune to a difference in DI range
  (3) paired Delta rho      -- rank-based; no functional form assumed

Reads only from the confirmed results pickle. No re-simulation.

Requires in the pickle:
    D['dual_pathway_sweep']['fixed_consolidation']['da']   encoding regime
    D['dual_pathway_sweep']['fixed_encoding']['da']        consolidation regime
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------- config ---
PKL = FILES['results']

N_BOOT = 10_000
BOOT_SEED = 20260818

# mode key -> (regime label, regime indicator R)
REGIMES = {
    'fixed_consolidation': ('Encoding-dominant',      0),
    'fixed_encoding':      ('Consolidation-dominant', 1),
}


# ------------------------------------------------------------- utilities ---
def load_arm(dp, mode):
    """Return (ox_levels asc, DI matrix [n_reps x n_levels]) for one arm."""
    if mode not in dp:
        raise KeyError(f"dual_pathway_sweep has no mode {mode!r}. "
                       f"Present: {sorted(dp)}")
    da = dp[mode].get('da')
    if da is None:
        raise KeyError(f"dual_pathway_sweep[{mode!r}] has no 'da'. Re-dump the "
                       f"model cell with per-replicate arrays saved; this "
                       f"script will not re-simulate.")
    ox = sorted(da.keys(), key=float)
    mat = np.column_stack([np.asarray(da[k], float) for k in ox])
    return np.array([float(k) for k in ox]), mat


def design(x_centred, R):
    """Design matrix for one regime at the given centred OX levels."""
    n = len(x_centred)
    return np.column_stack([
        np.ones(n),                       # b0
        x_centred,                        # b1
        x_centred ** 2,                   # b2
        np.full(n, float(R)),             # b3
        np.full(n, float(R)) * x_centred,        # b4
        np.full(n, float(R)) * x_centred ** 2,   # b5
    ])


def fit_b5(mat_enc, mat_con, X_enc, X_con, rows):
    """OLS on the pooled design for one bootstrap draw. Returns full coefs."""
    y = np.concatenate([mat_enc[rows].ravel(), mat_con[rows].ravel()])
    n_rep = len(rows)
    X = np.vstack([np.tile(X_enc, (n_rep, 1)), np.tile(X_con, (n_rep, 1))])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def pct_ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def per_rep_rho(mat, ox):
    """Spearman rho computed WITHIN each replicate, one value per replicate.

    The replicates are paired under common random numbers, so a within-replicate
    rank correlation removes the across-replicate offset that dominates a
    pooled rho. This is neither rho-on-condition-means (which is 1.0 by
    construction on a monotone 5-point mean curve) nor rho on pooled
    observations (which is swamped by between-replicate variance).
    """
    return np.array([spearmanr(ox, row).statistic for row in mat])


def vertex(ox, curve):
    """Turning point of the fitted quadratic, clipped to the sampled range.

    Draws whose unclipped vertex falls outside the grid accumulate at the
    endpoints, so a clipped vertex sitting exactly on a boundary is censored
    rather than located (Box & Wilson, 1951).
    """
    q2, q1, _ = np.polyfit(ox, curve, 2)
    if q2 == 0:
        return np.nan
    return float(np.clip(-q1 / (2.0 * q2), ox.min(), ox.max()))


def is_interior(ox, curve):
    """True when the largest mean DI falls at a level other than an endpoint.

    Discrete companion to the vertex estimate; makes no assumption that the
    response is quadratic.
    """
    j = int(np.argmax(curve))
    return 0 < j < len(ox) - 1


def fmt_p(p, n_boot):
    """Report bootstrap tail probabilities at the resolution the draw allows."""
    res = 1.0 / n_boot
    if p < res:
        return f"< {res:.0e}"
    if p > 1.0 - res:
        return f"> {1.0 - res:.4f}"
    return f"= {p:.4f}"


# ------------------------------------------------------------------ main ---
def main():
    with open(PKL, 'rb') as f:
        D = pickle.load(f)

    dp = D['dual_pathway_sweep']
    ox_enc, mat_enc = load_arm(dp, 'fixed_consolidation')
    ox_con, mat_con = load_arm(dp, 'fixed_encoding')

    if not np.allclose(ox_enc, ox_con):
        raise ValueError(f"OX2R grids differ between arms: {ox_enc} vs {ox_con}")
    if mat_enc.shape[0] != mat_con.shape[0]:
        raise ValueError(
            f"Replicate counts differ ({mat_enc.shape[0]} vs "
            f"{mat_con.shape[0]}); the arms cannot be paired. Re-run both at "
            f"the same n_reps, or switch to an unpaired bootstrap.")

    ox = ox_enc
    n_rep, n_lev = mat_enc.shape
    x = ox - ox.mean()

    et_enc = dp['fixed_consolidation'].get('enc_trials', 'n/a')
    et_con = dp['fixed_encoding'].get('enc_trials', 'n/a')

    X_enc = design(x, REGIMES['fixed_consolidation'][1])
    X_con = design(x, REGIMES['fixed_encoding'][1])

    # -- point estimates, raw DI ------------------------------------------
    all_rows = np.arange(n_rep)
    coef = fit_b5(mat_enc, mat_con, X_enc, X_con, all_rows)
    b2_hat, b5_hat = coef[2], coef[5]

    # -- point estimates, DI standardised within regime --------------------
    z_enc = (mat_enc - mat_enc.mean()) / mat_enc.std(ddof=1)
    z_con = (mat_con - mat_con.mean()) / mat_con.std(ddof=1)
    coef_z = fit_b5(z_enc, z_con, X_enc, X_con, all_rows)
    b5_z_hat = coef_z[5]

    # -- rank statistic, within-replicate ----------------------------------
    rho_enc_rep = per_rep_rho(mat_enc, ox)
    rho_con_rep = per_rep_rho(mat_con, ox)
    rho_enc = float(np.mean(rho_enc_rep))
    rho_con = float(np.mean(rho_con_rep))
    drho_hat = rho_con - rho_enc

    # -- interior optimum, per arm -----------------------------------------
    curve_enc_hat = mat_enc.mean(axis=0)
    curve_con_hat = mat_con.mean(axis=0)
    vtx_enc_hat = vertex(ox, curve_enc_hat)
    vtx_con_hat = vertex(ox, curve_con_hat)

    # -- paired bootstrap over replicate indices ---------------------------
    rng = np.random.default_rng(BOOT_SEED)
    b5_b = np.empty(N_BOOT)
    b5_z_b = np.empty(N_BOOT)
    drho_b = np.empty(N_BOOT)
    vtx_enc_b = np.empty(N_BOOT)
    vtx_con_b = np.empty(N_BOOT)
    int_enc_b = np.empty(N_BOOT, bool)
    int_con_b = np.empty(N_BOOT, bool)

    for i in range(N_BOOT):
        rows = rng.integers(0, n_rep, n_rep)          # same rows for both arms
        b5_b[i] = fit_b5(mat_enc, mat_con, X_enc, X_con, rows)[5]
        b5_z_b[i] = fit_b5(z_enc, z_con, X_enc, X_con, rows)[5]
        drho_b[i] = rho_con_rep[rows].mean() - rho_enc_rep[rows].mean()

        c_enc = mat_enc[rows].mean(axis=0)
        c_con = mat_con[rows].mean(axis=0)
        vtx_enc_b[i] = vertex(ox, c_enc)
        vtx_con_b[i] = vertex(ox, c_con)
        int_enc_b[i] = is_interior(ox, c_enc)
        int_con_b[i] = is_interior(ox, c_con)

    # ----------------------------------------------------------- report ---
    W = 78
    print("=" * W)
    print("ENCODING vs CONSOLIDATION DISSOCIATION - INTERACTION TEST")
    print("=" * W)
    print(f"  source        : {PKL.name}")
    print(f"  OX2R grid     : {', '.join(f'{v:g}' for v in ox)}  ({n_lev} levels)")
    print(f"  replicates    : {n_rep} per regime, paired (seed = rep * 1000)")
    print(f"  bootstrap     : {N_BOOT:,} paired resamples of replicate index")
    print(f"  enc_trials    : encoding arm {et_enc}, consolidation arm {et_con}")
    if et_enc != et_con:
        print("                  -> arms differ; an enc_trials offset loads on")
        print("                     b3, not b5. Cite the encoding-sensitivity")
        print("                     result (vertical offset, curvature")
        print("                     unchanged) in Methods.")

    print("\n  -- mean DI by regime --")
    print(f"  {'OX2R':<8}{'Encoding':<14}{'Consolidation':<14}")
    for j, v in enumerate(ox):
        print(f"  {v:<8.2f}{mat_enc[:, j].mean():<14.4f}{mat_con[:, j].mean():<14.4f}")

    print("\n  -- (1) b5, raw DI  [primary] --")
    lo, hi = pct_ci(b5_b)
    p = float(np.mean(b5_b >= 0))
    print(f"  b2 (encoding arm)          = {b2_hat:+.5f}")
    print(f"  b5 (regime x OX2R^2)       = {b5_hat:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  P(b5 >= 0)                 {fmt_p(p, N_BOOT)}")
    print(f"  implied consolidation q2   = {b2_hat + b5_hat:+.5f}")

    print("\n  -- (2) b5, DI standardised within regime  [scale-free] --")
    lo_z, hi_z = pct_ci(b5_z_b)
    p_z = float(np.mean(b5_z_b >= 0))
    print(f"  b5 (standardised)          = {b5_z_hat:+.5f}  95% CI [{lo_z:+.5f}, {hi_z:+.5f}]")
    print(f"  P(b5 >= 0)                 {fmt_p(p_z, N_BOOT)}")

    print("\n  -- (3) within-replicate Spearman rho  [rank-based] --")
    lo_r, hi_r = pct_ci(drho_b)
    p_r = float(np.mean(drho_b >= 0))
    print(f"  rho encoding               = {rho_enc:+.4f}  "
          f"(SD across reps {rho_enc_rep.std(ddof=1):.4f})")
    print(f"  rho consolidation          = {rho_con:+.4f}  "
          f"(SD across reps {rho_con_rep.std(ddof=1):.4f})")
    print(f"  delta rho                  = {drho_hat:+.4f}  95% CI [{lo_r:+.4f}, {hi_r:+.4f}]")
    print(f"  P(delta rho >= 0)          {fmt_p(p_r, N_BOOT)}")
    print("  NOTE: computed within replicate, not on condition means. A rho of")
    print("        1.00 on a monotone 5-point mean curve is uninformative; do")
    print("        not report both in the same manuscript without saying which")
    print("        is which.")

    print("\n  -- (4) interior optimum  [locates the regime difference] --")
    ve_lo, ve_hi = pct_ci(vtx_enc_b)
    vc_lo, vc_hi = pct_ci(vtx_con_b)
    pi_enc = float(np.mean(int_enc_b))
    pi_con = float(np.mean(int_con_b))
    edge = (ox.min(), ox.max())
    print(f"  {'':<27}{'Encoding':<24}{'Consolidation'}")
    print(f"  {'fitted vertex':<27}{vtx_enc_hat:<24.3f}{vtx_con_hat:.3f}")
    print(f"  {'95% CI':<27}{f'[{ve_lo:.3f}, {ve_hi:.3f}]':<24}"
          f"[{vc_lo:.3f}, {vc_hi:.3f}]")
    print(f"  {'P(interior)':<27}{pi_enc:<24.4f}{pi_con:.4f}")
    for lbl, lo_v, hi_v in (('Encoding', ve_lo, ve_hi),
                            ('Consolidation', vc_lo, vc_hi)):
        if np.isclose(lo_v, edge[0]) or np.isclose(hi_v, edge[1]):
            print(f"  -> {lbl} interval touches a sampled boundary; that limit")
            print(f"     is censored, not located. Report descriptively.")
    print("  Concavity alone is compatible with a saturating monotone curve.")
    print("  The regime claim rests on the vertex falling strictly inside the")
    print("  sampled range, together with P(interior) -- not on the sign of q2.")

    agree = sum([hi < 0, hi_z < 0, hi_r < 0])
    print("\n" + "-" * W)
    print(f"  {agree}/3 curvature framings place the upper CI limit below zero.")
    if agree == 3:
        print("  The curvature difference is supported as a single tested")
        print("  contrast, independent of DI scale and of functional form.")
    elif agree == 0:
        print("  No framing separates the regimes. The dissociation as")
        print("  currently claimed is not supported.")
    else:
        print("  Framings disagree. Check whether the separation is carried by")
        print("  DI amplitude rather than curve shape before reporting.")

    if pi_con > 0.95 and pi_enc < 0.05:
        print("  P(interior) separates cleanly: an interior optimum is present")
        print("  in consolidation and absent in encoding.")
    else:
        print(f"  P(interior) does NOT separate cleanly (enc {pi_enc:.3f}, "
              f"con {pi_con:.3f}).")
        print("  Both arms are concave, so this is the statistic that carries")
        print("  the claim. Resolve it before writing the Results sentence.")
    print("=" * W)
