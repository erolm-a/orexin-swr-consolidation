"""
common.stats — statistical / bootstrap helpers shared by the agent model, the
figure script and the diagnostic checks.

All functions here are pure: they take per-replicate ``(ox -> array)`` dicts
and return plain dicts of statistics. Historically they were defined inside the
agent script and re-invented / expected-from-kernel-state in the figure script;
they now live in one place.
"""

import numpy as np
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


def test_curvature(oxs, da, n_boot=10000, seed=42):
    """
    Sign-tested quadratic fit of the DI-vs-OX2R curve on condition means.

    Returns ``curvature`` (quadratic coefficient, negative = inverted-U),
    ``boot_sign_prob_nonneg`` and ``optimal_ox2r`` (vertex, if concave).
    """
    rng = np.random.RandomState(seed); nr = len(da[oxs[0]]); X = np.array(oxs)
    def fq(m):
        A = np.column_stack([np.ones_like(X), X, X**2])
        return np.linalg.lstsq(A, m, rcond=None)[0]
    obs = fq(np.array([np.mean(da[ox]) for ox in oxs]))
    bc  = np.array([fq(np.array([np.mean(da[ox][rng.choice(nr, nr, True)]) for ox in oxs]))[2]
                    for _ in range(n_boot)])
    v   = -obs[1] / (2 * obs[2]) if obs[2] < 0 else np.nan
    return {'curvature': float(obs[2]), 'boot_sign_prob_nonneg': float(np.mean(bc >= 0)),
            'optimal_ox2r': float(v) if not np.isnan(v) else None}


def test_monotonicity(oxs, da):
    """Spearman rank correlation between OX2R level and the DI means."""
    m = np.array([np.mean(da[ox]) for ox in oxs])
    rho, p = spearmanr(np.array(oxs), m)
    return {'spearman_rho': float(rho), 'spearman_p': float(p)}


def bootstrap_continuous_peak(da, oxs, n_boot=5000, seed=42):
    """
    Bootstrap the location of the fitted quadratic vertex (continuous peak).

    Returns the per-boot ``peaks`` array plus ``mean`` and ``ci_95``. Resamples
    replicate indices, preserving common random numbers across OX2R levels.
    """
    rng = np.random.RandomState(seed); nr = len(da[oxs[0]]); X = np.array(oxs)
    def fqp(m):
        A = np.column_stack([np.ones_like(X), X, X**2]); c = np.linalg.lstsq(A, m, rcond=None)[0]
        return np.clip(-c[1]/(2*c[2]), oxs[0], oxs[-1]) if c[2] < 0 else (
               oxs[-1] if c[1] > 0 else oxs[0])
    pks = np.array([fqp(np.array([np.mean(da[ox][rng.choice(nr, nr, True)]) for ox in oxs]))
                    for _ in range(n_boot)])
    return {'peaks': pks, 'mean': np.mean(pks),
            'ci_95': [np.percentile(pks, 2.5), np.percentile(pks, 97.5)]}


def bootstrap_peak_location(da, oxs, n_boot=5000, seed=42):
    """Fraction of bootstrap argmax locations that are strictly interior."""
    rng = np.random.RandomState(seed); nr = len(da[oxs[0]]); interior = 0
    for _ in range(n_boot):
        idx = rng.choice(nr, nr, True); bm = {ox: np.mean(da[ox][idx]) for ox in oxs}
        pi  = oxs.index(max(bm, key=bm.get))
        if 0 < pi < len(oxs) - 1: interior += 1
    return {'p_interior': interior / n_boot}


def validate_peak_shift(cpb, sls):
    """Slope of the per-boot continuity-peak location vs. depth level ``sls``."""
    nb = len(cpb[sls[0]]['peaks'])
    slopes = np.array([np.polyfit(sls, [cpb[s]['peaks'][b] for s in sls], 1)[0] for b in range(nb)])
    return {'slope_mean': np.mean(slopes),
            'slope_ci_95': [np.percentile(slopes, 2.5), np.percentile(slopes, 97.5)],
            'p_slope_negative': np.mean(slopes < 0)}


def limb_decomposition(full_da, frozen_da, oxs, split=1.0, n_boot=5000, seed=42):
    """
    Ascending and descending limb amplitudes for the full and frozen arms,
    with the difference between arms bootstrapped over replicates.

    Limb amplitudes are DIFFERENCES within an arm, so any constant vertical
    offset between arms cancels.  Bootstrap resamples replicate indices, which
    preserves common random numbers across OX2R levels and across arms.
    """
    lo, hi = oxs[0], oxs[-1]
    nr  = len(full_da[lo])
    rng = np.random.RandomState(seed)

    out = {'asc_full': [], 'asc_froz': [], 'asc_diff': [],
           'des_full': [], 'des_froz': [], 'des_diff': []}
    for _ in range(n_boot):
        i  = rng.choice(nr, nr, True)
        af = full_da[split][i].mean()   - full_da[lo][i].mean()
        az = frozen_da[split][i].mean() - frozen_da[lo][i].mean()
        df = full_da[hi][i].mean()      - full_da[split][i].mean()
        dz = frozen_da[hi][i].mean()    - frozen_da[split][i].mean()
        out['asc_full'].append(af); out['asc_froz'].append(az)
        out['asc_diff'].append(af - az)
        out['des_full'].append(df); out['des_froz'].append(dz)
        out['des_diff'].append(df - dz)

    def _s(v):
        v = np.asarray(v)
        return {'mean': float(v.mean()),
                'ci_95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                'p_pos': float((v > 0).mean())}
    return {k: _s(v) for k, v in out.items()}


def test_complexity_interaction(cpx_da, sls, oxs, n_boot=5000, seed=42, l_values=None):
    """
    Censoring-free test of a depth-dependent peak shift.

    Fits, on condition means, with the depth axis L mean-centred:

        DI = b0 + b1*OX + b2*OX^2 + b3*L + b4*(OX*L) + b5*(OX^2*L)

    Concavity requires b2 < 0.  Given b2 < 0 the vertex is
        OX*(L) = -(b1 + b4*L) / (2*(b2 + b5*L))
    and its derivative at the mean depth is
        dOX*/dL = (b1*b5 - b2*b4) / (2*b2^2)
    reported as the implied peak shift in OX2R units per unit of L.

    `sls` indexes cpx_da; `l_values` supplies the numeric axis when the two
    differ (e.g. divergence fractions as keys, divergence STEPS as the axis).
    Passing steps for both the sequence-length and divergence sweeps puts the
    two implied shifts in the same units.

    Bootstrap resamples REPLICATE INDICES, which preserves common random
    numbers across depth levels and OX2R levels.
    """
    L  = np.array(sls if l_values is None else l_values, dtype=float)
    Lc = L - L.mean()
    nr = len(cpx_da[sls[0]][oxs[0]])

    rows, cells = [], []
    for i, sl in enumerate(sls):
        for ox in oxs:
            rows.append([1.0, ox, ox ** 2, Lc[i], ox * Lc[i], (ox ** 2) * Lc[i]])
            cells.append((sl, ox))
    X     = np.array(rows, dtype=float)
    X_pin = np.linalg.pinv(X)                 # 6 x n_conditions

    rng    = np.random.RandomState(seed)
    B      = np.empty((n_boot, 6))
    shifts = np.empty(n_boot)
    for b in range(n_boot):
        idx  = rng.choice(nr, nr, True)       # CRN-preserving
        y    = np.array([cpx_da[sl][ox][idx].mean() for (sl, ox) in cells])
        beta = X_pin @ y
        B[b] = beta
        b1, b2, b4, b5 = beta[1], beta[2], beta[4], beta[5]
        shifts[b] = (b1 * b5 - b2 * b4) / (2 * b2 ** 2) if abs(b2) > 1e-12 else np.nan

    def _s(v):
        v = v[np.isfinite(v)]
        return {'mean': float(np.mean(v)),
                'ci_95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                'p_neg': float(np.mean(v < 0))}

    return {'beta_names':     ['int', 'OX', 'OX2', 'L', 'OX:L', 'OX2:L'],
            'beta_mean':      B.mean(axis=0).tolist(),
            'beta_ci_95':     [[float(np.percentile(B[:, j], 2.5)),
                                float(np.percentile(B[:, j], 97.5))] for j in range(6)],
            'quadratic':      _s(B[:, 2]),    # b2, must be < 0
            'ox_by_length':   _s(B[:, 4]),    # b4, the interaction of interest
            'implied_shift':  _s(shifts),     # OX2R units per unit of L
            'n_boot':         n_boot}


def dist_stats(e, s):
    """KS distance + p-value, EMD and means over two sample arrays."""
    e = e[np.isfinite(e)]; s = s[np.isfinite(s)]
    if len(e) == 0 or len(s) == 0:
        return None
    ks, p = ks_2samp(e, s)
    return {"ks": ks, "p": p, "emd": wasserstein_distance(e, s),
            "me": np.mean(e), "ms": np.mean(s)}


__all__ = [
    "test_curvature",
    "test_monotonicity",
    "bootstrap_continuous_peak",
    "bootstrap_peak_location",
    "validate_peak_shift",
    "limb_decomposition",
    "test_complexity_interaction",
    "dist_stats",
]