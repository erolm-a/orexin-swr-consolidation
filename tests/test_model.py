"""
Model smoke tests and diagnostic checks.

Reads:  results/agent_model_results.pkl
Writes: figures/figure_S_learning_rule.{png,pdf}

Run after stage 3 to confirm the pipeline reproduces the reported
values before regenerating figures.

Paths and artefact versions are resolved via orexin_paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orexin_paths import (  # noqa: E402
    DIRS, FILES, PROJECT_ROOT, PIPELINE_VERSION,
    ensure_dirs, load_artifact,
)

ensure_dirs()

# =============================================================================
# learning_rule_diagnostics_v3.py
#
# Run after learning_rule_control_addendum.py (`out`, `OXL`, `OXS`, `TASK`,
# `NREPS`, `SangerReadout` must exist). Supersedes v2.
#
#   [1] CHANCE BASELINE     what alignment does an untrained readout give?
#   [2] STATIONARY POINTS   sign-checked quadratic vertices, features labelled
#   [3] SANGER DIAGNOSIS    what did the two components converge to?
#   [4] OJA TROUGH          is it mask heterogeneity? (replaces the v2
#                           "common-cause test", which framed the trough as a
#                           mirror image of the consolidation optimum. That
#                           framing is withdrawn: mask variance and induction
#                           mass are both driven by the same truncation
#                           distribution, so a coincidence of locations carries
#                           no information about mechanism.)
#
# For [4b], add to the dict returned by RateConsolidationAgent.consolidate(),
# beside 'mean_unique_frac' (~line 4849):     'sd_unique_frac': float(np.std(ufa)),
# and in _empty() (~line 4877):               'sd_unique_frac': np.nan,
# =============================================================================
import numpy as np, pickle
from pathlib import Path

# =============================================================================
# [1] chance alignment for a random unit vector in d dimensions
# =============================================================================
D_DIM = RATE_DIM                                                       # noqa: F821
chance = float(np.sqrt(2.0 / (np.pi * D_DIM)))
chance_sd = chance * float(np.sqrt(np.pi / 2.0 - 1.0))

_r = np.random.RandomState(0)
_W = _r.randn(20000, D_DIM); _U = _r.randn(20000, D_DIM)
_emp = np.abs(np.einsum('ij,ij->i', _W, _U)) / (
    np.linalg.norm(_W, axis=1) * np.linalg.norm(_U, axis=1))

print("=" * 78)
print("[1] CHANCE ALIGNMENT BASELINE")
print("=" * 78)
print(f"  dim = {D_DIM}")
print(f"  analytic E|cos| = {chance:.4f}   (SD {chance_sd:.4f})")
print(f"  empirical       = {_emp.mean():.4f}   95% range for ONE draw "
      f"[{np.percentile(_emp, 2.5):.4f}, {np.percentile(_emp, 97.5):.4f}]")
_n = len(out['delta']['align'][OXL[0]])                                # noqa: F821
_lo = chance - 1.96 * chance_sd / np.sqrt(_n)
_hi = chance + 1.96 * chance_sd / np.sqrt(_n)
print(f"  null band for an n={_n} MEAN = [{_lo:.4f}, {_hi:.4f}]  <- the relevant one")
print(f"\n  {'rule':<10}{'min ALIGN':<12}{'max ALIGN':<12}{'max / chance':<14}{'inside null?'}")
for rule in ['delta', 'hebb', 'oja', 'sanger']:
    m = np.array([np.nanmean(out[rule]['align'][ox]) for ox in OXL])   # noqa: F821
    inside = "YES — at chance" if (m.min() >= _lo and m.max() <= _hi) else "no"
    print(f"  {rule:<10}{m.min():<12.4f}{m.max():<12.4f}{m.max()/chance:<14.2f}{inside}")
print("  -> a rule whose whole curve lies inside the null band has not learned")
print("     the discriminative axis, and its curvature is not interpretable.")

# =============================================================================
# [2] stationary-point locations, sign-checked
# =============================================================================
FEATURE = {'delta': 'max', 'hebb': 'max', 'oja': 'min', 'sanger': None}


def _quad_fit(oxs, means):
    """Return (vertex_location, c2). CALLER MUST CHECK THE SIGN:
    c2 < 0 -> the vertex is a maximum;  c2 > 0 -> a minimum."""
    X = np.asarray(oxs, float)
    A = np.column_stack([np.ones_like(X), X, X ** 2])
    c = np.linalg.lstsq(A, np.asarray(means, float), rcond=None)[0]
    if abs(c[2]) < 1e-12:
        return np.nan, 0.0
    return float(np.clip(-c[1] / (2 * c[2]), X[0], X[-1])), float(c[2])


def _ok(c2, feature):
    return (c2 < 0) if feature == 'max' else (c2 > 0)


def loc_difference(rule_a, rule_b, metric='align', n_boot=5000, seed=42):
    """Paired bootstrap on the difference between two curve features, using the
    SAME resampled seed indices for both rules. Replicates whose fitted
    curvature has the wrong sign are discarded and counted.

    Only like-for-like comparisons belong here (maximum against maximum).
    Comparing Oja's minimum against delta's maximum is not a location offset
    and is no longer computed — see [4]."""
    fa, fb = FEATURE[rule_a], FEATURE[rule_b]
    if fa is None or fb is None:
        raise ValueError(f"{rule_a}/{rule_b}: no declared stationary point")
    if fa != fb:
        raise ValueError(f"{rule_a}({fa}) vs {rule_b}({fb}): not like-for-like")
    a = out[rule_a][metric]; b = out[rule_b][metric]                   # noqa: F821
    nr = len(a[OXL[0]])                                                # noqa: F821
    r = np.random.RandomState(seed); d = []; bad = 0
    for _ in range(n_boot):
        idx = r.choice(nr, nr, True)
        va, ca = _quad_fit(OXL, [a[ox][idx].mean() for ox in OXL])     # noqa: F821
        vb, cb = _quad_fit(OXL, [b[ox][idx].mean() for ox in OXL])     # noqa: F821
        if not (_ok(ca, fa) and _ok(cb, fb)):
            bad += 1
            continue
        d.append(va - vb)
    d = np.asarray(d)
    return (float(d.mean()),
            [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            bad / n_boot)


print()
print("=" * 78)
print("[2] STATIONARY-POINT LOCATIONS (sign-checked, features labelled)")
print("=" * 78)
print(f"  {'rule':<10}{'metric':<9}{'feature':<10}{'vertex':<10}{'curvature'}")
for rule in ['delta', 'hebb', 'oja']:
    for metric, key in [('di', 'stats_DI'), ('align', 'stats_ALIGN')]:
        mns = [np.nanmean(out[rule][metric][ox]) for ox in OXL]        # noqa: F821
        v, c2 = _quad_fit(OXL, mns)                                    # noqa: F821
        out[rule][key]['vertex'] = v                                   # noqa: F821
        out[rule][key]['feature'] = FEATURE[rule]                      # noqa: F821
        print(f"  {rule:<10}{metric:<9}{FEATURE[rule]:<10}{v:<10.3f}"
              f"{'concave-up' if c2 > 0 else 'concave-down'}")

# Oja's stored 'peak' is an argmax pinned at the sweep boundary, a DIFFERENT
# statistic from the vertex above. Blank it so it cannot reach a caption.
for key in ['stats_DI', 'stats_ALIGN']:
    st = out['oja'].get(key)                                           # noqa: F821
    if st:
        st['trough'] = st['vertex']
        st['peak'] = np.nan
        st['peak_ci'] = [np.nan, np.nan]
print("\n  oja 'peak' set to nan: the stored 0.30 was an argmax pinned at the")
print("  sweep boundary, not the vertex. Use ['trough'].")

print(f"\n  {'like-for-like comparison':<28}{'difference':<12}{'95% CI':<24}{'discarded'}")
for metric in ['di', 'align']:
    m, ci, bad = loc_difference('hebb', 'delta', metric=metric)
    lbl = f"hebb(max) - delta(max) [{metric}]"
    zero = "contains 0" if ci[0] <= 0 <= ci[1] else "excludes 0"
    print(f"  {lbl:<28}{m:<+12.3f}[{ci[0]:+.3f}, {ci[1]:+.3f}]  {zero:<12}{bad:.1%}")
print(f"  grid spacing near the optimum = {float(OXL[5]) - float(OXL[4]):.2f} OX2R units")  # noqa: F821
print("  Judge these magnitudes against the grid spacing and against the")
print("  complexity-driven peak shift in Fig. 5A.")

# =============================================================================
# [3] what did Sanger's two components actually converge to?
# =============================================================================
def _cos(u, v):
    return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


print()
print("=" * 78)
print("[3] SANGER COMPONENT DIAGNOSIS (single seed, OX2R = 1.0)")
print("=" * 78)
seed = 0
ma, mb = RateMemory.create_pair(TASK.seq_len, 16, TASK.overlap,        # noqa: F821
                                TASK.divergence_point, seed)           # noqa: F821
ti = _sample_crn(np.random.RandomState(seed + 123),                    # noqa: F821
                 N_TRIAL_BLOCKS,                                       # noqa: F821
                 len(_get_trials(SWP_LIBRARY, MODEL_NAME_LIB, CONTEXT_LIB, 1.0)))  # noqa: F821
plan = build_replay_plan_ox(1.0, TASK.seq_len, 16,                     # noqa: F821
                            np.random.RandomState(seed + 123),
                            np.random.RandomState(seed + 500), ti)
ag = _make_agent(TASK.seq_len, seed=seed)                              # noqa: F821
ag.readout = SangerReadout(ag.dim, lr=ag.base_lr, seed=seed)           # noqa: F821
ag.encode(ma, TASK.enc_str, TASK.enc_trials, ox2r=1.0)                 # noqa: F821
ag.encode(mb, TASK.enc_str * 0.3, max(1, TASK.enc_trials // 3), ox2r=1.0)  # noqa: F821
ag.consolidate(plan, 1.0, CORE_SWITCHES, diverge_idx=DIVERGE_IDX)      # noqa: F821

shared = np.zeros_like(ma.vector); shared[:ma.n_shared] = ma.vector[:ma.n_shared]
diff = ma.vector - mb.vector
print(f"  {'':<14}{'shared axis':<14}{'x_A':<10}{'x_B':<10}{'A - B axis'}")
for i, lbl in enumerate(['PC1 (W[0])', 'PC2 (W[1])']):
    w = ag.readout.W[i]
    print(f"  {lbl:<14}{_cos(w, shared):<14.3f}{_cos(w, ma.vector):<10.3f}"
          f"{_cos(w, mb.vector):<10.3f}{_cos(w, diff):.3f}")
print(f"  chance for any of these = {chance:.3f}")
print(f"  sweep-mean PC2 alignment at OX2R=1.0 = "
      f"{np.nanmean(out['sanger']['align'][OXL[4]]):.3f}")               # noqa: F821
print("  -> ONE SEED. Illustrative of the structure; do not quote these values.")
print("     PC1 tracking x_A more closely than the shared axis means the")
print("     replayed ensemble is dominated by the reinforced memory, so PC2 is")
print("     a residual direction rather than the discriminative contrast.")

# =============================================================================
# [4] is the Oja trough mask heterogeneity?
#
# Oja converges to PC1 of the replayed ensemble. Variance there comes from
# memory identity and from mask variation. Mask variance follows a p(1-p) shape
# in the fraction of events truncated before divergence: at low OX2R almost all
# events are complete, at high OX2R almost all are cut short, and both extremes
# give a homogeneous mask. Where mask variance dominates, PC1 tracks the mask,
# which is close to the shared prefix and so near-orthogonal to the A-B axis.
# =============================================================================
print()
print("=" * 78)
print("[4] OJA TROUGH — MASK-HETEROGENEITY TEST")
print("=" * 78)
RES = DIRS['results']
with open(FILES['results'], 'rb') as f:
    DMAIN = pickle.load(f)
OJA_TROUGH = float(out['oja']['stats_ALIGN']['trough'])                # noqa: F821

# ---- [4a] zero-cost proxy: where does p_pre_divergence cross 0.5? ----------
print("  [4a] proxy from p_pre_divergence (no re-run)")
key = next((k for k in ('p_pre_divergence', 'p_prediv_by_ox', 'p_prediv')
            if k in DMAIN), None)
if key is None:
    print("       p_pre_divergence not in the pickle — skip to [4b].")
    p_cross = np.nan
else:
    pd_ = DMAIN[key]
    ks = sorted(pd_.keys(), key=float)
    xs = np.array([float(k) for k in ks])
    ps = np.array([float(np.mean(pd_[k])) if np.ndim(pd_[k]) else float(pd_[k])
                   for k in ks])
    for x, p in zip(xs, ps):
        print(f"       OX2R {x:<6.2f} p = {p:<8.4f} p(1-p) = {p * (1 - p):.4f}")
    if ps.min() <= 0.5 <= ps.max():
        o = np.argsort(ps)
        p_cross = float(np.interp(0.5, ps[o], xs[o]))
        print(f"       p = 0.5 crossing at OX2R = {p_cross:.3f}")
    else:
        p_cross = float(xs[np.argmax(ps * (1 - ps))])
        print(f"       p never reaches 0.5; p(1-p) maximal at OX2R = {p_cross:.3f}")

# ---- [4b] direct: across-event SD of unique_frac by OX2R -------------------
print("\n  [4b] direct check — across-event SD of unique_frac")
N_SD = min(NREPS, 15)                                                  # noqa: F821
_probe = _make_agent(TASK.seq_len, seed=0)                             # noqa: F821
_ti = _sample_crn(np.random.RandomState(123), N_TRIAL_BLOCKS,          # noqa: F821
                  len(_get_trials(SWP_LIBRARY, MODEL_NAME_LIB, CONTEXT_LIB, 1.0)))  # noqa: F821
_pl = build_replay_plan_ox(1.0, TASK.seq_len, 16,                      # noqa: F821
                           np.random.RandomState(123),
                           np.random.RandomState(500), _ti)
_probe.encode(ma, TASK.enc_str, TASK.enc_trials, ox2r=1.0)             # noqa: F821
_o0 = _probe.consolidate(_pl, 1.0, CORE_SWITCHES, diverge_idx=DIVERGE_IDX)  # noqa: F821

if 'sd_unique_frac' not in _o0:
    print("       'sd_unique_frac' not in consolidate() output. Add the line")
    print("       shown in this file's header, re-run the model cells, repeat.")
    sd_vertex = np.nan
else:
    sd_by_ox = {ox: [] for ox in OXL}                                  # noqa: F821
    for rep in range(N_SD):
        sd = rep * 1000
        _a, _b = RateMemory.create_pair(TASK.seq_len, 16, TASK.overlap,   # noqa: F821
                                        TASK.divergence_point, sd)        # noqa: F821
        _t = _sample_crn(np.random.RandomState(sd + 123), N_TRIAL_BLOCKS,  # noqa: F821
                         len(_get_trials(SWP_LIBRARY, MODEL_NAME_LIB,      # noqa: F821
                                         CONTEXT_LIB, 1.0)))
        for ox in OXL:                                                 # noqa: F821
            pl = build_replay_plan_ox(ox, TASK.seq_len, 16,            # noqa: F821
                                      np.random.RandomState(sd + 123),
                                      np.random.RandomState(sd + 500), _t)
            g = _make_agent(TASK.seq_len, seed=sd)                     # noqa: F821
            g.encode(_a, TASK.enc_str, TASK.enc_trials, ox2r=1.0)      # noqa: F821
            g.encode(_b, TASK.enc_str * 0.3,                           # noqa: F821
                     max(1, TASK.enc_trials // 3), ox2r=1.0)           # noqa: F821
            c = g.consolidate(pl, ox, CORE_SWITCHES, diverge_idx=DIVERGE_IDX)  # noqa: F821
            sd_by_ox[ox].append(c['sd_unique_frac'])
    mns = [float(np.nanmean(sd_by_ox[ox])) for ox in OXL]              # noqa: F821
    sd_vertex, c2 = _quad_fit(OXL, mns)                                # noqa: F821
    for ox, m in zip(OXL, mns):                                        # noqa: F821
        print(f"       OX2R {float(ox):<6.2f} SD(unique_frac) = {m:.4f}")
    print(f"       vertex = {sd_vertex:.3f}  "
          f"({'maximum' if c2 < 0 else 'MINIMUM — hypothesis not supported'})")

# ---- [4c] can the data tell the two accounts apart? -----------------------
print("\n  [4c] discriminator")
mass_vertex = np.nan
if 'mass_by_ox' in DMAIN:
    mox = DMAIN['mass_by_ox']
    mk = sorted(mox.keys(), key=float)
    mv = [float(np.mean(mox[k])) if np.ndim(mox[k]) else float(mox[k]) for k in mk]
    mass_vertex, _ = _quad_fit([float(k) for k in mk], mv)
print(f"       {'Oja alignment trough':<34}{OJA_TROUGH:.3f}")
print(f"       {'mask-variance max (proxy)':<34}{p_cross:.3f}")
print(f"       {'mask-variance max (direct)':<34}{sd_vertex:.3f}")
print(f"       {'induction-mass max':<34}{mass_vertex:.3f}")
cands = [v for v in (p_cross, sd_vertex) if np.isfinite(v)]
grid = float(OXL[5]) - float(OXL[4])                                   # noqa: F821
if cands and np.isfinite(mass_vertex):
    sep = abs(float(np.mean(cands)) - mass_vertex)
    print(f"\n       separation between the two accounts = {sep:.3f} "
          f"({sep / grid:.2f} grid steps)")
    if sep < grid:
        print("       -> NOT SEPARABLE. Mask variance and induction mass are")
        print("          both driven by the same truncation distribution, so a")
        print("          coincidence of locations carries no information about")
        print("          mechanism. Report the trough as a mask-heterogeneity")
        print("          artifact in one caption sentence, do not interpret it")
        print("          further, and make no mirror-image claim.")
    else:
        print("       -> SEPARABLE. Whichever lands with the Oja trough is the")
        print("          operative account; state it and cite this comparison.")
else:
    print("\n       insufficient inputs to run the discriminator.")

# =============================================================================
# figure_S_learning_rule_final.py  (v2)
#
# Run after learning_rule_diagnostics_v2.py (needs `out`, `OXL`, `OXS`,
# `loc_difference`, `_quad_fit`, `FEATURE`, `chance`, `chance_sd`).
#
# Changes from v1:
#   - caption table reports only the rules actually plotted in each panel
#   - the two near-coincident vertical lines in A become one shaded band with a
#     bracket labelled by the offset and its CI
#   - a sign check decides whether Oja's DI is interpretable and therefore
#     whether it belongs in panel A at all
# =============================================================================
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

NREP = len(out['delta']['align'][OXL[0]])                              # noqa: F821
null_lo = chance - 1.96 * chance_sd / np.sqrt(NREP)                    # noqa: F821
null_hi = chance + 1.96 * chance_sd / np.sqrt(NREP)                    # noqa: F821

# ---------------------------------------------------------------------------
# SIGN CHECK — is DI interpretable for the PC-based rules?
# A principal component's sign is undetermined in principle, so DI under oja
# and sanger should be bimodal across seeds. If it is NOT, the sign is being
# pinned by something in the encoding phase and DI is interpretable after all,
# in which case oja belongs in panel A. Decide from this, not from theory.
# ---------------------------------------------------------------------------
print("=" * 78)
print("SIGN CHECK — fraction of seeds with DI > 0, by rule and OX2R")
print("=" * 78)
print(f"  {'rule':<10}" + "".join(f"{float(ox):<7.2f}" for ox in OXL))     # noqa: F821
frac_pos = {}
for rule in ['delta', 'hebb', 'oja', 'sanger']:
    f = [float(np.mean(np.asarray(out[rule]['di'][ox]) > 0)) for ox in OXL]  # noqa: F821
    frac_pos[rule] = f
    print(f"  {rule:<10}" + "".join(f"{v:<7.2f}" for v in f))
OJA_DI_OK = min(frac_pos['oja']) > 0.9 or max(frac_pos['oja']) < 0.1
print(f"\n  oja DI sign consistent across seeds: {OJA_DI_OK}")
print("  -> True  : sign is pinned by encoding; plot oja in panel A.")
print("  -> False : bimodal, DI is not defined in the same sense; keep it out")
print("             of panel A and say so in the caption.")
PANEL_A = ['delta', 'hebb'] + (['oja'] if OJA_DI_OK else [])

# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
COL = {'delta': '#9aa0a6', 'hebb': '#3b6ea5', 'oja': '#b1543a', 'sanger': '#3f8f6b'}
LBL = {'delta':  'Delta rule (supervised)',
       'hebb':   'Reward-modulated Hebbian (global scalar)',
       'oja':    'Oja (unsupervised, PC1)',
       'sanger': 'Sanger (unsupervised, PC2)'}
C_T, C_REF = '#1c1917', '#9aa0a6'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.linewidth': 0.8, 'axes.labelsize': 9.5,
    'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})


def band(ax, rule, metric):
    d = out[rule][metric]                                              # noqa: F821
    m = np.array([np.nanmean(d[ox]) for ox in OXL])                    # noqa: F821
    e = np.array([np.nanstd(d[ox], ddof=1) / np.sqrt(len(d[ox])) for ox in OXL])  # noqa: F821
    ax.fill_between(OXS, m - 1.96 * e, m + 1.96 * e, color=COL[rule],   # noqa: F821
                    alpha=0.13, lw=0, zorder=1)
    ax.plot(OXS, m, '-o', color=COL[rule], lw=1.8, ms=4.5, mfc='white',  # noqa: F821
            mew=1.3, zorder=3, label=LBL[rule])
    return m


def plabel(ax, s):
    ax.text(-0.15, 1.06, s, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', color=C_T)


fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), facecolor='white')

# ── A: discrimination index ────────────────────────────────────────────────
ax = axes[0]; plabel(ax, 'A')
for rule in PANEL_A:
    band(ax, rule, 'di')

# one shaded band spanning both optima, with a bracket labelled by the offset
vx = {}
for rule in ['delta', 'hebb']:
    mns = [np.nanmean(out[rule]['di'][ox]) for ox in OXL]              # noqa: F821
    vx[rule], _ = _quad_fit(OXL, mns)                                  # noqa: F821
dm, dci, _ = loc_difference('hebb', 'delta', metric='di')              # noqa: F821
xa, xb = sorted(vx.values())
ax.axvspan(xa, xb, color=C_REF, alpha=0.25, lw=0, zorder=0)
y0, y1 = ax.get_ylim(); span = y1 - y0
yb = y1 - 0.10 * span; h = 0.022 * span
ax.plot([xa, xa, xb, xb], [yb - h, yb, yb, yb - h], color=C_T, lw=0.9,
        zorder=4, clip_on=False)
ax.annotate(f'optima {xa:.2f} / {xb:.2f}\nΔ = {dm:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]',
            xy=((xa + xb) / 2, yb), xytext=(0.62, 0.93), textcoords='axes fraction',
            fontsize=6.8, color=C_T, ha='left', va='top',
            arrowprops=dict(arrowstyle='-', lw=0.7, color=C_T,
                            shrinkA=0, shrinkB=2))
ax.set_ylim(y0, y1)
ax.set_xlabel('OX2R level'); ax.set_ylabel('Discrimination index')
ax.set_title('Memory outcome', fontsize=10)
ax.legend(fontsize=7, frameon=False, loc='lower center')
ax.spines[['top', 'right']].set_visible(False)

# ── B: alignment against the chance band ───────────────────────────────────
ax = axes[1]; plabel(ax, 'B')
ax.axhspan(null_lo, null_hi, color=C_REF, alpha=0.22, lw=0, zorder=0)
ax.axhline(chance, color=C_REF, lw=0.9, ls='--', zorder=1)             # noqa: F821
for rule in ['delta', 'hebb', 'oja', 'sanger']:
    band(ax, rule, 'align')
ax.text(OXS[-1], null_hi, f' chance (n={NREP})', fontsize=6.8,          # noqa: F821
        color=C_REF, va='bottom', ha='right')
ax.set_xlabel('OX2R level'); ax.set_ylabel('Alignment with A–B axis')
ax.set_title('Discriminative information reaching the readout', fontsize=10)
ax.set_ylim(0, None)
ax.legend(fontsize=6.6, frameon=False, loc='center right')
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Supplementary Figure — the optimum does not require a supervised teaching signal',
             fontsize=11.5, fontweight='bold', y=1.02, color=C_T)
fig.tight_layout()

OUTDIR = DIRS['figures']
OUTDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTDIR / 'figure_S_learning_rule.png', dpi=300, facecolor='white',
            bbox_inches='tight')
fig.savefig(OUTDIR / 'figure_S_learning_rule.pdf', facecolor='white',
            bbox_inches='tight')
plt.show(); plt.close(fig)

# ---------------------------------------------------------------------------
# CAPTION NUMBERS — only what is plotted, under its own correct label
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("CAPTION NUMBERS")
print("=" * 78)
print(f"  PANEL A (DI):     {', '.join(PANEL_A)}")
print(f"  {'rule':<8}{'curvature':<12}{'P(c>=0)':<10}{'optimum'}")
for rule in PANEL_A:
    st = out[rule]['stats_DI']                                          # noqa: F821
    print(f"  {rule:<8}{st['curvature']:<+12.4f}{st['p_nonneg']:<10.4f}{vx.get(rule, np.nan):.2f}")
print(f"  hebb - delta optimum offset = {dm:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]")
if not OJA_DI_OK:
    print("  omitted from panel A and from this table: oja, sanger "
          "(DI not defined in the same sense)")

print(f"\n  PANEL B (alignment): all four rules")
print(f"  {'rule':<8}{'feature':<9}{'vertex':<9}{'curvature':<12}{'P(c>=0)'}")
for rule in ['delta', 'hebb', 'oja', 'sanger']:
    st = out[rule]['stats_ALIGN']                                       # noqa: F821
    feat = st.get('feature') or 'none'
    vtx = st.get('vertex', np.nan)
    print(f"  {rule:<8}{feat:<9}{vtx:<9.3f}{st['curvature']:<+12.4f}{st['p_nonneg']:.4f}")
print(f"  chance = {chance:.4f}; null band for n={NREP} mean = "                 # noqa: F821
      f"[{null_lo:.4f}, {null_hi:.4f}]")