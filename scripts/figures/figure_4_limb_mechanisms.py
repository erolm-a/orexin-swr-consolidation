"""
Figure 4 - truncation and plasticity gating act on opposite limbs.

Reads:  results/agent_model_results.pkl
Writes: figures/figure_4.{png,pdf,tiff}

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

from orexin_paths import DIRS, FILES, ensure_dirs  # noqa: E402

ensure_dirs()

"""
Figure 4: Replay truncation and plasticity gating act on opposite
limbs of the memory curve.                                    [revised v4]

  A. Replay duration and truncation exposure   — MODEL OUTPUT
  B. Planned event count and LTP drive gate    — MODEL OUTPUT
  C. DI curves, full model vs frozen-gate      — MODEL OUTPUT
  D. Limb amplitudes with bootstrap CIs        — MODEL OUTPUT

The two reference lines in A are derived from the task constants rather than
hard-coded, so they cannot drift away from Supplementary Figure 1. With
RATE_DIM 64, SEQ_LEN 16 and TASK_DIVERGENCE 0.6 the model diverges at step 9.
See the note under TASK CONSTANTS.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import pickle

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
    from IPython.display import display
except Exception:
    def display(obj): pass

# ── Load ───────────────────────────────────────────────────────────────────
PKL = FILES['results']
with open(PKL, 'rb') as f:
    D = pickle.load(f)

OX      = D['ox2r_levels']
ox_arr  = np.array(OX, float)
DI_ARR  = D['ox2r_di_arrays']
M       = D['main_mechanism_by_ox']
FL      = D['frozen_ltp_gate']

# ── TASK CONSTANTS ─────────────────────────────────────────────────────────
# Read from the pickle when the model saves them, otherwise the model
# defaults. DIVERGE_STEP is computed by the same expression the model uses, so
# the reference line in panel A cannot drift away from Supplementary Figure 1.
_ts = D.get('task_structure', {}) if isinstance(D.get('task_structure'), dict) else {}
RATE_DIM = int(_ts.get('rate_dim', 64))
SEQ_LEN  = int(_ts.get('seq_len', 16))
TASK_DIV = float(_ts.get('task_divergence', 0.6))
N_SHARED     = int(RATE_DIM * TASK_DIV)
DIVERGE_STEP = max(1, int(SEQ_LEN * N_SHARED / RATE_DIM))
FULL_REPLAY  = SEQ_LEN

for k in ('t_stop', 't_stop_p10', 't_stop_p90', 'drive_gate', 'quality_gate'):
    if k not in M[OX[0]]:
        raise KeyError(f"main_mechanism_by_ox is missing '{k}'. Re-run the model "
                       f"cell with the integrated save block — the figure script "
                       f"will not re-simulate.")
if 'limb_decomposition' not in FL:
    raise KeyError("frozen_ltp_gate is missing 'limb_decomposition'. Add "
                   "limb_decomposition(ox_da, frozen_da, OX_MAIN, split=1.0) to "
                   "the model cell and re-dump.")

LD    = FL['limb_decomposition']
SPLIT = 1.0

_fg = FL.get('frozen_gate_value', None)
if _fg is not None and np.isfinite(_fg):
    FROZ_LABEL = f'OX2R drive gate frozen at {_fg:.2f}'
else:
    FROZ_LABEL = 'OX2R drive gate frozen at baseline'
    print("  ⚠ 'frozen_gate_value' not in pickle — add "
          "\"'frozen_gate_value': float(FROZEN_LTP_GATE),\" to the "
          "frozen_ltp_gate save dict and re-dump.")


# ── Aggregation ────────────────────────────────────────────────────────────
def _boot_ci(vals, n_boot=5000, seed=42):
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(a) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    bs = np.array([a[rng.choice(len(a), len(a), True)].mean() for _ in range(n_boot)])
    return a.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def series(key):
    out = np.array([_boot_ci(M[ox][key]) for ox in OX])
    return out[:, 0], out[:, 1], out[:, 2]


def mean_sem(arrs):
    m = np.array([np.mean(arrs[ox]) for ox in OX])
    s = np.array([np.std(arrs[ox], ddof=1) / np.sqrt(len(arrs[ox])) for ox in OX])
    return m, s


quality_m, _, _ = series('quality_gate')
trunc_exp       = 1.0 - quality_m
n_plan_m, _, _  = series('n_plan_events')
dg_arr, dg_lo, dg_hi = series('drive_gate')

# mean replay length and its uncertainty, kept separate from event-level dispersion
ts_arr, ts_cilo, ts_cihi = series('t_stop')
ts_p10 = np.array([np.nanmean(M[ox]['t_stop_p10']) for ox in OX])
ts_p90 = np.array([np.nanmean(M[ox]['t_stop_p90']) for ox in OX])
ts_reps = [np.asarray([v for v in M[ox]['t_stop'] if np.isfinite(v)], float)
           for ox in OX]

di_full, se_full = mean_sem(DI_ARR)
di_froz, se_froz = mean_sem(FL['di_arrays'])

# ── Style (matches Figure 2) ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11, 'axes.linewidth': 0.9, 'axes.labelsize': 11.5,
    'axes.titlesize': 12, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'legend.fontsize': 9, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
FS_OXTICK = 9.5     # nine non-uniform OX2R ticks; larger would collide near 1.0
FS_NOTE   = 8.5     # reference-line labels, limb annotations
FS_LEG    = 9.0

# Panel titles: left-aligned bold above each panel, as in Figure 3.
TITLE_LOC, TITLE_BOLD, TITLE_SIZE, TITLE_PAD = 'left', True, 12, 8
TITLE_LINE_H = 0.057        # axes fraction per extra title line, for the letter

# Figure 1-2 palette
FIG2_EMP, FIG2_SYN, FIG2_SYND = '#e05a4b', '#5b9bd5', '#1a5fa8'
C_STEPS  = FIG2_SYND          # replay length          (blue,  solid, circles)
C_TRUNC  = FIG2_EMP           # truncation exposure    (red,   dashed, squares)
C_COUNT  = FIG2_SYN           # planned event count    (light blue bars)
C_COUNT_E = FIG2_SYND
C_DRIVE  = FIG2_EMP           # LTP drive gate         (red,   solid, circles)
C_FULL   = FIG2_EMP           # full model             (red,   solid, circles)
C_FROZ   = FIG2_SYND          # frozen gate            (blue,  dashed, squares)
C_BL     = '#9aa0a6'
C_T      = '#1c1917'
BAND_ALPHA = 0.18
DISP_ALPHA = 0.10


_LADDER = (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10)


def _tickset(lo, hi, k):
    """Smallest rounded step giving k intervals that cover [lo, hi].
    Returns (ticks, waste); waste is the empty span added below and above."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.linspace(0, 1, k + 1), np.inf
    span = hi - lo
    best = None
    for mag_shift in (-1, 0, 1):
        mag = 10.0 ** (np.floor(np.log10(span / k)) + mag_shift)
        for m in _LADDER:
            step = m * mag
            start = np.floor(lo / step) * step
            if start + k * step >= hi - 1e-9:
                waste = (start + k * step - hi) + (lo - start)
                if best is None or waste < best[1]:
                    best = (start + step * np.arange(k + 1), waste)
                break
    return best if best is not None else (np.linspace(lo, hi, k + 1), np.inf)


def align_twin(axL, axR, kmin=3, kmax=6):
    """Same number of ticks on both y axes, so the gridlines coincide."""
    best_cost, best_t = np.inf, None
    for k in range(kmin, kmax + 1):
        tL, wL = _tickset(*axL.get_ylim(), k)
        tR, wR = _tickset(*axR.get_ylim(), k)
        spanL = max(tL[-1] - tL[0], 1e-12); spanR = max(tR[-1] - tR[0], 1e-12)
        cost = wL / spanL + wR / spanR
        if cost < best_cost:
            best_cost, best_t = cost, (tL, tR)
    tL, tR = best_t
    axL.set_ylim(tL[0], tL[-1]); axL.set_yticks(tL)
    axR.set_ylim(tR[0], tR[-1]); axR.set_yticks(tR)
    axL.grid(True, axis='y', color='0.88', lw=0.6, zorder=0)
    axR.grid(False)


def set_ox_ticks(ax):
    """Every sampled level is a tick: the grid is non-uniform by design."""
    ax.set_xlim(0.15, 1.95)
    ax.set_xticks(ox_arr)
    ax.set_xticklabels([f'{v:g}' for v in ox_arr], fontsize=FS_OXTICK)
    ax.grid(True, axis='x', color='0.92', lw=0.5, zorder=0)


def panel_head(ax, lbl, title, x=-0.17):
    """Bold left-aligned title flush with the left spine, with the panel letter
    set level with the first title line however many lines the title runs to."""
    ax.set_title(title, loc=TITLE_LOC, x=0.0, pad=TITLE_PAD,
                 fontsize=TITLE_SIZE, color=C_T, linespacing=1.3,
                 fontweight='bold' if TITLE_BOLD else 'normal')
    y = 1.075 + TITLE_LINE_H * title.count('\n')
    ax.text(x, y, lbl, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', color=C_T)


fig = plt.figure(figsize=(13.2, 10.6), facecolor='white')
gs  = gridspec.GridSpec(2, 2, wspace=0.46, hspace=0.42,
                        left=0.085, right=0.925, top=0.945, bottom=0.08)

# ── A: replay duration + truncation ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.fill_between(ox_arr, ts_p10, ts_p90, color=C_STEPS, alpha=DISP_ALPHA, lw=0, zorder=1)
ax.plot(ox_arr, ts_p10, color=C_STEPS, lw=0.7, ls=':', alpha=0.6, zorder=1)
ax.plot(ox_arr, ts_p90, color=C_STEPS, lw=0.7, ls=':', alpha=0.6, zorder=1)
ax.fill_between(ox_arr, ts_cilo, ts_cihi, color=C_STEPS, alpha=0.35, lw=0, zorder=2)
_rng_j = np.random.RandomState(7)
for xi, vals in zip(ox_arr, ts_reps):
    if len(vals):
        ax.plot(xi + _rng_j.uniform(-0.022, 0.022, len(vals)), vals, '.',
                color=C_STEPS, ms=2.2, alpha=0.32, zorder=3)
ax.plot(ox_arr, ts_arr, '-o', color=C_STEPS, lw=2.0, ms=5.5,
        mfc='white', mew=1.4, zorder=5, label='Mean replay length')
ax.axhline(FULL_REPLAY, color=C_BL, lw=0.8, ls='--', alpha=0.6, zorder=1)
ax.text(1.93, FULL_REPLAY + 0.2, f'full replay ({FULL_REPLAY})',
        fontsize=FS_NOTE, color=C_BL, ha='right')
ax.axhline(DIVERGE_STEP, color=C_BL, lw=0.8, ls=':', alpha=0.5, zorder=1)
ax.text(1.93, DIVERGE_STEP + 0.2, f'diverge step ({DIVERGE_STEP})',
        fontsize=FS_NOTE, color=C_BL, ha='right')
ax.set_xlabel('OX2R level'); ax.set_ylabel('Replay steps', color=C_STEPS)
panel_head(ax, 'A', 'Replay duration and truncation exposure')
ax.set_ylim(max(0.0, float(np.nanmin(ts_p10)) - 1.0),
            float(np.nanmax(ts_p90)) + 0.6)
ax.tick_params(axis='y', colors=C_STEPS)

ax2 = ax.twinx()
ax2.plot(ox_arr, trunc_exp, '--s', color=C_TRUNC, lw=1.7, ms=5.0,
         mfc='white', mew=1.1, zorder=4, label='Truncation exposure')
ax2.set_ylabel('Truncation exposure', color=C_TRUNC)
ax2.set_ylim(0.0, float(np.nanmax(trunc_exp)) * 1.10)
ax2.tick_params(axis='y', colors=C_TRUNC)
ax.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
set_ox_ticks(ax); align_twin(ax, ax2)

h1, lb1 = ax.get_legend_handles_labels()
h2, lb2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + [Patch(facecolor=C_STEPS, alpha=0.35, edgecolor='none'),
                 Patch(facecolor=C_STEPS, alpha=DISP_ALPHA, edgecolor='none'),
                 Line2D([], [], marker='.', ls='none', color=C_STEPS, alpha=0.5)] + h2,
           lb1 + ['95% CI of mean', 'p10–p90, events',
                  'replicate means'] + lb2,
           loc='center left', bbox_to_anchor=(0.01, 0.40), ncol=1, fontsize=FS_LEG - 1,
           frameon=True, framealpha=0.95, edgecolor='0.85', borderpad=0.4,
           labelspacing=0.35, handlelength=1.3, handletextpad=0.45)

# ── B: event count + drive gate ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.bar(ox_arr, n_plan_m, width=0.075, color=C_COUNT, alpha=0.75,
       edgecolor=C_COUNT_E, linewidth=0.7, label='Planned events', zorder=2)
ax.set_xlabel('OX2R level'); ax.set_ylabel('Planned replay events', color=C_COUNT_E)
ax.set_ylim(0, max(n_plan_m) * 1.10); ax.tick_params(axis='y', colors=C_COUNT_E)
panel_head(ax, 'B', 'Event count and LTP drive gate')

ax2 = ax.twinx()
ax2.fill_between(ox_arr, dg_lo, dg_hi, color=C_DRIVE, alpha=BAND_ALPHA, lw=0, zorder=2)
ax2.plot(ox_arr, dg_arr, '-o', color=C_DRIVE, lw=2.0, ms=5.5,
         mfc='white', mew=1.4, zorder=4, label='LTP drive gate')
ax2.set_ylim(float(np.nanmin(dg_lo)) * 0.96, float(np.nanmax(dg_hi)) * 1.04)
ax2.set_ylabel('LTP drive gate  P(LTP)', color=C_DRIVE)
ax2.tick_params(axis='y', colors=C_DRIVE)
ax.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
set_ox_ticks(ax); align_twin(ax, ax2)

h1, lb1 = ax.get_legend_handles_labels()
h2, lb2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + h2 + [Patch(facecolor=C_DRIVE, alpha=BAND_ALPHA, edgecolor='none')],
           lb1 + lb2 + ['95% CI of the mean'],
           loc='upper left', fontsize=FS_LEG, frameon=True,
           framealpha=0.95, edgecolor='none')

# ── C: full vs frozen DI curves ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.axvspan(ox_arr[0], SPLIT, color='0.94', lw=0, zorder=0)
ax.axvline(SPLIT, color=C_BL, lw=0.8, ls=':', zorder=1)
ax.errorbar(ox_arr, di_full, yerr=se_full, fmt='-o', color=C_FULL, lw=2.0,
            ms=5.5, mfc='white', mew=1.4, capsize=3, elinewidth=1.0,
            zorder=3, label='Full model')
ax.errorbar(ox_arr, di_froz, yerr=se_froz, fmt='--s', color=C_FROZ, lw=1.8,
            ms=5.0, mfc='white', mew=1.2, capsize=3, elinewidth=1.0,
            zorder=3, label=FROZ_LABEL)
_ylo, _yhi = ax.get_ylim()
ax.text((ox_arr[0] + SPLIT) / 2, _ylo + 0.015 * (_yhi - _ylo), 'ascending limb',
        fontsize=FS_NOTE + 0.5, color='0.45', ha='center', va='bottom')
ax.set_xlabel('OX2R level'); ax.set_ylabel('Discrimination index')
panel_head(ax, 'C', 'Freezing the OX2R drive gate prevents the rise')
set_ox_ticks(ax)
ax.grid(True, axis='y', color='0.88', lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='lower center', bbox_to_anchor=(0.64, 0.02), fontsize=FS_LEG,
          frameon=True, framealpha=0.95, edgecolor='none')

# ── D: limb amplitudes ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
limbs = [('asc', f'ascending limb\nOX2R {OX[0]:.2f} → {SPLIT:.2f}'),
         ('des', f'descending limb\nOX2R {SPLIT:.2f} → {OX[-1]:.2f}')]
x, w  = np.arange(2), 0.34
for off, arm, col, lab in [(-w / 2, 'full', C_FULL, 'Full model'),
                           (+w / 2, 'froz', C_FROZ, 'Drive gate frozen')]:
    vals = np.array([LD[f'{k}_{arm}']['mean'] for k, _ in limbs])
    lo   = np.array([LD[f'{k}_{arm}']['ci_95'][0] for k, _ in limbs])
    hi   = np.array([LD[f'{k}_{arm}']['ci_95'][1] for k, _ in limbs])
    ax.bar(x + off, vals, w, color=col, alpha=0.85, edgecolor=col,
           linewidth=0.6, label=lab, zorder=2,
           hatch='' if arm == 'full' else '///')
    ax.errorbar(x + off, vals, yerr=[np.maximum(0, vals - lo),
                                     np.maximum(0, hi - vals)],
                fmt='none', ecolor=C_T, lw=1.1, capsize=3.5, zorder=3)
ax.axhline(0, color=C_T, lw=0.7, zorder=1)

_tops = []
for xi, (k, _) in enumerate(limbs):
    d   = LD[f'{k}_diff']
    top = max(LD[f'{k}_full']['ci_95'][1], LD[f'{k}_froz']['ci_95'][1], 0.0)
    _tops.append(top)
    excl = (d['ci_95'][0] > 0) or (d['ci_95'][1] < 0)
    verdict = 'CI excludes 0' if excl else 'CI includes 0'
    ax.annotate(f"Δ = {d['mean']:+.3f}\n[{d['ci_95'][0]:+.3f}, "
                f"{d['ci_95'][1]:+.3f}]\n{verdict}",
                xy=(xi, top), xytext=(xi, top + 0.002),
                fontsize=FS_NOTE + 0.5, color=C_T, ha='center', va='bottom',
                linespacing=1.25)

_bot = min([LD[f'{k}_{a}']['ci_95'][0] for k, _ in limbs for a in ('full', 'froz')] + [0.0])
_rng = max(_tops) - _bot
ax.set_ylim(_bot - 0.12 * _rng, max(_tops) + 0.62 * _rng)
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in limbs], fontsize=10.5)
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel('Limb amplitude, Δ discrimination index\n(DI units, end − start of limb)')
panel_head(ax, 'D', 'Freezing the OX2R drive gate removes the rise,\nbut not the fall')
ax.grid(True, axis='y', color='0.88', lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', fontsize=FS_LEG, frameon=True,
          framealpha=0.95, edgecolor='none')

out = DIRS['figures']
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'figure_4.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figure_4.pdf', facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figure_4.tiff', dpi=300, facecolor='white', bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'})
display(fig); plt.show(); plt.close(fig)

# ── Caption numbers ────────────────────────────────────────────────────────
print("\nFigure 4 saved (PNG, PDF, TIFF)")
print("CAPTION NUMBERS")
print("-" * 72)
print(f"  OX2R grid          {', '.join(f'{v:g}' for v in ox_arr)}  "
      f"(non-uniform: denser near 1.0)")
print(f"  frozen gate        {FROZ_LABEL}")
print(f"  replicates         n = {len(ts_reps[0])} per level")
print(f"  task structure     {RATE_DIM} dimensions, {SEQ_LEN} steps, "
      f"{N_SHARED} shared; divergence at step {DIVERGE_STEP}")
print(f"\n  A  {'OX2R':>6}{'mean t_stop':>13}{'95% CI of mean':>22}"
      f"{'p10-p90 across replays':>26}")
for i, v in enumerate(ox_arr):
    print(f"     {v:>6g}{ts_arr[i]:>13.2f}"
          f"{f'[{ts_cilo[i]:.2f}, {ts_cihi[i]:.2f}]':>22}"
          f"{f'[{ts_p10[i]:.1f}, {ts_p90[i]:.1f}]':>26}")
print(f"\n  B  truncation exposure {trunc_exp[0]:.3f} at OX2R {ox_arr[0]:g} "
      f"→ {trunc_exp[-1]:.3f} at {ox_arr[-1]:g}")
print(f"     LTP drive gate      {dg_arr[0]:.3f} → {dg_arr[-1]:.3f}")
print(f"     planned events      {n_plan_m[0]:.2f} → {n_plan_m[-1]:.2f}")
print(f"\n  D  ascending  full {LD['asc_full']['mean']:+.4f}   "
      f"frozen {LD['asc_froz']['mean']:+.4f}   "
      f"Δ {LD['asc_diff']['mean']:+.4f} "
      f"[{LD['asc_diff']['ci_95'][0]:+.4f}, {LD['asc_diff']['ci_95'][1]:+.4f}] "
      f"P={LD['asc_diff']['p_pos']:.3f}")
print(f"     descending full {LD['des_full']['mean']:+.4f}   "
      f"frozen {LD['des_froz']['mean']:+.4f}   "
      f"Δ {LD['des_diff']['mean']:+.4f} "
      f"[{LD['des_diff']['ci_95'][0]:+.4f}, {LD['des_diff']['ci_95'][1]:+.4f}] "
      f"P={LD['des_diff']['p_pos']:.3f}")

# Panel C level offset: the frozen arm sits above the full model at every level
# (Jensen's inequality on the thresholded induction function), so panel C
# compares shape; the limb differences in D subtract the offset by construction.
_full_mass = np.array([np.nanmean(M[ox].get('ltp_mass', [np.nan])) for ox in OX], float)
_froz_mass = FL.get('frozen_mass_mean', np.nan)
_offset    = float(np.mean(di_froz - di_full))
print("\n  C  level offset")
if np.all(np.isfinite(_full_mass)) and np.isfinite(_froz_mass):
    print(f"     induction mass, full arm   {np.nanmean(_full_mass):.3f} "
          f"(mean over the {len(OX)} levels)")
    print(f"     induction mass, frozen arm {_froz_mass:.3f}  "
          f"ratio {_froz_mass / np.nanmean(_full_mass):.3f}")
else:
    print("     ⚠ 'ltp_mass' or 'frozen_mass_mean' absent from the pickle.")
print(f"     mean DI offset, frozen minus full: {_offset:+.4f}")
print("-" * 72)

import numpy as np
