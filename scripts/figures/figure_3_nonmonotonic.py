"""
Figure 3 - consolidation produces a non-monotonic orexin dependence.

Reads:  results/agent_model_results.pkl
Writes: figures/figure_3.{png,pdf,tiff}

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
Figure 3: Orexin-dependent consolidation produces a non-monotonic
memory curve that dissociates from encoding.          [revised v3]

  A. Consolidation-dependent memory      — main sweep
  B. Encoding-dominant                   — encoding varies, consolidation fixed
  C. Consolidation-dominant              — consolidation varies, encoding fixed
  D. Coupled pathways                    — both vary together

WHAT CHANGED FROM v2

  PANEL TITLES. Still above each panel, now left-aligned and bold, flush with
  the left spine and sitting beside the panel letter — the house style used in
  Figures 1, 2, 4 and 5. v2 centred them in regular weight, which read as a
  different figure series.

  Set TITLE_BOLD = False or TITLE_LOC = 'center' at the top of the style block
  if a journal template wants otherwise; nothing downstream depends on either.

WHAT CHANGED FROM v1 (unchanged in v3, kept for the record)

  PEAK ANNOTATION.  v1 circled argmax(mean curve) and labelled it "peak
  OX2R = x". That is a single draw on a discrete grid, and it is NOT the
  quantity the caption reports (peak mean with a bootstrap CI and
  P(interior)). A reader would take the circled grid point as the optimum.
  Each non-monotone panel now draws the bootstrap peak estimate as a vertical
  line inside its 95% CI band, with P(interior) printed on the axis. The grid
  argmax is kept as a small open marker and named as such, so the two are
  visibly different objects.

  INTACT-SIGNALLING REFERENCE.  A dotted vertical at OX2R = 1.0 labelled
  "intact" now appears on every non-monotone panel. The claim in the text is
  that the optimum lies within about 15% of intact signalling, not that
  intact signalling is optimal; without the reference line the reader cannot
  see the distance being claimed.

  PANEL D.  The peak of panel C is redrawn faintly in D, so the shift under
  coupling is visible rather than inferred by flipping between panels. That
  shift is the whole content of D.

  PALETTE.  Harmonised with Figures 1-2. Blues are carried over; red is NOT
  used, because red denotes empirical data in Figures 1 and 2 and every curve
  here is model output.

  BASELINE LABEL.  The no-consolidation line is labelled on all four panels;
  in v1 only panel A carried the label while all four drew the line.

  NAMING.  The caption block reports the quadratic coefficient as q2, matching
  the manuscript, while still reading the backward-compatible 'curvature' key.

Test statistics live in the CAPTION, not the panels. The block at the end
prints every number the caption needs, straight from the pickle.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
NCM     = D['no_consolidation_baseline']['mean']

dp      = D['dual_pathway_sweep']
OX_C    = sorted([float(k) for k in dp['fixed_encoding']['means'].keys()])
oxC_arr = np.array(OX_C, float)

INTACT = 1.0          # intact OX2R signalling, the reference the text uses


# ── Bootstrap 95% CI of the MEAN from per-rep DI arrays ────────────────────
def boot_ci(arr_dict, oxs, n_boot=5000, seed=42):
    """Bands are the 95% CI OF THE MEAN over replicates, not replicate spread."""
    rng = np.random.RandomState(seed)
    mean, lo, hi = [], [], []
    for ox in oxs:
        a = np.asarray(arr_dict[ox], float); n = len(a)
        mean.append(a.mean())
        if n > 1:
            bs = np.array([a[rng.choice(n, n, True)].mean() for _ in range(n_boot)])
            lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        else:
            lo.append(a.mean()); hi.append(a.mean())
    return np.array(mean), np.array(lo), np.array(hi)


def band_for(branch_key, oxs):
    br    = dp[branch_key]
    means = np.array([br['means'][ox] for ox in oxs], float)
    if 'da' in br:
        return boot_ci(br['da'], oxs)
    if 'sems' in br:
        s = np.array([br['sems'][ox] for ox in oxs], float)
        return means, means - 1.96 * s, means + 1.96 * s
    return means, means, means


DI_MEAN, DI_LO, DI_HI = boot_ci(DI_ARR, OX)
enc_m, enc_lo, enc_hi = band_for('fixed_consolidation', OX_C)   # encoding varies
con_m, con_lo, con_hi = band_for('fixed_encoding',      OX_C)   # consolidation varies
cpl_m, cpl_lo, cpl_hi = band_for('coupled',             OX_C)   # both vary

if np.allclose(enc_lo, enc_hi) or np.allclose(con_lo, con_hi) or np.allclose(cpl_lo, cpl_hi):
    print("  ⚠ A dual-pathway arm has no per-rep arrays in the pickle — its band "
          "is flat.\n    Check that 'da' is being saved for all three arms.")

# ── Peak estimates: bootstrap, not grid argmax ─────────────────────────────
def peak_info(mean_block, ci_block, p_int):
    """Package a bootstrap peak estimate; degrade to None if absent."""
    try:
        m = float(mean_block); c = [float(ci_block[0]), float(ci_block[1])]
        return {'mean': m, 'ci': c, 'p_int': float(p_int)}
    except Exception:
        return None


PK_A = peak_info(D['consolidation_regime']['peak']['mean'],
                 D['consolidation_regime']['peak']['ci_95'],
                 D['consolidation_regime']['p_interior'])
PK_C = peak_info(dp['fixed_encoding'].get('peak_mean'),
                 dp['fixed_encoding'].get('peak_ci', (np.nan, np.nan)),
                 dp['fixed_encoding'].get('p_interior'))
PK_D = peak_info(dp['coupled'].get('peak_mean'),
                 dp['coupled'].get('peak_ci', (np.nan, np.nan)),
                 dp['coupled'].get('p_interior'))

# ── Shape checks: warn, never crash ────────────────────────────────────────
_enc_rho  = dp['fixed_consolidation']['spearman_rho']
_con_pint = dp['fixed_encoding']['p_interior']
_cpl_pint = dp['coupled']['p_interior']
if _enc_rho < 0.90:
    print(f"  ⚠ Encoding arm rho = {_enc_rho:+.3f} (expected ~+1.0). "
          f"Check the fixed_consolidation mapping.")
if _con_pint < 0.90:
    print(f"  ⚠ Consolidation arm P(interior) = {_con_pint:.2f}. Peak near a boundary.")
if _cpl_pint < 0.90:
    print(f"  ⚠ Coupled arm P(interior) = {_cpl_pint:.2f}. Peak near a boundary.")

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.linewidth': 0.8, 'axes.labelsize': 10,
    'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 8, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
# Panel titles: left-aligned bold above each panel, as in Figures 1, 2, 4, 5.
TITLE_LOC, TITLE_BOLD, TITLE_SIZE, TITLE_PAD = 'left', True, 10, 8

# Harmonised with Figures 1-2. Red is reserved for empirical data there, so it
# is not used here: every curve in this figure is model output.
C_A  = '#1a5fa8'   # dark blue  — main sweep (Figure 2 synthetic dark)
C_B  = '#5b9bd5'   # light blue — encoding    (Figure 2 synthetic)
C_C  = '#d98c2b'   # amber      — consolidation
C_D  = '#7d6b9e'   # violet     — coupled (both pathways)
C_BL = '#9aa0a6'
C_T  = '#1c1917'
BAND_ALPHA = 0.16


def plabel(ax, lbl, x=-0.16, y=1.17):
    """Panel letter, top-aligned with the first line of the title beside it."""
    ax.text(x, y, lbl, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', color=C_T)


def draw_panel(ax, x, m, lo, hi, color, title, xlabel,
               pk=None, show_intact=False, ghost_peak=None):
    ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.plot(x, m, '-', color=color, lw=1.8, zorder=3)
    ax.plot(x, m, 'o', color=color, ms=4.5, mfc='white', mew=1.3, zorder=4)
    ax.axhline(NCM, color=C_BL, lw=1.0, ls='--', zorder=0)

    if show_intact:
        ax.axvline(INTACT, color='#555', lw=0.9, ls=':', zorder=2)

    if pk is not None and np.isfinite(pk['mean']):
        if np.isfinite(pk['ci'][0]) and np.isfinite(pk['ci'][1]):
            ax.axvspan(pk['ci'][0], pk['ci'][1], color=color, alpha=0.13,
                       lw=0, zorder=1)
        ax.axvline(pk['mean'], color=color, lw=1.4, ls='-', alpha=0.85, zorder=5)
        # grid argmax kept, but named for what it is — a single draw
        gi = int(np.argmax(m))
        ax.scatter([x[gi]], [m[gi]], s=55, facecolor='none', edgecolor=color,
                   lw=1.3, zorder=6)

    if ghost_peak is not None and np.isfinite(ghost_peak):
        ax.axvline(ghost_peak, color=C_C, lw=1.2, ls='--', alpha=0.65, zorder=2)

    ax.set_title(title, loc=TITLE_LOC, x=0.0, pad=TITLE_PAD,
                 fontsize=TITLE_SIZE, color=C_T, linespacing=1.35,
                 fontweight='bold' if TITLE_BOLD else 'normal')
    ax.set_xlabel(xlabel)
    ax.set_xlim(0.15, 1.95); ax.set_xticks([0.5, 1.0, 1.5])
    ax.spines[['top', 'right']].set_visible(False)


# shared y-limits so the four panels are directly comparable
ymin = min(DI_LO.min(), enc_lo.min(), con_lo.min(), cpl_lo.min(), NCM) - 0.004
ymax = max(DI_HI.max(), enc_hi.max(), con_hi.max(), cpl_hi.max()) + 0.014

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.6, 7.8), facecolor='white')
gs  = gridspec.GridSpec(2, 2, wspace=0.28, hspace=0.46,
                        left=0.095, right=0.97, top=0.925, bottom=0.135)

axA = fig.add_subplot(gs[0, 0]); axA.set_ylim(ymin, ymax); plabel(axA, 'A')
draw_panel(axA, ox_arr, DI_MEAN, DI_LO, DI_HI, C_A,
           'Consolidation-dependent memory\n(main sweep)',
           'OX2R level', pk=PK_A, show_intact=True)
axA.set_ylabel('Discrimination index')

axB = fig.add_subplot(gs[0, 1]); axB.set_ylim(ymin, ymax); plabel(axB, 'B')
draw_panel(axB, oxC_arr, enc_m, enc_lo, enc_hi, C_B,
           'Encoding-dominant\n(consolidation fixed)',
           'Orexin level (encoding)')

axC = fig.add_subplot(gs[1, 0]); axC.set_ylim(ymin, ymax); plabel(axC, 'C')
draw_panel(axC, oxC_arr, con_m, con_lo, con_hi, C_C,
           'Consolidation-dominant\n(encoding fixed)',
           'OX2R level (consolidation)', pk=PK_C, show_intact=True)
axC.set_ylabel('Discrimination index')

axD = fig.add_subplot(gs[1, 1]); axD.set_ylim(ymin, ymax); plabel(axD, 'D')
draw_panel(axD, oxC_arr, cpl_m, cpl_lo, cpl_hi, C_D,
           'Coupled pathways\n(both vary together)',
           'Orexin tone (both pathways)', pk=PK_D, show_intact=True,
           ghost_peak=PK_C['mean'] if PK_C else None)

fig.legend(handles=[
    Line2D([0], [0], color='#555', lw=1.6, label='Bootstrap peak estimate (line) '
                                                 'and its 95% CI (shaded)'),
    Line2D([0], [0], marker='o', color='w', mec='#555', mew=1.3, ms=8,
           label='Grid argmax of the mean curve (a single draw)'),
    Line2D([0], [0], color='#555', lw=0.9, ls=':', label='Intact signalling, OX2R = 1.0'),
    Line2D([0], [0], color=C_C, lw=1.2, ls='--', alpha=0.65,
           label='Panel C peak, redrawn in D'),
    Line2D([0], [0], color=C_BL, lw=1.0, ls='--',
           label='Discrimination index with consolidation disabled'),
], loc='lower center', bbox_to_anchor=(0.5, 0.004), ncol=2, frameon=False,
    fontsize=7.6, handletextpad=0.7, columnspacing=2.0)

out = DIRS['figures']
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'figure_3.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figure_3.pdf', facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figure_3.tiff', dpi=300, facecolor='white', bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'})
display(fig); plt.show(); plt.close(fig)

print(f"\nSaved to {out}")
for _ext in ('png', 'pdf', 'tiff'):
    _p = out / f'figure_3.{_ext}'
    print(f"  {_p.name:<16}" + (f"{_p.stat().st_size / 1e6:.1f} MB" if _p.exists() else "NOT FOUND"))
print()

# ── Caption numbers, straight from the pickle ──────────────────────────────
print("CAPTION NUMBERS (do not retype these by hand)")
print("-" * 66)
_c  = D['curvature']; _u = D['u_strength']
_q2 = _c.get('quadratic_coefficient', _c['curvature'])   # renamed; key kept
print(f"  bands              95% bootstrap CI OF THE MEAN over "
      f"{len(DI_ARR[OX[0]])} replicates (not replicate spread)")
print(f"  A  q2              {_q2:+.4f}   P(q2 >= 0) = "
      f"{_c['boot_sign_prob_nonneg']:.4f}")
print(f"     peak            {PK_A['mean']:.2f} [{PK_A['ci'][0]:.2f}, {PK_A['ci'][1]:.2f}]"
      f"   P(interior) = {PK_A['p_int']:.2f}")
print(f"     grid argmax     {ox_arr[int(np.argmax(DI_MEAN))]:.2f}  "
      f"(single draw on the {len(ox_arr)}-level grid; not the reported optimum)")
_dist = abs(PK_A['mean'] - INTACT) / INTACT * 100
print(f"     distance from intact signalling: {_dist:.0f}% "
      f"(CI spans {abs(PK_A['ci'][0] - INTACT) / INTACT * 100:.0f}% to "
      f"{abs(PK_A['ci'][1] - INTACT) / INTACT * 100:.0f}%)")
print(f"     U-strength      {_u['point']:+.4f} [{_u['ci_95'][0]:+.4f}, {_u['ci_95'][1]:+.4f}]")
print(f"     no-consol DI    {NCM:.4f}")
print(f"     benefit         {DI_MEAN.max() - NCM:+.4f} "
      f"({100 * _u['point'] / (DI_MEAN.max() - NCM):.0f}% of it lost off-optimum)")
print(f"     enc_trials      {D['consolidation_regime']['enc_trials_used']}")
for lbl, key in [('B  encoding-dominant', 'fixed_consolidation'),
                 ('C  consolidation-dom', 'fixed_encoding'),
                 ('D  coupled', 'coupled')]:
    br = dp[key]
    print(f"  {lbl:<22} rho={br['spearman_rho']:+.3f}  q2={br['curvature']:+.4f}  "
          f"peak={br['peak_mean']:.2f} [{br['peak_ci'][0]:.2f}, {br['peak_ci'][1]:.2f}]  "
          f"P(interior)={br['p_interior']:.2f}")
if PK_C and PK_D and np.isfinite(PK_C['mean']) and np.isfinite(PK_D['mean']):
    print(f"  C -> D peak shift under coupling: {PK_D['mean'] - PK_C['mean']:+.3f} "
          f"OX2R units")
print("\n  CAPTION MUST ALSO STATE:")
print(f"   - Panels A and C are the same manipulation on different grids "
      f"({len(ox_arr)} vs {len(oxC_arr)} levels);")
print(f"     they are not independent replications and should not be read as such.")
print(f"   - Shaded vertical band = 95% CI of the bootstrap peak; the open circle is")
print(f"     the grid argmax and is shown only to make the difference explicit.")
print(f"   - Dashed grey line = discrimination index with consolidation disabled.")
print("-" * 66)
