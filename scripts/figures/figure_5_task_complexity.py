"""
Figure 5 - the optimum shifts with required replay depth.

Reads:  results/agent_model_results.pkl
Writes: figures/figure_5.{png,pdf,tiff}

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
Figure 5: The optimal orexin level shifts with required replay depth
on two independent task axes; the non-monotonicity is robust.   [revised v3]

  A. Sequence-length axis     - headline, upper left
  B. Divergence axis          - headline, upper right
  C. Robustness, 5 axes       - control, lower left
  D. E/I termination slope    - mechanism sweep, lower right

WHAT CHANGED FROM v2

  PANEL C NO LONGER CONTAINS THE E/I SLOPE FAMILY. The extended k_slope sweep
  includes k = 0, at which truncation is disabled and the interior optimum is
  gone (P(interior) = 0.15) even though the fitted q2 stays negative, because
  a quadratic fitted to a saturating monotone curve turns over past the data.
  Counting those conditions under a "q2 < 0" criterion inflated the robustness
  claim with the very conditions that demonstrate the criterion is not
  sufficient. The k_slope family is a MECHANISM sweep and appears only in
  panel D. This follows the precedent already set for the divergence axis,
  which was removed from panel C when it became a headline axis.

  PANEL C REPORTS P(interior) WHEN THE PICKLE CARRIES IT. Negative curvature
  is compatible with a monotone saturating response, so where a robustness
  record stores p_interior the panel counts interior optima instead, and the
  title says which criterion was used. Records without p_interior fall back to
  the sign of q2.

  PANEL D DRAWS THE EXTENDED GRID. Bar width derives from the tightest gap in
  the grid rather than a constant sized for the old five-point grid; k = 0 is
  drawn in a separate colour and labelled as a control rather than as a point
  on a continuum of termination strength; and the panel title asserts the
  strong claim only if the control supports it.

REQUIRES in the pickle (all written by the agent model):
  complexity_peaks, peak_shift, complexity_interaction
  divergence_results (with 'peak_ci' and 'diverge_step'), divergence_interaction
  k_slope_extended  (from the extended mechanism-sweep cell; falls back to
                     inh_robustness with a warning if absent)
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

PKL = FILES['results']
with open(PKL, 'rb') as f:
    D = pickle.load(f)

for k in ('complexity_interaction', 'divergence_interaction'):
    if k not in D:
        raise KeyError(f"Pickle is missing '{k}'. Re-run the model cell with the "
                       f"interaction tests wired in - this script will not "
                       f"re-simulate or fall back to the censored vertex slope.")

OX_FLOOR, OX_CEIL = 0.30, 1.80

# -- A: sequence-length axis ------------------------------------------------
cpx      = D['complexity_peaks']
sls      = sorted(cpx.keys()); sls_arr = np.array(sls, float)
cpx_peak = np.array([cpx[sl]['mean']     for sl in sls])
cpx_lo   = np.array([cpx[sl]['ci_95'][0] for sl in sls])
cpx_hi   = np.array([cpx[sl]['ci_95'][1] for sl in sls])
ps       = D['peak_shift']
ci_seq   = D['complexity_interaction']

# -- B: divergence axis -----------------------------------------------------
divr     = D['divergence_results']
divs     = sorted(divr.keys())
_DIV_STEP_INFERRED = not all('diverge_step' in divr[d] for d in divs)
if _DIV_STEP_INFERRED:
    print("  \u26a0 divergence_results has no 'diverge_step' for every level; the "
          "x axis of panel B\n    falls back to round(16 * divergence), which is an "
          "assumption, not a logged value.")
div_step = np.array([divr[d].get('diverge_step', int(16 * float(d))) for d in divs], float)
div_peak = np.array([divr[d]['peak'] for d in divs])
_HAVE_DIV_CI = all('peak_ci' in divr[d] for d in divs)
if _HAVE_DIV_CI:
    div_lo = np.array([divr[d]['peak_ci'][0] for d in divs])
    div_hi = np.array([divr[d]['peak_ci'][1] for d in divs])
else:
    div_lo = div_hi = div_peak
    print("  \u26a0 divergence_results has no 'peak_ci'. Add "
          "\"'peak_ci': p['ci_95'],\" to div_res in the model cell for error bars.")
ci_div = D['divergence_interaction']

# -- C: robustness ----------------------------------------------------------
# The E/I slope family is deliberately absent: it is a mechanism sweep and
# lives in panel D. See the module docstring.
def _curv(v): return v['curvature'] if isinstance(v, dict) else v
def _peak(v): return v.get('peak', np.nan) if isinstance(v, dict) else np.nan
def _pint(v): return v.get('p_interior', np.nan) if isinstance(v, dict) else np.nan

rob_groups = []
rob_groups.append(('Dimension',
    [(_curv(D['dimension_robustness'][k]), _peak(D['dimension_robustness'][k]),
      _pint(D['dimension_robustness'][k]))
     for k in sorted(D['dimension_robustness'])]))
ind = D['induction_robustness']
rob_groups.append(('Induction theta/tau',
    [(_curv(ind[k]), _peak(ind[k]), _pint(ind[k]))
     for k in sorted(ind, key=lambda s: (s.split('_')[0], float(s.split('_')[1])))]))
rob_groups.append(('Consol LR',
    [(_curv(D['consol_lr_robustness'][k]), _peak(D['consol_lr_robustness'][k]),
      _pint(D['consol_lr_robustness'][k]))
     for k in sorted(D['consol_lr_robustness'])]))
rob_groups.append(('Trunc. gamma',
    [(_curv(D['gamma_robustness'][k]), _peak(D['gamma_robustness'][k]),
      _pint(D['gamma_robustness'][k]))
     for k in sorted(D['gamma_robustness'])]))
if 'encoding_sensitivity' in D:
    es = D['encoding_sensitivity']
    rob_groups.append(('Enc. trials',
        [(es[k]['curvature'], es[k]['peak'], es[k].get('p_interior', np.nan))
         for k in sorted(es)]))

inh = D['inh_robustness']          # panel D fallback only

flat_curv, flat_peak, flat_pint, flat_grp = [], [], [], []
for g, items in rob_groups:
    for c, p, pi in items:
        flat_curv.append(c); flat_peak.append(p); flat_pint.append(pi); flat_grp.append(g)
flat_curv = np.array(flat_curv); flat_peak = np.array(flat_peak)
flat_pint = np.array(flat_pint, float)
n_total = len(flat_curv)

# Prefer the interior-optimum criterion; fall back to the sign of q2.
_HAVE_PINT = bool(np.all(np.isfinite(flat_pint)))
if _HAVE_PINT:
    n_pass    = int(np.sum(flat_pint >= 0.5))
    ROB_CRIT  = 'retain an interior optimum'
    ROB_SHORT = 'P(interior) >= 0.5'
else:
    n_pass    = int(np.sum(flat_curv < 0))
    ROB_CRIT  = 'stay non-monotonic'
    ROB_SHORT = 'q2 < 0'
    print("  \u26a0 Robustness records carry no 'p_interior'; panel C falls back to "
          "the sign of q2.\n    Negative curvature alone is compatible with a "
          "saturating monotone response - add\n    \"'p_interior': "
          "bootstrap_peak_location(da, OX_COARSE)['p_interior'],\" to each "
          "robustness\n    record for the stronger criterion.")

# -- D: E/I termination slope -----------------------------------------------
# Prefer the extended record written by the mechanism-sweep cell; fall back to
# the legacy inh_robustness keys so the script still runs against an older
# pickle. The fallback cannot contain k_slope = 0 and the panel says so.
_EXT = D.get('k_slope_extended')
if _EXT:
    _ke    = sorted(_EXT.items(), key=lambda kv: float(kv[0]))
    k_arr  = np.array([float(k) for k, _ in _ke])
    k_curv = np.array([v['curvature'] for _, v in _ke])
    k_peak = np.array([v['peak'] for _, v in _ke])
    k_pint = np.array([v.get('p_interior', np.nan) for _, v in _ke])
    K_HAS_CONTROL = bool(np.any(np.isclose(k_arr, 0.0)))
else:
    k_entries = sorted((float(k.split('_')[1]), inh[k])
                       for k in inh if k.startswith('slope_'))
    k_arr  = np.array([k for k, _ in k_entries])
    k_curv = np.array([_curv(v) for _, v in k_entries])
    k_peak = np.array([_peak(v) for _, v in k_entries])
    k_pint = np.full(len(k_arr), np.nan)
    K_HAS_CONTROL = False
    print("  \u26a0 'k_slope_extended' not in the pickle. Panel D falls back to "
          "inh_robustness,\n    which contains no k_slope = 0 condition, so the "
          "panel cannot show whether\n    removing truncation removes the "
          "interior optimum. Run the extended sweep cell.")

K_OPERATING = 0.06          # the value the model runs at - a choice, not a bound

# Control row, read out for the caption and for the title logic below.
if K_HAS_CONTROL:
    _i0 = int(np.argmin(np.abs(k_arr)))
    K_CTL_CURV, K_CTL_PEAK = float(k_curv[_i0]), float(k_peak[_i0])
    K_CTL_PINT = float(k_pint[_i0])
    # "removes the optimum" is judged on P(interior), not on the sign of q2:
    # a quadratic fitted to a saturating monotone response keeps q2 < 0.
    K_CONTROL_KILLS_U = (np.isfinite(K_CTL_PINT) and K_CTL_PINT < 0.5) or \
                        (not np.isfinite(K_CTL_PINT) and K_CTL_CURV >= 0)
else:
    K_CTL_CURV = K_CTL_PEAK = K_CTL_PINT = np.nan
    K_CONTROL_KILLS_U = False

# -- Style ------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11, 'axes.linewidth': 0.9, 'axes.labelsize': 11.5,
    'axes.titlesize': 12, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'legend.fontsize': 9, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
FIG2_EMP, FIG2_SYN, FIG2_SYND = '#e05a4b', '#5b9bd5', '#1a5fa8'
C_SEQ  = FIG2_EMP
C_DIV  = FIG2_SYND
C_CURV = FIG2_SYND
C_PEAK = FIG2_EMP
C_CTL  = '#7d6b9e'
C_BL   = '#9aa0a6'
C_T    = '#1c1917'
group_palette = {
    'Dimension': '#1a5fa8', 'Induction theta/tau': '#5b9bd5', 'Consol LR': '#e05a4b',
    'Trunc. gamma': '#d98c2b', 'Enc. trials': '#4c8c7a',
}
# Unicode subscripts and Greek are missing from Helvetica/Arial and render as
# empty boxes; mathtext always resolves, so every label goes through this map.
group_label = {
    'Dimension': 'Dimension', 'Induction theta/tau': r'Induction $\theta/\tau$',
    'Consol LR': 'Consolidation LR',
    'Trunc. gamma': r'Truncation $\gamma$', 'Enc. trials': 'Encoding trials',
}
Q2 = r'Quadratic coefficient $q_2$'

_LADDER = (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10)


def _tickset(lo, hi, k):
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.linspace(0, 1, k + 1), np.inf
    span = hi - lo
    best = None
    for mag_shift in (-1, 0, 1):
        mag = 10.0 ** (np.floor(np.log10(span / k)) + mag_shift)
        for m in _LADDER:
            step = m * mag
            start = np.floor(lo / step) * step
            if start + k * step >= hi - 1e-12:
                waste = (start + k * step - hi) + (lo - start)
                if best is None or waste < best[1]:
                    best = (start + step * np.arange(k + 1), waste)
                break
    return best if best is not None else (np.linspace(lo, hi, k + 1), np.inf)


def align_twin(axL, axR, kmin=3, kmax=6):
    """Same tick count on both axes so the two grids coincide."""
    best_cost, best_t = np.inf, None
    for k in range(kmin, kmax + 1):
        tL, wL = _tickset(*axL.get_ylim(), k)
        tR, wR = _tickset(*axR.get_ylim(), k)
        cost = wL / max(tL[-1] - tL[0], 1e-12) + wR / max(tR[-1] - tR[0], 1e-12)
        if cost < best_cost:
            best_cost, best_t = cost, (tL, tR)
    tL, tR = best_t
    axL.set_ylim(tL[0], tL[-1]); axL.set_yticks(tL)
    axR.set_ylim(tR[0], tR[-1]); axR.set_yticks(tR)
    axL.grid(True, axis='y', color='0.90', lw=0.6, zorder=0)
    axR.grid(False)


def plabel(ax, lbl, x=-0.13, y=1.08):
    ax.text(x, y, lbl, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', color=C_T)


def peak_panel(ax, x, pk, lo, hi, color, xlabel, title, mark_censored=True):
    cens = (lo <= OX_FLOOR + 1e-6) | (hi >= OX_CEIL - 1e-6)
    ax.axhspan(0.15, OX_FLOOR, color='0.93', lw=0, zorder=0)
    ax.axhspan(OX_CEIL, 2.0, color='0.93', lw=0, zorder=0)
    ax.axhline(OX_FLOOR, color=C_BL, lw=0.7, ls='-', alpha=0.6, zorder=1)
    ax.axhline(OX_CEIL, color=C_BL, lw=0.7, ls='-', alpha=0.6, zorder=1)
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0, zorder=1)
    ax.errorbar(x, pk, yerr=[np.maximum(0, pk - lo), np.maximum(0, hi - pk)],
                fmt='none', ecolor=color, elinewidth=1.2, capsize=4,
                capthick=1.0, alpha=0.7, zorder=2)
    ax.plot(x, pk, '-o', color=color, lw=2.0, ms=6.5, mfc='white', mew=1.5, zorder=4)
    if mark_censored and cens.any():
        ax.scatter(x[cens], pk[cens], s=165, facecolor='none', edgecolor=color,
                   lw=1.4, ls=':', zorder=5)
    ax.axhline(1.0, color=C_T, lw=1.2, ls=(0, (5, 3)), zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Optimal OX2R level')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0.15, 2.0)
    ax.grid(True, axis='y', color='0.92', lw=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color=color, lw=2.0, marker='o', mfc='white', mew=1.5,
               label='Bootstrapped vertex, 95% CI'),
        Line2D([], [], color=color, lw=1.4, ls=':', marker='o', mfc='none',
               ms=9, label='CI reaches a sampled OX2R boundary'),
        Line2D([], [], color=C_T, lw=1.2, ls=(0, (5, 3)), label='baseline'),
        Line2D([], [], color=C_BL, lw=0.7, label='Sampled OX2R range'),
    ], loc='lower left', fontsize=8.5, frameon=True, framealpha=0.95,
        edgecolor='0.85', borderpad=0.35, labelspacing=0.32, handlelength=1.4)
    return cens


# -- Layout -----------------------------------------------------------------
fig = plt.figure(figsize=(14.0, 11.2), facecolor='white')
gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.32, hspace=0.42,
                        left=0.075, right=0.935, top=0.955, bottom=0.14)

# === A: sequence length ===
axA = fig.add_subplot(gs[0, 0]); plabel(axA, 'A', x=-0.13, y=1.10)
censA = peak_panel(axA, sls_arr, cpx_peak, cpx_lo, cpx_hi, C_SEQ,
                   'Sequence length (steps)',
                   'Depth axis 1: sequence length')
axA.set_xticks(sls_arr); axA.set_xlim(sls_arr[0] - 2, sls_arr[-1] + 2)

# === B: divergence ===
axB = fig.add_subplot(gs[0, 1]); plabel(axB, 'B', x=-0.13, y=1.10)
censB = peak_panel(axB, div_step, div_peak, div_lo, div_hi, C_DIV,
                   'Divergence step (steps of shared prefix)',
                   'Depth axis 2: divergence point',
                   mark_censored=_HAVE_DIV_CI)
axB.set_xticks(div_step); axB.set_xlim(div_step[0] - 1.5, div_step[-1] + 1.5)

# === C: robustness ===
axC = fig.add_subplot(gs[1, 0]); plabel(axC, 'C', x=-0.13, y=1.16)
axC.axvline(0, color=C_T, lw=1.0, zorder=2)
axC.axhline(1.0, color=C_T, lw=1.2, ls=(0, (5, 3)), zorder=2)
for g, _ in rob_groups:
    m = np.array([gg == g for gg in flat_grp])
    axC.plot(flat_curv[m], flat_peak[m], 'o', ms=5.5, alpha=0.85,
             color=group_palette[g], mec='white', mew=0.7,
             label=group_label[g], zorder=4)
axC.set_xlabel(Q2 + '   (negative = non-monotonic)')
axC.set_ylabel('Optimal OX2R level')
axC.set_xlim(float(np.nanmin(flat_curv)) * 1.25,
             max(0.006, float(np.nanmax(flat_curv)) * 1.3))
_pk_lo = float(np.nanmin(flat_peak)); _pk_hi = float(np.nanmax(flat_peak))
axC.set_ylim(_pk_lo - 0.12, _pk_hi + 0.12)
axC.grid(True, color='0.92', lw=0.5, zorder=0)
axC.spines[['top', 'right']].set_visible(False)
axC.set_title(f'Robustness: {n_pass}/{n_total} parameter variations\n{ROB_CRIT}',
              fontsize=12, fontweight='bold', pad=10)
axC.legend(loc='lower left', fontsize=8.5, frameon=True, framealpha=0.95,
           edgecolor='0.85', ncol=2, borderpad=0.35, labelspacing=0.3,
           handletextpad=0.35, columnspacing=0.8)

# === D: E/I termination slope ===
axD = fig.add_subplot(gs[1, 1]); plabel(axD, 'D', x=-0.13, y=1.16)

# Width from the tightest gap in the grid, so bars neither overlap nor shrink
# to hairlines when the grid is unevenly spaced.
_gaps = np.diff(k_arr)
_bw = float(np.min(_gaps)) * 0.62 if len(_gaps) else 0.003

_is_ctl = np.isclose(k_arr, 0.0)
_bar_c = [C_CTL if b else C_CURV for b in _is_ctl]

axD.bar(k_arr, k_curv, width=_bw, color=_bar_c, alpha=0.75,
        edgecolor=_bar_c, linewidth=0.5, zorder=2)
axD.axhline(0, color=C_T, lw=0.6, zorder=3)
axD.axvline(K_OPERATING, color=C_T, lw=1.0, ls=':', zorder=3)

_ylo = axD.get_ylim()[0]
axD.annotate(f'model runs here\n(k = {K_OPERATING:g}, a choice)',
             xy=(K_OPERATING, _ylo * 0.06),
             xytext=(K_OPERATING + 0.012, _ylo * 0.34),
             fontsize=9, color=C_T, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color=C_T, lw=0.8))

if K_HAS_CONTROL:
    axD.annotate('truncation off',
                 xy=(0.0, k_curv[_i0]),
                 xytext=(0.004, _ylo * 0.72),
                 fontsize=9, color=C_CTL, ha='left', va='center',
                 arrowprops=dict(arrowstyle='->', color=C_CTL, lw=0.8))

axD2 = axD.twinx()
axD2.plot(k_arr, k_peak, '--s', color=C_PEAK, lw=1.5, ms=5, mfc=C_PEAK,
          mec='white', mew=0.8, zorder=4)
axD2.set_ylabel('Optimal OX2R level', color=C_PEAK)
axD2.tick_params(axis='y', colors=C_PEAK)
axD2.set_ylim(float(np.nanmin(k_peak)) - 0.06, float(np.nanmax(k_peak)) + 0.06)

axD.set_xlabel(r'$k_{slope}$   (E/I termination slope)')
axD.set_ylabel(Q2, color=C_CURV)
axD.tick_params(axis='y', colors=C_CURV)
axD.tick_params(axis='x', labelsize=9.5)

# The title states the strong claim only when the control supports it.
if K_CONTROL_KILLS_U:
    _title_D = ('Removing E/I termination removes the optimum;\n'
                'stronger termination lowers it')
elif K_HAS_CONTROL:
    _title_D = ('The optimum survives with E/I termination disabled;\n'
                'stronger termination lowers it')
else:
    _title_D = ('Stronger E/I termination deepens the curve\n'
                'and lowers the optimum')
axD.set_title(_title_D, fontsize=12, fontweight='bold', pad=10)

axD.set_xlim(float(k_arr[0]) - _bw, float(k_arr[-1]) + _bw)
axD.set_xticks(k_arr)
axD.set_xticklabels([f'{v:g}' for v in k_arr], rotation=45, ha='right')
axD.spines['top'].set_visible(False); axD2.spines['top'].set_visible(False)
align_twin(axD, axD2)

_handles = [
    Line2D([], [], color=C_CURV, lw=6, alpha=0.75, label=Q2 + ', left axis'),
    Line2D([], [], color=C_PEAK, lw=1.5, ls='--', marker='s', ms=5,
           label='Optimal OX2R level, right axis'),
]
if K_HAS_CONTROL:
    _handles.insert(1, Line2D([], [], color=C_CTL, lw=6, alpha=0.75,
                              label='k = 0, truncation disabled (control)'))
# Legend below the x-axis label, outside the data area, so it covers
# neither the title nor the bars.
axD.legend(handles=_handles, loc='upper center', bbox_to_anchor=(0.5, -0.22),
           ncol=2, frameon=False, fontsize=9, handlelength=1.6,
           columnspacing=1.4, handletextpad=0.5)

out = DIRS['figures']
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'figure_5.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figure_5.pdf', facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figure_5.tiff', dpi=300, facecolor='white', bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'})
display(fig); plt.show(); plt.close(fig)

# -- Caption numbers --------------------------------------------------------
print("\nFigure 5 saved (PNG, PDF, TIFF)")
print("CAPTION NUMBERS (do not retype these by hand)")
print("-" * 72)
for nm, ci in [('sequence length', ci_seq), ('divergence step', ci_div)]:
    sh, ib, qd = ci['implied_shift'], ci['ox_by_length'], ci['quadratic']
    print(f"  {nm:<16} implied shift {sh['mean']:+.4f} OX2R/step "
          f"[{sh['ci_95'][0]:+.4f}, {sh['ci_95'][1]:+.4f}]  P(neg)={sh['p_neg']:.3f}")
    print(f"  {'':<16} OX\u00d7depth \u03b2 {ib['mean']:+.5f} "
          f"[{ib['ci_95'][0]:+.5f}, {ib['ci_95'][1]:+.5f}]  P(\u03b2<0)={ib['p_neg']:.3f}")
    print(f"  {'':<16} concavity \u03b2\u2082 {qd['mean']:+.5f}  P(\u03b2\u2082<0)={qd['p_neg']:.3f}")
print(f"  vertex slope (descriptive only) {ps['slope_mean']:+.4f} "
      f"[{ps['slope_ci_95'][0]:+.4f}, {ps['slope_ci_95'][1]:+.4f}]  "
      f"P(neg)={ps['p_slope_negative']:.3f}")

print(f"\n  A  {'length':>8}{'peak OX2R':>12}{'95% CI':>22}{'censored':>10}")
for i, sl in enumerate(sls):
    print(f"     {sl:>8}{cpx_peak[i]:>12.3f}"
          f"{f'[{cpx_lo[i]:.3f}, {cpx_hi[i]:.3f}]':>22}{'yes' if censA[i] else 'no':>10}")
print(f"  B  {'div step':>8}{'peak OX2R':>12}{'95% CI':>22}{'censored':>10}")
for i, d in enumerate(divs):
    print(f"     {div_step[i]:>8.0f}{div_peak[i]:>12.3f}"
          f"{f'[{div_lo[i]:.3f}, {div_hi[i]:.3f}]':>22}"
          f"{('yes' if censB[i] else 'no') if _HAVE_DIV_CI else 'n/a':>10}")

print(f"\n  C  robustness {n_pass}/{n_total} conditions {ROB_CRIT} ({ROB_SHORT}), "
      f"across {len(rob_groups)} axes; each point is one condition")
for g, items in rob_groups:
    if _HAVE_PINT:
        ok = sum(1 for _, _, pi in items if np.isfinite(pi) and pi >= 0.5)
    else:
        ok = sum(1 for c, _, _ in items if c < 0)
    print(f"       {g:<22} {ok}/{len(items)}")
print("       (the E/I slope family is a mechanism sweep and appears in panel D)")

print(f"\n  D  {'k_slope':>9}{'q2':>12}{'peak OX2R':>12}{'P(interior)':>13}")
for kk, cc, pp, pi in zip(k_arr, k_curv, k_peak, k_pint):
    _tag = '  <- truncation off' if np.isclose(kk, 0.0) else (
           '  <- operating point' if np.isclose(kk, K_OPERATING) else '')
    _pis = f'{pi:>13.3f}' if np.isfinite(pi) else f'{"n/a":>13}'
    print(f"     {kk:>9.3f}{cc:>12.4f}{pp:>12.3f}{_pis}{_tag}")

if K_HAS_CONTROL:
    print(f"\n     CONTROL  k = 0: q2 = {K_CTL_CURV:+.4f}, peak = {K_CTL_PEAK:.3f}, "
          f"P(interior) = {K_CTL_PINT:.3f}")
    print("     " + ("Removing truncation removes the interior optimum."
                     if K_CONTROL_KILLS_U else
                     "The interior optimum SURVIVES without truncation - the "
                     "descending-limb\n     claim does not hold as written."))

print("\n  CAPTION MUST STATE:")
print("   - The dotted ring marks estimates whose 95% CI reaches the sampled OX2R")
print(f"     floor or ceiling ({OX_FLOOR:g}, {OX_CEIL:g}). Those vertices are censored,")
print("     which is why the primary statistic is the OX x depth interaction fitted")
print("     to every replicate at every level; it never locates a vertex and is")
print("     therefore unaffected by the sampling boundaries.")
print("   - Panel D is a MECHANISM sweep, not a robustness check. k_slope is a free")
print("     structural parameter: nothing in the calibration data bounds it, since")
print("     Kostansek et al. constrain LFP signatures rather than replay content.")
print(f"     The model runs at k_slope = {K_OPERATING:g}; the sweep shows how the")
print("     curvature and the optimum depend on termination strength around it.")
print("   - At k = 0 no event is truncated at any OX2R level. The fitted q2 stays")
print("     negative there because a quadratic fitted to a saturating monotone")
print("     response turns over past the sampled range; P(interior) is the")
print("     criterion that separates the two cases, which is why the k_slope")
print("     family is excluded from the panel C robustness count.")
print("-" * 72)

import numpy as np

def show_peaks(name, key, xlabel):
    if key not in D:
        print(f"{name}: NOT in pickle"); return None
    pk = D[key]
    xs = sorted(pk.keys())
    print(f"\n{name}  [{key}]")
    print(f"  {xlabel:<14}{'peak OX2R':<12}{'95% CI':<22}{'censored?'}")
    print("  " + "-"*56)
    rows = []
    for x in xs:
        m = pk[x]['mean'] if isinstance(pk[x], dict) and 'mean' in pk[x] else pk[x].get('peak', np.nan)
        ci = pk[x].get('ci_95', [np.nan, np.nan]) if isinstance(pk[x], dict) else [np.nan, np.nan]
        cens = "← floor" if ci[0] <= 0.301 else ("← ceil" if ci[1] >= 1.799 else "")
        print(f"  {str(x):<14}{m:<12.3f}[{ci[0]:.3f}, {ci[1]:.3f}]      {cens}")
        rows.append((x, m, ci[0], ci[1]))
    # direction + rough slope
    xs_a = np.array([r[0] for r in rows], float); ms_a = np.array([r[1] for r in rows])
    if len(xs_a) >= 2:
        slope = np.polyfit(xs_a, ms_a, 1)[0]
        print(f"  → peak moves {'UP' if slope>0 else 'DOWN'} with {xlabel} (slope ≈ {slope:+.4f})")
    return rows

show_peaks("LOAD (memories held)",   "load_peaks",   "n_memories")
show_peaks("BUDGET (consol. budget)", "budget_peaks", "budget_frac")

# also check if a formal shift stat exists for either
for k in D:
    if 'load' in k.lower() or 'budget' in k.lower():
        if 'shift' in k.lower() or 'slope' in k.lower():
            print(f"\nFormal shift stat: {k} →", D[k])
