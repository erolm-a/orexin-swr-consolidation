"""
Stage 4 — publication figures.

Reads:  data_processed/swr_detection.pkl
        models/synthetic_libraries.pkl
        results/agent_model_results.pkl
Writes: figures/figure_[1-5].{png,pdf} and supplementary panels

Self-contained: reloads every pickle it needs, assumes no kernel state.

Paths and artefact versions are resolved via orexin_paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import setup  # noqa: E402

DIRS, FILES, PROJECT_ROOT, VERSIONS, load_artifact = setup()
from common.plotting import panel_label, plabel  # noqa: E402
from common.stats import (  # noqa: E402
    bootstrap_continuous_peak, dist_stats, test_curvature,
)
from common.utils import (  # noqa: E402
    _cat, _mean_or_nan, _safe_np, bandpass, coupled_event_arrays_from_trial,
    ox_key, pooled_coupled_event_arrays,
)

# =============================================================================
# CELL 7: FIGURE 1 - SWR DETECTION & VALIDATION (PUBLICATION)  [revised v2]
#   - Two Cicero sessions disambiguated by date (0910 / 0917) across C/D/E
#   - Panel E: histogram at PSD frequency resolution (~9.8 Hz) + light KDE guide
#     (replaces over-smoothed solo KDE; shows true concentration honestly)
#   - Panel D title/legend decrowded; negative-lag convention noted
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import signal, stats
from pathlib import Path
import pickle

# Paths come from orexin_paths (see header). DIRS / FILES are already bound.
PKL_PATH = FILES['detector']

EXAMPLE_SESSION = 'Achilles_10252013'
EXAMPLE_SEED = 7
WINDOW_MS = 200

SESS_COLORS = {
    'Achilles_10252013': '#1f77b4',
    'Cicero_09102014':   '#ff7f0e',
    'Cicero_09172014':   '#2ca02c',
}
DEFAULT_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'figure.dpi': 110,
})

# -----------------------------------------------------------------------------
# (bandpass is imported from common.utils)

def short(name):
    """Compact, DISAMBIGUATED session label.
    Two Cicero sessions share a stem → append the date (MMDD) so C/D/E
    legends don't collapse them into one indistinct 'Cicero'."""
    parts = name.split('_')
    stem = parts[0]
    if stem == 'Cicero' and len(parts) > 1 and len(parts[1]) >= 4:
        return f"Cicero {parts[1][:4]}"   # e.g. 'Cicero 0910', 'Cicero 0917'
    return stem

def color_for(name, i):
    return SESS_COLORS.get(name, DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])

def load_example_traces(sess):
    try:
        session_dir = DIRS['data_raw'] / sess['name']
        lfp_files = list(session_dir.glob('*.eeg')) + list(session_dir.glob('*.lfp'))
        if not lfp_files:
            return None
        lfp_path = lfp_files[0]
        n_ch = sess['lfp_params']['n_channels']
        fs = sess['lfp_params']['fs']
        swr = sess['gold'].get('ripple_gold') or sess['ripple_tiers']['ripples_swr']
        if not swr:
            return None
        evt = swr[EXAMPLE_SEED % len(swr)]
        spw_ch = sess['channel_info']['spw_channels'][0]
        rip_ch = sess['channel_info']['ripple_channels'][0]
        mmap = np.memmap(lfp_path, dtype=np.int16, mode='r').reshape(-1, n_ch)
        peak_samp = int(evt['peak_sec'] * fs)
        half = int((WINDOW_MS / 2) * fs / 1000)
        s = max(0, peak_samp - half); e = min(mmap.shape[0], peak_samp + half)
        raw_spw = mmap[s:e, spw_ch].astype(np.float32)
        raw_rip = mmap[s:e, rip_ch].astype(np.float32)
        del mmap
        spw_filt = bandpass(raw_spw, fs, 5, 40)
        spw_env = np.abs(signal.hilbert(spw_filt))
        rip_filt = bandpass(raw_rip, fs, 150, 250)
        rip_env = np.abs(signal.hilbert(rip_filt))
        t_ms = np.arange(len(raw_spw)) / fs * 1000 - WINDOW_MS / 2
        meta = {'dur': evt['duration_ms'], 'freq': evt.get('peak_freq', np.nan),
                'purity': evt.get('spectral_purity', np.nan),
                'spw_ch': spw_ch, 'rip_ch': rip_ch, 'rip_env': rip_env}
        return t_ms, raw_spw, spw_filt, spw_env, rip_filt, meta
    except Exception as ex:
        print(f"  [panel B] could not load raw traces: {ex}")
        return None

# -----------------------------------------------------------------------------
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
results = data['datasets']
print(f"Loaded {len(results)} sessions from {PKL_PATH.name}")

session_data = {}
for i, sess in enumerate(results):
    name = sess['name']
    swr = sess['ripple_tiers']['ripples_swr']
    nrem_min = sess['quality']['nrem_duration_min']
    peak_freqs = [e['peak_freq'] for e in swr if np.isfinite(e.get('peak_freq', np.nan))]
    durations = [e['duration_ms'] for e in swr]
    lags = sess['coupling_swr']['coupling_lags_ms']
    lags = lags.tolist() if len(lags) else []
    session_data[name] = {
        'color': color_for(name, i),
        'peak_freq': peak_freqs, 'duration_ms': durations, 'coupling_lags': lags,
        'spw_rate': sess['quality']['spw_rate'], 'swr_rate': sess['quality']['swr_rate'],
        'gold_swr_rate': sess['gold']['n_ripple_gold'] / nrem_min if nrem_min else 0,
        'coupled_swr_rate': (sess['coupling_swr']['n_coupled_spw'] / nrem_min) if nrem_min else 0,
        'n_swr': len(swr),
        'median_freq': np.median(peak_freqs) if peak_freqs else np.nan,
    }

names = list(session_data.keys())
all_freq = np.asarray([f for d in session_data.values() for f in d['peak_freq']], float)
all_lags = np.asarray([l for d in session_data.values() for l in d['coupling_lags']], float)

# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 12))
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.0, 0.9],
                       width_ratios=[1, 1.15], hspace=0.46, wspace=0.26,
                       left=0.07, right=0.97, top=0.95, bottom=0.06)

# (panel_label is imported from common.plotting)

# ── A: pipeline ──
axA = fig.add_subplot(gs[0, 0]); axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis('off')
panel_label(axA, 'A', dx=-0.02, dy=1.10)
axA.set_title('Signal-processing pipeline', fontweight='bold', loc='left', x=0.06)
steps = [
    ("Raw LFP  (NREM epochs only)", '#e8e8e8'),
    ("Channel selection\n(SPW / ripple / noise; raw power)", '#dbe9f6'),
    ("SPW detect 5-40 Hz\nHilbert env., 4x MAD, >=2 ch", '#cfe3d4'),
    ("Ripple detect 150-250 Hz\n5x MAD, >=4 cycles, >=3 ch\nnoise veto 300-500 Hz", '#fde6cf'),
    ("Co-occurrence  (-20 .. +50 ms)\n-> SWR tier (purity>=2, 150-200 Hz)", '#e7dcf0'),
]
n = len(steps); box_h = 1.45; gap = (10 - n * box_h) / (n + 1)
y = 10 - gap - box_h; centres = []
for label, fc in steps:
    box = FancyBboxPatch((1.0, y), 8.0, box_h, boxstyle="round,pad=0.06,rounding_size=0.12",
                         linewidth=1.0, edgecolor='#444', facecolor=fc)
    axA.add_patch(box)
    axA.text(5.0, y + box_h / 2, label, ha='center', va='center', fontsize=7.4)
    centres.append(y + box_h / 2); y -= (box_h + gap)
for y0, y1 in zip(centres[:-1], centres[1:]):
    axA.add_patch(FancyArrowPatch((5.0, y0 - box_h/2 + 0.04), (5.0, y1 + box_h/2 - 0.04),
                  arrowstyle='-|>', mutation_scale=11, linewidth=1.1, color='#333'))

# ── B: example SWR ──
gsB = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1], hspace=0.12)
axB1 = fig.add_subplot(gsB[0]); axB2 = fig.add_subplot(gsB[1], sharex=axB1)
axB3 = fig.add_subplot(gsB[2], sharex=axB1)
panel_label(axB1, 'B', dx=-0.10)
ex_sess = next((r for r in results if r['name'] == EXAMPLE_SESSION), results[0])
traces = load_example_traces(ex_sess)
if traces is not None:
    t_ms, raw_spw, spw_filt, spw_env, rip_filt, meta = traces
    axB1.set_title(f"Example SWR  |  {meta['dur']:.0f} ms, {meta['freq']:.0f} Hz, "
                   f"purity {meta['purity']:.1f}", fontsize=8.5, loc='right')
    axB1.plot(t_ms, raw_spw, color='k', lw=0.7)
    axB1.set_ylabel(f"Raw (ch {meta['spw_ch']})", fontsize=7.5)
    axB1.axvline(0, color='r', lw=0.8, alpha=0.5)
    axB2.plot(t_ms, spw_filt, color='#1f5fa8', lw=0.7, alpha=0.6)
    axB2.plot(t_ms, spw_env, color='#1f5fa8', lw=1.3)
    axB2.set_ylabel("SPW", fontsize=7.5); axB2.axvline(0, color='r', lw=0.8, alpha=0.5)
    _ix = int(0.06 * len(t_ms))
    axB2.annotate('Hilbert env.', xy=(t_ms[_ix], spw_env[_ix]), xytext=(0.02, 0.90),
                  textcoords='axes fraction', fontsize=6.5, color='#1f5fa8',
                  arrowprops=dict(arrowstyle='-', color='#1f5fa8', lw=0.6))
    axB3.plot(t_ms, rip_filt, color='#7b2fbf', lw=0.7)
    axB3.set_ylabel("Ripple", fontsize=7.5); axB3.axvline(0, color='r', lw=0.8, alpha=0.5)
    axB3.set_xlabel("Time relative to SPW peak (ms)")
    x1 = t_ms[-1]; x0 = x1 - 50; yl0, yl1 = axB1.get_ylim()
    yb = yl0 + 0.12 * (yl1 - yl0)
    axB1.plot([x0, x1], [yb, yb], color='k', lw=2.0, solid_capstyle='butt')
    axB1.text((x0 + x1)/2, yb + 0.04*(yl1 - yl0), '50 ms', ha='center', va='bottom', fontsize=7)
else:
    for ax in (axB1, axB2, axB3): ax.set_xticks([]); ax.set_yticks([])
    axB1.set_title("Example SWR (raw LFP unavailable here)", fontsize=8.5)
    axB2.text(0.5, 0.5, "Raw .eeg/.lfp not reachable.\nPlace the CRCNS session under\n"
              "data_raw/ to render example traces.", ha='center', va='center',
              transform=axB2.transAxes, fontsize=8, color='#888')
for ax in (axB1, axB2): plt.setp(ax.get_xticklabels(), visible=False)

# ── C: event rates ──
axC = fig.add_subplot(gs[1, 0]); panel_label(axC, 'C')
axC.set_title('Event rates: SPW vs SWR', fontweight='bold', loc='left', x=0.0)
x = np.arange(len(names))
spw_rates = [session_data[n]['spw_rate'] for n in names]
swr_rates = [session_data[n]['swr_rate'] for n in names]
gold_rates = [session_data[n]['gold_swr_rate'] for n in names]
axC.bar(x, spw_rates, width=0.62, color='#cfe0f2', edgecolor='#5b7fa6', label='SPW (5-40 Hz)')
axC.bar(x, swr_rates, width=0.40, color='#2ca02c', edgecolor='#1b5e1b',
        label='SWR tier (coupled + pure)')
axC.bar(x, gold_rates, width=0.40, color='#9bd99b', edgecolor='#1b5e1b',
        hatch='///', label='Gold SWR subset')
for xi, nm in zip(x, names):
    axC.text(xi, swr_rates[names.index(nm)] + 0.5, f"{swr_rates[names.index(nm)]:.1f}",
             ha='center', va='bottom', fontsize=7, color='#1b5e1b')
axC.set_xticks(x); axC.set_xticklabels([short(n) for n in names], fontsize=7.5)
axC.set_ylabel('Event rate (events / min)')
axC.legend(loc='upper right', frameon=False)
axC.spines[['top', 'right']].set_visible(False)

# ── D: coupling lag ──
axD = fig.add_subplot(gs[1, 1]); panel_label(axD, 'D', dx=-0.10)
axD.set_title('SPW→SWR temporal coupling', fontweight='bold', loc='left', x=0.0, pad=26)
bins_lag = np.linspace(-30, 60, 31)
bottoms = np.zeros(len(bins_lag) - 1)
for nm in names:
    d = session_data[nm]
    if not d['coupling_lags']: continue
    h, _ = np.histogram(d['coupling_lags'], bins=bins_lag)
    axD.bar(bins_lag[:-1], h, width=np.diff(bins_lag), bottom=bottoms, align='edge',
            color=d['color'], alpha=0.75, edgecolor='white', linewidth=0.3,
            label=f"{short(nm)} (n={d['n_swr']})")
    bottoms += h
pooled_median = np.median(all_lags)
axD.axvline(0, color='k', lw=1.2)
axD.axvline(pooled_median, color='crimson', lw=1.4, ls='--',
            label=f'median {pooled_median:.1f} ms')
axD.set_xlabel('SPW→SWR lag (ms)    (negative = ripple precedes SPW peak)', fontsize=8)
axD.set_ylabel('Count'); axD.set_xlim(-25, 35)
axD.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
           fontsize=6.5, columnspacing=1.0, handlelength=1.2, handletextpad=0.4)
axD.spines[['top', 'right']].set_visible(False)
axDin = axD.inset_axes([0.66, 0.55, 0.31, 0.38])
med_lags = [np.median(session_data[n]['coupling_lags']) if session_data[n]['coupling_lags']
            else np.nan for n in names]
axDin.bar(range(len(names)), med_lags,
          color=[session_data[n]['color'] for n in names], alpha=0.85)
axDin.axhline(0, color='k', lw=0.6); axDin.set_xticks(range(len(names)))
axDin.set_xticklabels([short(n).replace('Cicero ', 'C') for n in names], fontsize=5.5)
axDin.set_ylabel('median lag (ms)', fontsize=6); axDin.tick_params(labelsize=6)

# ── E: ripple peak frequency — histogram at PSD resolution + light KDE guide ──
axE = fig.add_subplot(gs[2, :]); panel_label(axE, 'E', dx=-0.045)
axE.set_title('Ripple peak frequency', fontweight='bold', loc='left', x=0.0)
mean_freq = np.mean(all_freq); sd_freq = np.std(all_freq)

# Histogram binned at the PSD frequency resolution (~9.8 Hz). This is the data's
# true granularity; finer bins or narrower KDEs would resolve FFT-grid artifacts.
PSD_BIN_HZ = 1250.0 / 128.0          # ~9.77 Hz
edges = np.arange(145, 210 + PSD_BIN_HZ, PSD_BIN_HZ)
axE.hist(all_freq, bins=edges, density=True, color='#7a9ec2', alpha=0.55,
         edgecolor='white', linewidth=0.6, label=f'Histogram (~{PSD_BIN_HZ:.0f} Hz bins)')

# Light KDE as a smooth guide only — moderate bandwidth, clearly secondary
try:
    kde = stats.gaussian_kde(all_freq, bw_method=0.30)
    xs = np.linspace(145, 210, 400); dens = kde(xs)
    axE.plot(xs, dens, color='#1f1f1f', lw=1.6, ls='-', alpha=0.75,
             label='KDE guide (bw≈0.30)')
except Exception:
    pass

axE.axvline(150, color='#888', ls=':', lw=1.0)
axE.axvline(200, color='#888', ls=':', lw=1.0, label='Ripple band limits')
axE.axvline(mean_freq, color='crimson', ls='--', lw=1.5, label=f'mean {mean_freq:.0f} Hz')
axE.set_xlabel('Ripple peak frequency (Hz)'); axE.set_ylabel('Probability density')
axE.set_xlim(145, 210); axE.legend(loc='upper right', frameon=False, ncol=1, fontsize=7.5)
axE.spines[['top', 'right']].set_visible(False)
axE.text(0.015, 0.95, f"n = {len(all_freq)}\nmean = {mean_freq:.0f} Hz\nSD = {sd_freq:.0f} Hz",
         transform=axE.transAxes, ha='left', va='top', fontsize=8,
         bbox=dict(boxstyle='round', fc='white', ec='#bbb', alpha=0.9))
# Per-session agreement noted in caption (Achilles ~173, Cicero 0910 ~172,
# Cicero 0917 ~174 Hz; all SD ≈ 13 Hz) — inset removed to let the panel breathe.

# ── save ──
fig.suptitle('Figure 1.  SWR detection and physiological validation (v5.4.5)',
             fontsize=12, fontweight='bold', y=0.985)
png_path = DIRS['figures'] / 'fig1_swr_detection.png'
pdf_path = DIRS['figures'] / 'fig1_swr_detection.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
try:
    from IPython.display import display; display(fig)
except Exception: pass
plt.show()
print(f"\nSaved:\n  {png_path}\n  {pdf_path}")
print(f"Pooled: n={len(all_freq)} SWR | freq {mean_freq:.1f}+/-{sd_freq:.1f} Hz | "
      f"median lag {pooled_median:.1f} ms")

# =============================================================================
# MANUSCRIPT FIGURE 2 — Synthetic library validation (v5.8.3)
#   Panel A:  Session-level agreement via LEAVE-ONE-SESSION-OUT (LOSO)
#             prediction intervals (out-of-sample; no circularity).
#   Panels B–E: Event-level distribution agreement (KS / EMD) vs Girardeau
#             in-vivo empirical distributions.
#
# Sources (live load):
#   - swr_detection.pkl          (empirical ground truth)
#   - synthetic_libraries.pkl    (Cell 7 output)
#   - generator_config.pkl
#
# Panel A band = Gamma-Poisson (rates) / Beta-Binomial (P couple) prediction
# interval for each held-out session, with between-session dispersion estimated
# ONLY from the other sessions. Pooled MEAN uses all trials (the [CONSTRAINED]
# claim); only DISPERSION is held out. With 3 sessions this is a consistency
# check, not a tight test (phi_between has 1 dof).
# =============================================================================

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
    from IPython.display import display
except Exception:
    def display(obj): pass

print("=" * 70)
print("MANUSCRIPT FIGURE 2 — Synthetic library validation (v5.8.3) [LOSO Panel A]")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Paths come from orexin_paths (see header). DIRS / FILES are already bound.
FIG_DIR = DIRS["models"] / "figures_manuscript"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_PKL = FILES["detector"]
LIB_PKL      = FILES["libraries"]
CFG_PKL      = FILES["generator_config"]

CONTEXT, OX_NULL, N_PRED, RNG_SEED = "Context_A", 1.0, 4000, 42

COL_EMP, COL_SYN, COL_SYN_D = "#e05a4b", "#5b9bd5", "#1a5fa8"
TAG_COLORS = {"[CONSTRAINED]": "#fff3cd", "[FIT-PRESERVED]": "#cce5ff",
              "[REPRODUCED]": "#d4edda"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7,
})

# ─── Cell-8 helpers (canonical versions in common.utils: ox_key, _safe_np,
#     _mean_or_nan, coupled_event_arrays_from_trial, pooled_coupled_event_arrays,
#     _cat, dist_stats in common.stats) ───────────────────────────────────────

# ─── LOSO prediction-interval functions ─────────────────────────────────────
def loso_rate_prediction_interval(null_trials, sess_df, metric_col, count_key,
                                  duration_sec=3.0, n_pred=4000, rng_seed=42):
    """LOSO prediction interval for event RATES (Gamma-Poisson)."""
    rng = np.random.default_rng(rng_seed)
    counts    = np.array([int(t.get(count_key, 0)) for t in null_trials], float)
    mu_trial  = counts.mean()                       # pooled mean — ALL trials
    var_trial = counts.var(ddof=1)
    phi_within = max((var_trial - mu_trial) / (mu_trial**2), 0.0) if mu_trial > 0 else 0.0

    bands, phi_log = {}, {}
    for held in sess_df["session"]:
        others = sess_df.loc[sess_df["session"] != held, metric_col].dropna().values
        phi_between = (others.var(ddof=1) / (others.mean()**2)
                       if len(others) >= 2 and others.mean() > 0 else 0.0)
        phi = max(phi_within, phi_between)
        phi_log[held] = phi
        row = sess_df.loc[sess_df["session"] == held].iloc[0]
        n_trials = max(1, int(np.ceil(row["duration_min"] * 60.0 / duration_sec)))
        gains = rng.gamma(1.0/phi, phi, size=n_pred) if phi > 0 else np.ones(n_pred)
        sim = rng.poisson(mu_trial * n_trials * gains) / (n_trials * duration_sec) * 60.0
        bands[held] = sim
    return bands, phi_log

def loso_pcouple_prediction_interval(null_trials, sess_df,
                                     duration_sec=3.0, n_pred=4000, rng_seed=42):
    """LOSO prediction interval for P(couple) (Beta-Binomial)."""
    rng = np.random.default_rng(rng_seed)
    spw = np.array([int(t.get("n_spw", 0)) for t in null_trials], float)
    cpl = np.array([int(t.get("n_spw_with_ripple", 0)) for t in null_trials], float)
    tot_spw, tot_cpl = spw.sum(), cpl.sum()
    p_bar = tot_cpl / tot_spw if tot_spw > 0 else np.nan

    bands, phi_log = {}, {}
    for held in sess_df["session"]:
        others = sess_df.loc[sess_df["session"] != held, "p_couple"].dropna().values
        phi_between = (others.var(ddof=1) / (p_bar*(1-p_bar))
                       if len(others) >= 2 and 0 < p_bar < 1 else 1e-6)
        phi = max(phi_between, 1e-6)
        phi_log[held] = phi
        row = sess_df.loc[sess_df["session"] == held].iloc[0]
        n_trials   = max(1, int(np.ceil(row["duration_min"] * 60.0 / duration_sec)))
        n_spw_sess = max(1, int(round(spw.mean() * n_trials)))
        a = p_bar * (1/phi - 1); b = (1 - p_bar) * (1/phi - 1)
        ps = rng.beta(max(a, 1e-3), max(b, 1e-3), size=n_pred)
        bands[held] = rng.binomial(n_spw_sess, ps) / n_spw_sess
    return bands, phi_log

# (dist_stats is imported from common.stats; _cat from common.utils)

# ─── [1] Load empirical ──────────────────────────────────────────────────────
print("\n[1] Loading empirical...")
with open(DETECTOR_PKL, "rb") as f: det = pickle.load(f)
_dur, _frq, _cyc, _lag, _rows = [], [], [], [], []
for sess in det["datasets"]:
    ts = float(sess["nrem_intervals"]["total_sec"])
    spw, swr, coup = sess["spw_events"], sess["ripple_tiers"]["ripples_swr"], sess["coupling_swr"]
    n_spw, n_swr, n_c = len(spw), len(swr), int(coup.get("n_coupled_spw", 0))
    sd = np.array([e["duration_ms"] for e in swr], float)
    sf = np.array([e.get("peak_freq", np.nan) for e in swr], float)
    cc = []
    for e in swr:
        if e.get("n_cycles") is not None: cc.append(float(e["n_cycles"]))
        else:
            pf = e.get("peak_freq", np.nan)
            cc.append((e["duration_ms"]/1000.0)*pf if np.isfinite(pf) else np.nan)
    cc = np.array(cc, float)
    lg = np.array(coup.get("coupling_lags_ms", []), float)
    if len(sd): _dur.append(sd)
    if len(sf[np.isfinite(sf)]): _frq.append(sf[np.isfinite(sf)])
    if len(cc[np.isfinite(cc)]): _cyc.append(cc[np.isfinite(cc)])
    if len(lg[np.isfinite(lg)]): _lag.append(lg[np.isfinite(lg)])
    _rows.append({"session": sess.get("name", f"sess_{len(_rows)}"),
                  "duration_min": ts/60.0,
                  "spw_rate_per_min": n_spw/ts*60.0 if ts>0 else np.nan,
                  "swr_rate_per_min": n_swr/ts*60.0 if ts>0 else np.nan,
                  "p_couple": n_c/n_spw if n_spw>0 else np.nan})
# (_cat is imported from common.utils)
sess_df = pd.DataFrame(_rows)
emp = {"swr_durs": _cat(_dur), "swr_freqs": _cat(_frq),
       "swr_cycles": _cat(_cyc), "coupling_lags": _cat(_lag)}
print(f"  {len(sess_df)} sessions.")

# ─── [2] Load synthetic ──────────────────────────────────────────────────────
print("\n[2] Loading v5.8.3 libraries...")
with open(LIB_PKL, "rb") as f: lib = pickle.load(f)
libraries = lib["libraries"]; duration_sec = float(lib.get("duration_sec", 3.0))
null_trials = libraries["null"][f"{CONTEXT}_ox{ox_key(OX_NULL)}"]
null_swr_d, null_swr_f, null_swr_c = pooled_coupled_event_arrays(null_trials)
null_lags = np.concatenate([_safe_np(t.get("coupling_lags", [])) for t in null_trials]).astype(float)
null_lags = null_lags[np.isfinite(null_lags)]
syn = {"swr_durs": null_swr_d, "swr_freqs": null_swr_f,
       "swr_cycles": null_swr_c, "coupling_lags": null_lags}
print(f"  Null trials: {len(null_trials)}")

# ─── [3] LOSO bands for Panel A ──────────────────────────────────────────────
print("\n[3] Building LEAVE-ONE-SESSION-OUT prediction intervals for Panel A...")
spw_band, spw_phi = loso_rate_prediction_interval(
    null_trials, sess_df, "spw_rate_per_min", "n_spw", duration_sec, N_PRED, RNG_SEED)
swr_band, swr_phi = loso_rate_prediction_interval(
    null_trials, sess_df, "swr_rate_per_min", "n_coupled", duration_sec, N_PRED, RNG_SEED)
pc_band,  pc_phi  = loso_pcouple_prediction_interval(
    null_trials, sess_df, duration_sec, N_PRED, RNG_SEED)
pred_bands = {"spw_rate_per_min": spw_band,
              "swr_rate_per_min": swr_band,
              "p_couple": pc_band}

print("\n  Per-held-out phi (from the OTHER 2 sessions):")
print(f"  {'Held-out':<22} {'SPW phi':>9} {'SWR phi':>9} {'P(c) phi':>10}")
print("  " + "-"*54)
for sn in sess_df["session"]:
    print(f"  {sn:<22} {spw_phi[sn]:>9.4f} {swr_phi[sn]:>9.4f} {pc_phi[sn]:>10.4f}")

pctile = {}
for _, row in sess_df.iterrows():
    sn = row["session"]
    pctile[sn] = {c: float(np.mean(pred_bands[c][sn] <= row[c]))*100
                  for c in ["spw_rate_per_min","swr_rate_per_min","p_couple"]}

n_in = sum(1 < pctile[sn][c] < 99
           for sn in sess_df["session"]
           for c in ["spw_rate_per_min","swr_rate_per_min","p_couple"])
print(f"\n  LOSO coverage (within 95% PI): {n_in}/{len(sess_df)*3} session×metric cells")

# ─── [4] BUILD FIGURE ────────────────────────────────────────────────────────
def _pct(p):
    if p >= 99: return ">99th"
    if p <= 1:  return "<1st"
    return f"{p:.0f}th"

fig = plt.figure(figsize=(12.5, 12.5))
gs = gridspec.GridSpec(4, 2, figure=fig,
                       height_ratios=[1.0, 0.12, 1.0, 1.0],
                       hspace=0.55, wspace=0.28)

# ----- Panel A (LOSO) -----
gsA = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.38)
panelA_cfg = [
    ("spw_rate_per_min", "SPW rate (events/min)", "[CONSTRAINED]"),
    ("swr_rate_per_min", "SWR rate (events/min)", "[FIT-PRESERVED]"),
    ("p_couple",         "P(SPW→SWR)",            "[CONSTRAINED]"),
]
snames = list(sess_df["session"]); n_s = len(snames)
for si, (col, ylabel, tag) in enumerate(panelA_cfg):
    ax = fig.add_subplot(gsA[0, si])
    ax.set_facecolor(TAG_COLORS[tag] + "55")
    for s_idx, sn in enumerate(snames):
        bv = pred_bands[col][sn]
        ev = float(sess_df.loc[sess_df["session"]==sn, col].iloc[0])
        lo, hi = np.percentile(bv, [2.5, 97.5])
        vp = ax.violinplot(bv, positions=[s_idx], widths=0.62,
                           showmedians=True, showextrema=False)
        for pc in vp["bodies"]: pc.set_facecolor(COL_SYN); pc.set_alpha(0.45)
        vp["cmedians"].set_color(COL_SYN_D); vp["cmedians"].set_linewidth(2)
        ax.plot([s_idx, s_idx], [lo, hi], color=COL_SYN_D, lw=1.2, ls="--", alpha=0.7)
        ax.scatter([s_idx], [ev], color=COL_EMP, zorder=6, s=80, marker="D")
        ax.annotate(_pct(pctile[sn][col]), xy=(s_idx, ev),
                    xytext=(s_idx + 0.30, ev), fontsize=7, color=COL_EMP,
                    va="center",
                    arrowprops=dict(arrowstyle="->", color=COL_EMP, lw=0.8))
    ax.set_xlim(-0.6, n_s - 0.2)
    ax.set_xticks(range(n_s))
    ax.set_xticklabels(
        [f"{sn}\n({sess_df.loc[sess_df['session']==sn,'duration_min'].iloc[0]:.0f} min)"
         for sn in snames], fontsize=6.8)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(f"{ylabel.split(' (')[0]}  {tag}", fontsize=8.3)

ax_leg = fig.add_subplot(gs[1, :]); ax_leg.axis("off")
legend_items = [
    Line2D([0],[0], marker="D", color="w", markerfacecolor=COL_EMP,
           markersize=8, label="Empirical session"),
    plt.Rectangle((0,0),1,1, fc=COL_SYN, alpha=0.45,
                  label="LOSO prediction interval (from other sessions, matched duration)"),
    Line2D([0],[0], color=COL_SYN_D, ls="--", label="95% LOSO prediction interval"),
]
ax_leg.legend(handles=legend_items, loc="center", ncol=3, frameon=False,
              fontsize=7.5, handletextpad=0.5, columnspacing=1.6)

fig.text(0.065, 0.972, "A", fontsize=15, fontweight="bold")
fig.text(0.5, 0.972, "Session-level agreement   |   Null model @ ox = 1.0",
         ha="center", fontsize=9.5, fontweight="bold")
fig.text(0.5, 0.952,
         "(leave-one-session-out: each band uses only the OTHER sessions' dispersion)",
         ha="center", fontsize=7.5, style="italic", color="#555555")

# ----- Panels B–E -----
panelBE_cfg = [
    (gs[2,0], "B", "SWR duration (ms)",   "swr_durs",      "[REPRODUCED]",    (15,150),  30, False, "upper right"),
    (gs[2,1], "C", "SWR cycles",          "swr_cycles",    "[FIT-PRESERVED]", (2, 18),   28, True,  "upper right"),
    (gs[3,0], "D", "Peak frequency (Hz)", "swr_freqs",     "[FIT-PRESERVED]", (100,260), 30, False, "upper right"),
    (gs[3,1], "E", "Coupling lag (ms)",   "coupling_lags", "[FIT-PRESERVED]", (-25,60),  35, False, "upper right"),
]
for cell, letter, label, key, tag, xlim, nbins, mark_cyc, leg_loc in panelBE_cfg:
    ax = fig.add_subplot(cell)
    ec = emp[key]; ec = ec[np.isfinite(ec)]
    sc = syn[key]; sc = sc[np.isfinite(sc)]
    bins = np.linspace(xlim[0], xlim[1], nbins)
    ax.set_facecolor(TAG_COLORS[tag] + "44")
    ax.hist(ec, bins=bins, density=True, alpha=0.70, color=COL_EMP,
            label=f"Empirical (n={len(ec)})")
    ax.hist(sc, bins=bins, density=True, alpha=0.55, color=COL_SYN,
            label=f"Synthetic null@1.0 (n={len(sc)})")
    if len(ec): ax.axvline(np.mean(ec), color=COL_EMP, lw=1.5, ls="--",
                           label=f"Emp mean: {np.mean(ec):.1f}")
    if len(sc): ax.axvline(np.mean(sc), color=COL_SYN, lw=1.5, ls="--",
                           label=f"Syn mean: {np.mean(sc):.1f}")
    if mark_cyc: ax.axvline(3, color="red", lw=1.2, ls=":", label="Min 3 cycles")

    ax.set_xlim(xlim)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)   # headroom so legend/stats clear bars

    r = dist_stats(emp[key], syn[key])
    if r:
        box = (f"KS={r['ks']:.3f} (p={r['p']:.3f})\nEMD={r['emd']:.2f}  "
               f"Δμ={r['ms']-r['me']:+.2f}")
        ax.text(0.03, 0.97, box, transform=ax.transAxes, ha="left", va="top",
                fontsize=7.3,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7",
                          alpha=0.95))
    ax.set_title(f"{label}   {tag}", fontsize=8.3)
    ax.set_xlabel(label); ax.set_ylabel("Density")
    ax.legend(fontsize=6.2, loc=leg_loc, framealpha=0.95,
              borderpad=0.4, handlelength=1.4)
    pos = cell.get_position(fig)
    fig.text(pos.x0 - 0.03, pos.y1 + 0.012, letter, fontsize=15, fontweight="bold")

fig.suptitle(
    "Figure 2.  Synthetic SPW-R library reproduces empirical event statistics (v5.8.3)",
    fontsize=11, fontweight="bold", y=0.998)

# ─── [5] Save + show ─────────────────────────────────────────────────────────
png = FIG_DIR / "Fig2_validation_v5.8.3.png"
pdf = FIG_DIR / "Fig2_validation_v5.8.3.pdf"
fig.savefig(png, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
display(fig); plt.show(); plt.close(fig)
print("\n" + "="*70)
print(f"SAVED:\n  {png}\n  {pdf}")
print("="*70)

"""
Figure 3 (v2.5.11): Orexin-dependent consolidation produces a non-monotonic
memory curve that dissociates from encoding.

  A. Consolidation-dependent memory      — main sweep, 9 OX2R levels
  B. Encoding-dominant                   — encoding varies, consolidation fixed
  C. Consolidation-dominant              — consolidation varies, encoding fixed
  D. Coupled pathways                    — both vary together

Panel D is new in this version.  It is the biologically realistic condition:
in an animal, orexin tone moves encoding and consolidation together.  Showing
that the non-monotonicity survives coupling pre-empts the objection that the
result depends on holding encoding fixed at an arbitrary value.

Test statistics live in the CAPTION, not the panels.  The block at the end
prints every number the caption needs, straight from the pickle.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# ── Bootstrap 95% CI from per-rep DI arrays ────────────────────────────────
def boot_ci(arr_dict, oxs, n_boot=5000, seed=42):
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
    """(mean, lo, hi) for a dual-pathway arm; prefers per-rep arrays."""
    br    = dp[branch_key]
    means = np.array([br['means'][ox] for ox in oxs], float)
    if 'da' in br:
        m, lo, hi = boot_ci(br['da'], oxs)
        return m, lo, hi
    if 'sems' in br:
        s = np.array([br['sems'][ox] for ox in oxs], float)
        return means, means - 1.96 * s, means + 1.96 * s
    return means, means, means

DI_MEAN, DI_LO, DI_HI = boot_ci(DI_ARR, OX)
enc_m, enc_lo, enc_hi = band_for('fixed_consolidation', OX_C)   # encoding varies
con_m, con_lo, con_hi = band_for('fixed_encoding',      OX_C)   # consolidation varies
cpl_m, cpl_lo, cpl_hi = band_for('coupled',             OX_C)   # both vary

if np.allclose(enc_lo, enc_hi) or np.allclose(con_lo, con_hi) or np.allclose(cpl_lo, cpl_hi):
    print("  ⚠ A dual-pathway arm has no per-rep arrays in the pickle — its band is flat.\n"
          "    Check that 'da' is being saved for all three arms.")

# ── Shape checks: warn, never crash ────────────────────────────────────────
# Uses the statistics the caption actually cites.  Spearman rho tolerates a
# single non-monotone step; P(interior) is the right test for a peak, since
# argmax on the means is a single draw.
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
    'legend.fontsize': 8,
})
C_A  = '#3b6ea5'   # muted blue    — composite / main sweep
C_B  = '#3f8f6b'   # muted green   — encoding
C_C  = '#b1543a'   # muted brick   — consolidation
C_D  = '#7d6b9e'   # muted violet  — coupled
C_BL = '#9aa0a6'
C_T  = '#1c1917'
BAND_ALPHA = 0.16

def draw_panel(ax, x, m, lo, hi, color, title, xlabel,
               mark_peak=False, baseline_label=False):
    ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.plot(x, m, '-', color=color, lw=1.8, zorder=3)
    ax.plot(x, m, 'o', color=color, ms=4.5, mfc='white', mew=1.3, zorder=4)
    ax.axhline(NCM, color=C_BL, lw=1.0, ls='--', zorder=0)
    if baseline_label:
        ax.text(x[0], NCM, ' no consolidation', fontsize=7, color=C_BL,
                va='bottom', ha='left', style='italic')
    if mark_peak:
        pi = int(np.argmax(m)); pk = m[pi]
        ax.scatter([x[pi]], [pk], s=70, facecolor='none', edgecolor=color,
                   lw=1.6, zorder=6)
        ax.annotate(f'peak\nOX2R = {x[pi]:g}', xy=(x[pi], pk),
                    xytext=(x[pi], pk + 0.004), fontsize=7.5, color=color,
                    ha='center', va='bottom')
    ax.set_title(title, linespacing=1.35)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0.15, 1.95); ax.set_xticks([0.5, 1.0, 1.5])
    ax.spines[['top', 'right']].set_visible(False)

# shared y-limits so the four panels are directly comparable
ymin = min(DI_LO.min(), enc_lo.min(), con_lo.min(), cpl_lo.min(), NCM) - 0.004
ymax = max(DI_HI.max(), enc_hi.max(), con_hi.max(), cpl_hi.max()) + 0.012

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.6, 7.4), facecolor='white')
gs  = gridspec.GridSpec(2, 2, wspace=0.28, hspace=0.46,
                        left=0.095, right=0.97, top=0.865, bottom=0.085)

axA = fig.add_subplot(gs[0, 0]); plabel(axA, 'A')
draw_panel(axA, ox_arr, DI_MEAN, DI_LO, DI_HI, C_A,
           'Consolidation-dependent memory\n(main sweep)',
           'OX2R level', mark_peak=True, baseline_label=True)
axA.set_ylabel('Discrimination index'); axA.set_ylim(ymin, ymax)

axB = fig.add_subplot(gs[0, 1]); plabel(axB, 'B')
draw_panel(axB, oxC_arr, enc_m, enc_lo, enc_hi, C_B,
           'Encoding-dominant\n(consolidation fixed)',
           'Orexin level (encoding)', mark_peak=False)
axB.set_ylim(ymin, ymax)

axC = fig.add_subplot(gs[1, 0]); plabel(axC, 'C')
draw_panel(axC, oxC_arr, con_m, con_lo, con_hi, C_C,
           'Consolidation-dominant\n(encoding fixed)',
           'OX2R level (consolidation)', mark_peak=True)
axC.set_ylabel('Discrimination index'); axC.set_ylim(ymin, ymax)

axD = fig.add_subplot(gs[1, 1]); plabel(axD, 'D')
draw_panel(axD, oxC_arr, cpl_m, cpl_lo, cpl_hi, C_D,
           'Coupled pathways\n(both vary together)',
           'Orexin tone (both pathways)', mark_peak=True)
axD.set_ylim(ymin, ymax)

fig.suptitle('Figure 3. Orexin-dependent consolidation produces a non-monotonic '
             'memory curve\nthat dissociates from encoding',
             fontsize=11.5, fontweight='bold', y=0.975, color=C_T,
             linespacing=1.4)

out = DIRS['figures']
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'figure_3.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figure_3.pdf', facecolor='white', bbox_inches='tight')
display(fig); plt.show(); plt.close(fig)

# ── Caption numbers, straight from the pickle ──────────────────────────────
print("\nFigure 3 saved\n")
print("CAPTION NUMBERS (do not retype these by hand)")
print("-" * 62)
_c  = D['curvature']; _u = D['u_strength']
print(f"  A  curvature      {_c['curvature']:+.4f}  (sign-prob >= 0: {_c['boot_sign_prob_nonneg']:.4f})")
print(f"     peak           {D['consolidation_regime']['peak']['mean']:.2f} "
      f"[{D['consolidation_regime']['peak']['ci_95'][0]:.2f},"
      f"{D['consolidation_regime']['peak']['ci_95'][1]:.2f}]")
print(f"     P(interior)    {D['consolidation_regime']['p_interior']:.2f}")
print(f"     U-strength     {_u['point']:+.4f} [{_u['ci_95'][0]:+.4f},{_u['ci_95'][1]:+.4f}]")
print(f"     no-consol DI   {NCM:.4f}")
print(f"     benefit        {DI_MEAN.max() - NCM:+.4f} "
      f"({100 * _u['point'] / (DI_MEAN.max() - NCM):.0f}% of it lost off-optimum)")
print(f"     n replicates   {len(DI_ARR[OX[0]])}")
print(f"     enc_trials     {D['consolidation_regime']['enc_trials_used']}")
for lbl, key in [('B  encoding-dominant', 'fixed_consolidation'),
                 ('C  consolidation-dom', 'fixed_encoding'),
                 ('D  coupled', 'coupled')]:
    br = dp[key]
    print(f"  {lbl:<22} rho={br['spearman_rho']:+.3f}  curv={br['curvature']:+.4f}  "
          f"peak={br['peak_mean']:.2f} [{br['peak_ci'][0]:.2f},{br['peak_ci'][1]:.2f}]  "
          f"P(int)={br['p_interior']:.2f}")
print("-" * 62)

"""
Figure 4 (v2.5.11b): Replay truncation sets the descending limb;
                     plasticity gating sets the ascending limb.

  A. Replay duration (t_stop) and truncation exposure   — MODEL OUTPUT
  B. Planned event count and LTP drive gate             — MODEL OUTPUT
  C. DI curves, full model vs frozen-gate ablation      — MODEL OUTPUT
  D. Limb amplitudes with bootstrap CIs                 — MODEL OUTPUT

Changes in this revision (cosmetic only, no data change):
  - legends drawn on the twin axis so the overlaid line cannot cover them
  - panel A legend moved to centre-left (upper right is occupied by the
    truncation curve at high OX2R); p10-p90 band now labelled
  - panel C legend moved to upper right, clear of the low-DI region
  - panel D annotates BOTH limb differences, placed above each group;
    the descending delta near zero is half the dissociation and has to be shown
  - frozen gate value degrades to a text label if absent from the pickle
    instead of printing "nan"

All panels read logged model quantities. No re-simulation in the figure script.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
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
    FROZ_LABEL = f'Drive gate frozen at {_fg:.2f}'
else:
    FROZ_LABEL = 'Drive gate frozen at baseline'
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

ts_arr = np.array([np.nanmean(M[ox]['t_stop'])     for ox in OX])
ts_lo  = np.array([np.nanmean(M[ox]['t_stop_p10']) for ox in OX])
ts_hi  = np.array([np.nanmean(M[ox]['t_stop_p90']) for ox in OX])

di_full, se_full = mean_sem(DI_ARR)
di_froz, se_froz = mean_sem(FL['di_arrays'])

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.linewidth': 0.8, 'axes.labelsize': 9.5,
    'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 7.5,
})
C_REPLAY = '#b1543a'
C_TRUNC  = '#7a3b2e'
C_COUNT  = '#3f8f6b'
C_DRIVE  = '#c08a2e'
C_FULL   = '#b1543a'
C_FROZ   = '#3b6ea5'
C_BL     = '#9aa0a6'
C_T      = '#1c1917'
BAND_ALPHA = 0.16

fig = plt.figure(figsize=(10.4, 7.6), facecolor='white')
gs  = gridspec.GridSpec(2, 2, wspace=0.42, hspace=0.38,
                        left=0.085, right=0.925, top=0.885, bottom=0.075)

# ── A: replay duration + truncation ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0]); plabel(ax, 'A')
ax.fill_between(ox_arr, ts_lo, ts_hi, color=C_REPLAY, alpha=BAND_ALPHA, lw=0, zorder=1)
ax.plot(ox_arr, ts_arr, '-o', color=C_REPLAY, lw=1.8, ms=4.5,
        mfc='white', mew=1.3, zorder=3, label='Replay steps (t$_{stop}$)')
ax.axhline(16, color=C_BL, lw=0.8, ls='--', alpha=0.6)
ax.text(1.90, 16.25, 'full replay (16)', fontsize=6.5, color=C_BL, ha='right')
ax.axhline(10, color=C_BL, lw=0.8, ls=':', alpha=0.5)
ax.text(1.90, 10.25, 'diverge step (10)', fontsize=6.5, color=C_BL, ha='right')
ax.set_xlabel('OX2R level'); ax.set_ylabel('Replay steps', color=C_REPLAY)
ax.set_title('Replay duration and truncation')
ax.set_xlim(0.15, 1.95); ax.set_ylim(7, 17.5)
ax.tick_params(axis='y', colors=C_REPLAY)

ax2 = ax.twinx()
ax2.plot(ox_arr, trunc_exp, '--s', color=C_TRUNC, lw=1.5, ms=3.5,
         mfc='white', mew=0.9, label='Truncation exposure (1−quality)')
ax2.set_ylabel('Truncation exposure', color=C_TRUNC)
ax2.set_ylim(-0.03, 0.75); ax2.tick_params(axis='y', colors=C_TRUNC)
ax.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)

# legend lives on the TWIN axis: artists drawn on ax2 would otherwise
# render on top of a legend belonging to ax
_h_band = Patch(facecolor=C_REPLAY, alpha=BAND_ALPHA, edgecolor='none')
h1, lb1 = ax.get_legend_handles_labels()
h2, lb2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + [_h_band] + h2,
           lb1 + ['p10–p90 across replays'] + lb2,
           loc='center left', fontsize=6.6, frameon=True,
           framealpha=1.0, edgecolor='none')

# ── B: event count + drive gate ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1]); plabel(ax, 'B')
ax.bar(ox_arr, n_plan_m, width=0.09, color=C_COUNT, alpha=0.55,
       edgecolor=C_COUNT, linewidth=0.6, label='Planned events', zorder=2)
ax.set_xlabel('OX2R level'); ax.set_ylabel('Planned replay events', color=C_COUNT)
ax.set_ylim(0, max(n_plan_m) * 1.32); ax.tick_params(axis='y', colors=C_COUNT)
ax.set_title('Event count and plasticity gate')
ax.set_xlim(0.15, 1.95)

ax2 = ax.twinx()
ax2.fill_between(ox_arr, dg_lo, dg_hi, color=C_DRIVE, alpha=BAND_ALPHA, lw=0)
ax2.plot(ox_arr, dg_arr, '-o', color=C_DRIVE, lw=1.8, ms=4.5,
         mfc='white', mew=1.3, zorder=3, label='LTP drive gate')
ax2.set_ylabel('Drive gate  P(LTP)', color=C_DRIVE)
ax2.tick_params(axis='y', colors=C_DRIVE)
ax.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)

_h_dband = Patch(facecolor=C_DRIVE, alpha=BAND_ALPHA, edgecolor='none')
h1, lb1 = ax.get_legend_handles_labels()
h2, lb2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + h2 + [_h_dband],
           lb1 + lb2 + ['95% CI'],
           loc='upper left', fontsize=6.6, frameon=True,
           framealpha=1.0, edgecolor='none')

# ── C: full vs frozen DI curves ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0]); plabel(ax, 'C')
ax.axvspan(ox_arr[0], SPLIT, color='0.93', lw=0, zorder=0)
ax.axvline(SPLIT, color=C_BL, lw=0.8, ls=':', zorder=1)
ax.errorbar(ox_arr, di_full, yerr=se_full, fmt='-o', color=C_FULL, lw=1.8,
            ms=4.5, mfc='white', mew=1.3, capsize=2.5, elinewidth=0.9,
            zorder=3, label='Full model')
ax.errorbar(ox_arr, di_froz, yerr=se_froz, fmt='--s', color=C_FROZ, lw=1.6,
            ms=4.0, mfc='white', mew=1.1, capsize=2.5, elinewidth=0.9,
            zorder=3, label=FROZ_LABEL)
_ylo, _yhi = ax.get_ylim()
ax.text((ox_arr[0] + SPLIT) / 2, _ylo + 0.015 * (_yhi - _ylo), 'ascending limb',
        fontsize=7, color='0.45', ha='center', va='bottom')
ax.set_xlabel('OX2R level'); ax.set_ylabel('Discrimination index')
ax.set_title('Freezing the gate abolishes the rise')
ax.set_xlim(0.15, 1.95)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper right', fontsize=7, frameon=True,
          framealpha=1.0, edgecolor='none')

# ── D: limb amplitudes ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1]); plabel(ax, 'D')
limbs = [('asc', f'ascending\n{OX[0]:.1f} → {SPLIT:.1f}'),
         ('des', f'descending\n{SPLIT:.1f} → {OX[-1]:.1f}')]
x, w  = np.arange(2), 0.34
for off, arm, col, lab in [(-w / 2, 'full', C_FULL, 'Full model'),
                           (+w / 2, 'froz', C_FROZ, 'Gate frozen')]:
    vals = np.array([LD[f'{k}_{arm}']['mean'] for k, _ in limbs])
    lo   = np.array([LD[f'{k}_{arm}']['ci_95'][0] for k, _ in limbs])
    hi   = np.array([LD[f'{k}_{arm}']['ci_95'][1] for k, _ in limbs])
    ax.bar(x + off, vals, w, color=col, alpha=0.85, edgecolor=col,
           linewidth=0.6, label=lab, zorder=2)
    ax.errorbar(x + off, vals, yerr=[np.maximum(0, vals - lo),
                                     np.maximum(0, hi - vals)],
                fmt='none', ecolor=C_T, lw=1.0, capsize=3, zorder=3)

ax.axhline(0, color=C_T, lw=0.7, zorder=1)

# annotate BOTH differences, each above its own group. The descending
# difference sitting at ~0 with a CI touching zero is the second half of
# the dissociation and has to be visible in the panel.
_tops = []
for xi, (k, _) in enumerate(limbs):
    d   = LD[f'{k}_diff']
    top = max(LD[f'{k}_full']['ci_95'][1], LD[f'{k}_froz']['ci_95'][1], 0.0)
    _tops.append(top)
    ax.annotate(f"Δ = {d['mean']:+.3f}\n[{d['ci_95'][0]:+.3f}, {d['ci_95'][1]:+.3f}]",
                xy=(xi, top), xytext=(xi, top + 0.003),
                fontsize=7, color=C_T, ha='center', va='bottom')

_bot = min([LD[f'{k}_{a}']['ci_95'][0] for k, _ in limbs for a in ('full', 'froz')] + [0.0])
_rng = max(_tops) - _bot
ax.set_ylim(_bot - 0.10 * _rng, max(_tops) + 0.40 * _rng)

ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in limbs], fontsize=8)
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel('Limb amplitude  (ΔDI)')
ax.set_title('Gate carries the rise, truncation the fall')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper right', fontsize=7, frameon=True,
          framealpha=1.0, edgecolor='none')

fig.suptitle('Figure 4. Replay truncation sets the descending limb; '
             'plasticity gating sets the ascending limb',
             fontsize=11.5, fontweight='bold', y=0.965, color=C_T)

out = DIRS['figures']
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'figure_4.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figure_4.pdf', facecolor='white', bbox_inches='tight')
display(fig); plt.show(); plt.close(fig)

print("Figure 4 saved")
print(f"  frozen gate value: {FROZ_LABEL}")
print(f"  ascending  full {LD['asc_full']['mean']:+.4f}   "
      f"frozen {LD['asc_froz']['mean']:+.4f}   "
      f"diff {LD['asc_diff']['mean']:+.4f} "
      f"[{LD['asc_diff']['ci_95'][0]:+.4f},{LD['asc_diff']['ci_95'][1]:+.4f}] "
      f"P={LD['asc_diff']['p_pos']:.3f}")
print(f"  descending full {LD['des_full']['mean']:+.4f}   "
      f"frozen {LD['des_froz']['mean']:+.4f}   "
      f"diff {LD['des_diff']['mean']:+.4f} "
      f"[{LD['des_diff']['ci_95'][0]:+.4f},{LD['des_diff']['ci_95'][1]:+.4f}] "
      f"P={LD['des_diff']['p_pos']:.3f}")

import numpy as np

def _curv(d):
    return d.get('curvature', d) if isinstance(d, dict) else d

print("="*68)
print("ROBUSTNESS BATTERY — does the inverted-U survive perturbation?")
print("  (curvature < 0 = peak preserved;  peak should stay interior ~1.0)")
print("="*68)

axes = [
    ('inh_robustness',       'E/I truncation slope k_inh'),
    ('induction_robustness', 'Induction threshold θ / τ'),
    ('consol_lr_robustness', 'Consolidation learning rate'),
    ('dimension_robustness', 'Readout dimension'),
    ('gamma_robustness',     'Truncation gamma'),
    ('divergence_results',   'Task divergence point'),
]
for key, label in axes:
    if key not in D:
        print(f"\n{label}: (not in pickle)"); continue
    print(f"\n{label}  [{key}]")
    for param, v in D[key].items():
        c = _curv(v)
        curv = c if np.isscalar(c) else c.get('curvature', np.nan)
        peak = v.get('peak', np.nan) if isinstance(v, dict) else np.nan
        peak = float(peak) if np.isscalar(peak) else np.nan
        flag = "✓" if (np.isfinite(curv) and curv < 0) else "✗"
        print(f"   {str(param):<12} curvature={curv:+.4f}  peak={peak:.3f}  {flag}")

# headline curve for reference
cc = D['consolidation_regime']['curvature']
print("\n" + "-"*68)
print(f"MAIN curve:    curvature={cc['curvature']:+.4f}  "
      f"peak={D['consolidation_regime']['peak']['mean']:.3f}  "
      f"p_interior={D['consolidation_regime']['p_interior']}")
print(f"Peak shift:    slope={D['peak_shift']['slope_mean']:+.4f}  "
      f"CI={[round(x,3) for x in D['peak_shift']['slope_ci_95']]}  "
      f"p_neg={D['peak_shift']['p_slope_negative']}")

# count how many perturbation conditions preserve the peak
all_curv = []
for key,_ in axes:
    if key in D:
        for v in D[key].values():
            c = _curv(v)
            cv = c if np.isscalar(c) else (c.get('curvature', np.nan) if isinstance(c,dict) else np.nan)
            if np.isfinite(cv): all_curv.append(cv)
n_neg = sum(c < 0 for c in all_curv)
print("-"*68)
print(f"OVERALL: {n_neg}/{len(all_curv)} perturbation conditions preserve "
      f"negative curvature (inverted-U)")

# =============================================================================
# ENCODING-STRENGTH SENSITIVITY  (v2.5.10 API — corrected calls)
#   enc_trials lives inside TaskConfig → sweep via TaskConfig.MWM(enc_trials=et).
#   test_curvature(oxs, da_dict)          ← da is the per-rep DICT, not means
#   bootstrap_continuous_peak(da_dict, oxs)
#
# Requires the v2.5.10 script already executed in this kernel (defines
# run_paired_sweep, TaskConfig, CORE_SWITCHES, DIVERGE_IDX, TASK_DIVERGENCE)
# and D loaded from the pickle. test_curvature and bootstrap_continuous_peak
# are imported from common.stats (see header).
# =============================================================================
import numpy as np, pickle, shutil
from pathlib import Path

ENC_TRIALS_LIST = [3, 5, 8, 10, 15]
N_REPS_ENC      = 30
OX_GRID         = D['ox2r_levels']

enc_sensitivity = {}
print("="*66)
print("ENCODING SENSITIVITY: does the inverted-U survive stronger encoding?")
print("="*66)
print(f"  {'enc_trials':<12}{'curvature':<12}{'p(non-neg)':<12}{'peak':<10}{'verdict'}")
print("  " + "-"*60)

for et in ENC_TRIALS_LIST:
    # enc_trials set on the task; divergence matches the main consolidation curve
    task = TaskConfig.MWM(divergence=TASK_DIVERGENCE, enc_trials=et)

    res = run_paired_sweep(task, OX_GRID, N_REPS_ENC,
                           sw=CORE_SWITCHES, diverge_idx=DIVERGE_IDX)

    # per-rep DI arrays keyed by ox (this dict is what the analysis fns expect)
    da = {ox: np.array([r['discrimination_index'] for r in res[ox]], float)
          for ox in OX_GRID}

    cv = test_curvature(np.array(OX_GRID), da)             # (oxs, da_dict)
    pk = bootstrap_continuous_peak(da, np.array(OX_GRID))  # (da_dict, oxs)

    curv = cv['curvature']
    pnn  = cv.get('boot_sign_prob_nonneg', np.nan)
    peak = pk['mean']
    print(f"  {et:<12}{curv:<+12.4f}{pnn:<12.4f}{peak:<10.3f}"
          f"{'✓ inverted-U' if curv < 0 else '✗ lost'}")

    enc_sensitivity[et] = {
        'curvature':              float(curv),
        'boot_sign_prob_nonneg':  float(pnn),
        'peak':                   float(peak),
        'peak_ci':                [float(x) for x in pk['ci_95']],
        'di_mean':                {float(ox): float(da[ox].mean()) for ox in OX_GRID},
        'n_reps':                 N_REPS_ENC,
    }

n_neg = sum(v['curvature'] < 0 for v in enc_sensitivity.values())
print("  " + "-"*60)
print(f"  {n_neg}/{len(enc_sensitivity)} enc_trials regimes preserve the inverted-U")

# ── save into the pickle (back up first) ──
PKL = FILES['results']
shutil.copy(PKL, PKL.with_suffix('.pkl.bak_enc'))
D['encoding_sensitivity'] = enc_sensitivity
with open(PKL, 'wb') as f:
    pickle.dump(D, f)
print(f"\n  Saved D['encoding_sensitivity'] → {PKL.name}")

print("enc_trials   DI@peak(1.0-ish)   DI@0.3    DI@1.8    (floor rises, shape holds?)")
for et in [3,5,8,10,15]:
    dm = D['encoding_sensitivity'][et]['di_mean']
    oxs = sorted(dm.keys())
    print(f"  {et:<11}{dm[oxs[len(oxs)//2]]:<18.4f}{dm[oxs[0]]:<10.4f}{dm[oxs[-1]]:.4f}")

"""
Figure 5 (v2.5.11): The optimal orexin level shifts with required replay depth
on two independent task axes; the non-monotonicity is robust.

  A. Sequence-length axis   — headline, upper left
  B. Divergence axis        — headline, lower left  (second, independent axis)
  C. Robustness, 6 axes     — control, upper right
  D. k_slope cross-validation — control, lower right

WHAT CHANGED FROM THE v2.5.10 SCRIPT
  - Primary statistic in A and B is now the OX x depth interaction, not the
    bootstrapped vertex.  Four of six sequence-length vertices are clipped at
    the sampled OX2R floor, so the vertex slope is reported as descriptive only.
  - Divergence is PROMOTED out of the robustness panel.  It is a task-structure
    manipulation, not a nuisance parameter, and it cannot serve as both a
    complexity axis and a robustness control in the same figure.  Panel C
    therefore has 6 axes, not 7, and the bar count drops accordingly.
  - Both complexity axes are plotted against required replay depth in STEPS,
    so the two slopes are in the same units and can be compared directly.

REQUIRES in the pickle (all from the v2.5.11 model cell):
  complexity_peaks, peak_shift, complexity_interaction
  divergence_results (with 'peak_ci' and 'diverge_step'), divergence_interaction
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
                       f"interaction tests wired in — this script will not "
                       f"re-simulate or fall back to the censored vertex slope.")

OX_FLOOR, OX_CEIL = 0.30, 1.80

# ── A: sequence-length axis ────────────────────────────────────────────────
cpx      = D['complexity_peaks']
sls      = sorted(cpx.keys()); sls_arr = np.array(sls, float)
cpx_peak = np.array([cpx[sl]['mean']     for sl in sls])
cpx_lo   = np.array([cpx[sl]['ci_95'][0] for sl in sls])
cpx_hi   = np.array([cpx[sl]['ci_95'][1] for sl in sls])
ps       = D['peak_shift']
ci_seq   = D['complexity_interaction']

# ── B: divergence axis ─────────────────────────────────────────────────────
divr     = D['divergence_results']
divs     = sorted(divr.keys())
div_step = np.array([divr[d].get('diverge_step', int(16 * float(d))) for d in divs], float)
div_peak = np.array([divr[d]['peak'] for d in divs])
_HAVE_DIV_CI = all('peak_ci' in divr[d] for d in divs)
if _HAVE_DIV_CI:
    div_lo = np.array([divr[d]['peak_ci'][0] for d in divs])
    div_hi = np.array([divr[d]['peak_ci'][1] for d in divs])
else:
    div_lo = div_hi = div_peak
    print("  ⚠ divergence_results has no 'peak_ci'. Add "
          "\"'peak_ci': p['ci_95'],\" to div_res in the model cell for error bars.")
ci_div = D['divergence_interaction']

# ── C: robustness (divergence REMOVED — it is a headline axis now) ─────────
def _curv(v): return v['curvature'] if isinstance(v, dict) else v
def _peak(v): return v.get('peak', np.nan) if isinstance(v, dict) else np.nan

rob_groups = []
rob_groups.append(('Dimension',
    [(_curv(D['dimension_robustness'][k]), _peak(D['dimension_robustness'][k]))
     for k in sorted(D['dimension_robustness'])]))
ind = D['induction_robustness']
rob_groups.append(('Induction θ/τ',
    [(_curv(ind[k]), _peak(ind[k]))
     for k in sorted(ind, key=lambda s: (s.split('_')[0], float(s.split('_')[1])))]))
rob_groups.append(('Consol LR',
    [(_curv(D['consol_lr_robustness'][k]), _peak(D['consol_lr_robustness'][k]))
     for k in sorted(D['consol_lr_robustness'])]))
inh = D['inh_robustness']
rob_groups.append(('E/I slope',
    [(_curv(inh[k]), _peak(inh[k])) for k in sorted(k for k in inh if k.startswith('slope_'))]))
rob_groups.append(('Trunc. γ',
    [(_curv(D['gamma_robustness'][k]), _peak(D['gamma_robustness'][k]))
     for k in sorted(D['gamma_robustness'])]))
if 'encoding_sensitivity' in D:
    es = D['encoding_sensitivity']
    rob_groups.append(('Enc. trials', [(es[k]['curvature'], es[k]['peak']) for k in sorted(es)]))

flat_curv, flat_peak, flat_grp = [], [], []
for g, items in rob_groups:
    for c, p in items:
        flat_curv.append(c); flat_peak.append(p); flat_grp.append(g)
flat_curv = np.array(flat_curv); flat_peak = np.array(flat_peak)
n_total = len(flat_curv); n_neg = int(np.sum(flat_curv < 0))

print(f"robustness axes plotted: {len(rob_groups)}  (divergence promoted to panel B)")
for g, items in rob_groups:
    print(f"  {g}: {len(items)} conditions")
print(f"  total {n_total} conditions, {n_neg} preserve negative curvature")

# ── D: k_slope ─────────────────────────────────────────────────────────────
k_entries = sorted((float(k.split('_')[1]), inh[k]) for k in inh if k.startswith('slope_'))
k_arr  = np.array([k for k, _ in k_entries])
k_curv = np.array([_curv(v) for _, v in k_entries])
k_peak = np.array([_peak(v) for _, v in k_entries])

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.linewidth': 0.8, 'axes.labelsize': 9.5,
    'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 6.8,
})
C_SEQ  = '#b1543a'
C_DIV  = '#4b8a9c'
C_CURV = '#3b6ea5'
C_PEAK = '#c08a2e'
C_BL   = '#9aa0a6'
C_T    = '#1c1917'
group_palette = {
    'Dimension': '#3b6ea5', 'Induction θ/τ': '#3f8f6b', 'Consol LR': '#b1543a',
    'E/I slope': '#7b5ea7', 'Trunc. γ': '#c08a2e', 'Enc. trials': '#cc5577',
}

def shift_box(ax, ci, ps_desc=None, loc=(0.97, 0.97)):
    """Primary statistic = interaction-implied shift; vertex slope is secondary."""
    sh = ci['implied_shift']
    txt = (f"implied shift = {sh['mean']:+.4f} OX2R/step\n"
           f"95% CI [{sh['ci_95'][0]:+.4f}, {sh['ci_95'][1]:+.4f}]\n"
           f"P(negative) = {sh['p_neg']:.3f}\n"
           f"OX×depth β = {ci['ox_by_length']['mean']:+.5f}, "
           f"P(β<0) = {ci['ox_by_length']['p_neg']:.3f}")
    if ps_desc is not None:
        txt += (f"\nvertex slope {ps_desc['slope_mean']:+.3f} "
                f"(descriptive, see note)")
    ax.text(loc[0], loc[1], txt, transform=ax.transAxes, fontsize=7.3,
            ha='right', va='top', color=C_T,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.8', alpha=0.94))

def peak_panel(ax, x, pk, lo, hi, color, xlabel, title, mark_censored=True):
    cens = (lo <= OX_FLOOR + 1e-6) | (hi >= OX_CEIL - 1e-6)
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0, zorder=1)
    ax.errorbar(x, pk, yerr=[np.maximum(0, pk - lo), np.maximum(0, hi - pk)],
                fmt='none', ecolor=color, elinewidth=1.2, capsize=4,
                capthick=1.0, alpha=0.7, zorder=2)
    ax.plot(x, pk, '-o', color=color, lw=2.0, ms=6.5, mfc='white', mew=1.5, zorder=4)
    if mark_censored and cens.any():
        ax.scatter(x[cens], pk[cens], s=165, facecolor='none', edgecolor=color,
                   lw=1.4, ls=':', zorder=5)
    ax.axhline(1.0, color=C_BL, lw=1, ls='--', zorder=0)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel('Optimal OX2R level', fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=10)
    ax.set_ylim(0.15, 2.0)
    ax.spines[['top', 'right']].set_visible(False)
    return cens

# ── Layout: two headline axes left, two controls right ─────────────────────
fig = plt.figure(figsize=(13, 8.2), facecolor='white')
gs  = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0],
                        wspace=0.28, hspace=0.52,
                        left=0.065, right=0.93, top=0.885, bottom=0.115)

# ═══ A: sequence length ═══
axA = fig.add_subplot(gs[0, 0]); plabel(axA, 'A', x=-0.11, y=1.06)
censA = peak_panel(axA, sls_arr, cpx_peak, cpx_lo, cpx_hi, C_SEQ,
                   'Sequence length (steps)',
                   'Longer sequences lower the optimum')
axA.set_xticks(sls_arr); axA.set_xlim(sls_arr[0] - 2, sls_arr[-1] + 2)
axA.text(sls_arr[-1] + 1.6, 1.03, 'baseline (OX2R = 1.0)', fontsize=7.5,
         color=C_BL, ha='right', va='bottom')
shift_box(axA, ci_seq, ps)

# ═══ B: divergence ═══
axB = fig.add_subplot(gs[1, 0]); plabel(axB, 'B', x=-0.11, y=1.06)
censB = peak_panel(axB, div_step, div_peak, div_lo, div_hi, C_DIV,
                   'Divergence step (steps of shared prefix)',
                   'A second, independent depth manipulation',
                   mark_censored=_HAVE_DIV_CI)
axB.set_xticks(div_step); axB.set_xlim(div_step[0] - 1.5, div_step[-1] + 1.5)
axB.text(div_step[-1] + 1.2, 1.03, 'baseline (OX2R = 1.0)', fontsize=7.5,
         color=C_BL, ha='right', va='bottom')
shift_box(axB, ci_div)

if censA.any() or (_HAVE_DIV_CI and censB.any()):
    fig.text(0.065, 0.035,
             '○ dotted ring: 95% CI of the bootstrapped vertex reaches a sampled OX2R boundary. '
             'Because these estimates are censored, the primary statistic in A and B is the\n'
             'OX × depth interaction fitted to every replicate at every level, which never '
             'locates a vertex and is therefore unaffected by the sampling boundaries.',
             fontsize=7, color=C_T, ha='left', va='bottom')

# ═══ C: robustness ═══
axC = fig.add_subplot(gs[0, 1]); plabel(axC, 'C', x=-0.11, y=1.12)
x = np.arange(n_total)
colors = [group_palette[g] for g in flat_grp]
axC.bar(x, flat_curv, width=0.82, color=colors, alpha=0.72, edgecolor=colors,
        linewidth=0.3, zorder=2)
axC.axhline(0, color=C_T, lw=0.6, zorder=3)
axC.set_ylabel('Curvature q₂', color=C_CURV, fontsize=8.5)
axC.tick_params(axis='y', colors=C_CURV)
axC.set_ylim(min(flat_curv) * 1.20, 0.012)
axC2 = axC.twinx()
axC2.plot(x, flat_peak, 's', color=C_T, ms=3, mfc='white', mec=C_T, mew=0.6, zorder=4)
axC2.axhline(1.0, color=C_BL, lw=0.6, ls=':', alpha=0.5)
axC2.set_ylabel('Peak (□)', color=C_BL, fontsize=8)
axC2.tick_params(axis='y', colors=C_BL); axC2.set_ylim(0.45, 1.25)
axC.spines['top'].set_visible(False); axC2.spines['top'].set_visible(False)
b = 0
for g, items in rob_groups:
    span = len(items)
    if b > 0:
        axC.axvline(b - 0.5, color='0.85', lw=0.6, zorder=0)
    axC.text(b + span / 2 - 0.5, min(flat_curv) * 1.34, g, ha='center', va='top',
             fontsize=5.8, color=group_palette[g], fontweight='bold')
    b += span
axC.set_xticks([])
axC.set_title(f'Robustness · {len(rob_groups)} axes '
              f'({n_neg}/{n_total} preserve non-monotonicity)', fontsize=9.5, pad=6)

# ═══ D: k_slope ═══
axD = fig.add_subplot(gs[1, 1]); plabel(axD, 'D', x=-0.11, y=1.12)
axD.axvspan(0.050, 0.070, color='#3f8f6b', alpha=0.10, zorder=0)
axD.text(0.060, -0.002, 'admissible\n(Kostanšek)', fontsize=6, color='#3f8f6b',
         ha='center', va='top')
axD.bar(k_arr, k_curv, width=0.0045, color=C_CURV, alpha=0.70, edgecolor=C_CURV,
        linewidth=0.5, zorder=2)
axD.axhline(0, color=C_T, lw=0.6, zorder=3)
ki = int(np.argmin(np.abs(k_arr - 0.06)))
axD.annotate('current (0.06)', xy=(k_arr[ki], k_curv[ki]),
             xytext=(k_arr[ki] + 0.001, k_curv[ki] * 0.5), fontsize=6.3, color=C_CURV,
             ha='center', va='top', arrowprops=dict(arrowstyle='->', color=C_CURV, lw=0.9))
axD2 = axD.twinx()
axD2.plot(k_arr, k_peak, '--s', color=C_PEAK, lw=1.3, ms=4, mfc=C_PEAK, mec='white',
          mew=0.8, zorder=4, label='Peak OX2R')
axD2.axhline(1.0, color=C_BL, lw=0.6, ls=':', alpha=0.5)
axD2.set_ylabel('Peak OX2R', color=C_PEAK, fontsize=8)
axD2.tick_params(axis='y', colors=C_PEAK); axD2.set_ylim(0.45, 1.25)
axD.set_xlabel('k_slope (E/I truncation)', fontsize=8.5)
axD.set_ylabel('Curvature q₂', color=C_CURV, fontsize=8.5)
axD.tick_params(axis='y', colors=C_CURV)
axD.set_title('k_slope cross-validation', fontsize=9.5, pad=6)
axD.set_xlim(0.043, 0.077)
axD.spines['top'].set_visible(False); axD2.spines['top'].set_visible(False)
axD2.legend(loc='lower left', fontsize=6.3, frameon=False)

fig.suptitle('Figure 5. The optimal orexin level falls as required replay depth rises, '
             'on two independent task axes',
             fontsize=11.5, fontweight='bold', y=0.975, color=C_T)

out = DIRS['figures']
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'figure_5.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figure_5.pdf', facecolor='white', bbox_inches='tight')
display(fig); plt.show(); plt.close(fig)

print("\nFigure 5 saved")
print("CAPTION NUMBERS")
print("-" * 60)
for nm, ci in [('sequence length', ci_seq), ('divergence step', ci_div)]:
    sh, ib, qd = ci['implied_shift'], ci['ox_by_length'], ci['quadratic']
    print(f"  {nm:<16} shift {sh['mean']:+.4f} "
          f"[{sh['ci_95'][0]:+.4f},{sh['ci_95'][1]:+.4f}]  P(neg)={sh['p_neg']:.3f}")
    print(f"  {'':<16} OX×depth β {ib['mean']:+.5f} "
          f"[{ib['ci_95'][0]:+.5f},{ib['ci_95'][1]:+.5f}]  P(β<0)={ib['p_neg']:.3f}")
    print(f"  {'':<16} concavity β₂ {qd['mean']:+.5f}  P(β₂<0)={qd['p_neg']:.3f}")
print(f"  vertex slope (descriptive) {ps['slope_mean']:+.4f} "
      f"[{ps['slope_ci_95'][0]:+.4f},{ps['slope_ci_95'][1]:+.4f}]  "
      f"P(neg)={ps['p_slope_negative']:.3f}")
print(f"  robustness {n_neg}/{n_total} across {len(rob_groups)} axes")
print("-" * 60)

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

"""
Supplementary Figure S1: Optimal orexin level shifts with memory load and
consolidation budget — directionally consistent with the complexity result,
but floor-censored within the sampled OX2R range (hence qualitative).
  A. Load (n memories held)
  B. Consolidation budget
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

OX_FLOOR, OX_CEIL = 0.30, 1.80   # sampled OX2R range

def extract(peaks_dict):
    xs = sorted(peaks_dict.keys())
    m  = np.array([peaks_dict[x]['mean'] if isinstance(peaks_dict[x], dict) and 'mean' in peaks_dict[x]
                   else peaks_dict[x].get('peak', np.nan) for x in xs])
    lo = np.array([peaks_dict[x].get('ci_95', [np.nan, np.nan])[0] for x in xs])
    hi = np.array([peaks_dict[x].get('ci_95', [np.nan, np.nan])[1] for x in xs])
    return np.array(xs, float), m, lo, hi

load_x, load_m, load_lo, load_hi = extract(D['load_peaks'])
bud_x,  bud_m,  bud_lo,  bud_hi  = extract(D['budget_peaks'])

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.linewidth': 0.8, 'axes.labelsize': 9.5,
    'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 7.5,
})
C_LOAD, C_BUD, C_BL, C_T = '#7b5ea7', '#4b8a9c', '#9aa0a6', '#1c1917'

def draw(ax, xs, m, lo, hi, color, xlabel, title):
    # shade the sampled OX2R range so censoring is visually obvious
    ax.axhspan(OX_FLOOR, OX_CEIL, color='0.93', zorder=0)
    ax.axhline(OX_FLOOR, color=C_BL, lw=0.8, ls=':', zorder=1)
    ax.text(xs[-1], OX_FLOOR, ' sampled floor', fontsize=6.5, color=C_BL,
            va='bottom', ha='right')
    censored = lo <= OX_FLOOR + 1e-6
    ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt='none', ecolor=color,
                elinewidth=1.2, capsize=4, capthick=1.0, alpha=0.7, zorder=2)
    ax.plot(xs, m, '-o', color=color, lw=1.8, ms=6, mfc='white', mew=1.5, zorder=4)
    # ring the floor-censored points
    if censored.any():
        ax.scatter(xs[censored], m[censored], s=150, facecolor='none',
                   edgecolor=color, lw=1.4, ls=':', zorder=5)
    ax.axhline(1.0, color=C_BL, lw=1, ls='--', zorder=1)
    ax.text(xs[0], 1.02, 'baseline', fontsize=7, color=C_BL, va='bottom')
    ax.set_xlabel(xlabel); ax.set_ylabel('Optimal OX2R level')
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.15, OX_CEIL + 0.05)
    ax.set_xticks(xs)
    ax.spines[['top', 'right']].set_visible(False)

fig = plt.figure(figsize=(10, 4.3), facecolor='white')
gs = gridspec.GridSpec(1, 2, wspace=0.30, left=0.08, right=0.96, top=0.84, bottom=0.16)

axA = fig.add_subplot(gs[0]); plabel(axA, 'A')
draw(axA, load_x, load_m, load_lo, load_hi, C_LOAD,
     'Memory load (n items held)', 'Load-dependent optimum')

axB = fig.add_subplot(gs[1]); plabel(axB, 'B')
draw(axB, bud_x, bud_m, bud_lo, bud_hi, C_BUD,
     'Consolidation budget (fraction)', 'Budget-dependent optimum')

fig.suptitle('Supplementary Figure S1.  Optimal orexin level shifts with memory load and '
             'consolidation budget\n(directionally consistent, floor-censored within the sampled range)',
             fontsize=10, fontweight='bold', y=1.02)

# honest footnote
fig.text(0.5, -0.02,
         '○ dotted ring: 95% CI lower bound at the sampled OX2R floor (0.30); '
         'true optimum may lie lower. Trends shown are qualitative.',
         ha='center', fontsize=7, color=C_T, style='italic')

out = DIRS['figures']
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'figS_load_budget.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(out / 'figS_load_budget.pdf', facecolor='white', bbox_inches='tight')
display(fig); plt.show(); plt.close(fig)

# print the censoring summary for your own records
for label, xs, m, lo, hi in [('LOAD', load_x, load_m, load_lo, load_hi),
                              ('BUDGET', bud_x, bud_m, bud_lo, bud_hi)]:
    n_cens = int(np.sum(lo <= OX_FLOOR + 1e-6))
    print(f"{label}: {n_cens}/{len(xs)} conditions floor-censored; "
          f"peak {m[0]:.2f} → {m[-1]:.2f}")
print("Supplementary Fig S1 saved")

