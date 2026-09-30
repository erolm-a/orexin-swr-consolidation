"""
Figure 1 - SWR detection and physiological validation.

Reads:  data_processed/swr_detection.pkl
        data_raw/<session>/*.lfp|.eeg (panel B example traces)
Writes: figures/fig1_swr_detection.{png,pdf,tiff}

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

# =============================================================================
# Figure 1 — SWR detection and physiological validation
#
# Five sessions. Detection uses the band-power channel rule (SPW: top 5-40 Hz
#   power; ripple: top 150-250 Hz power among the remaining channels; noise
#   references: top 300-500 Hz power among the rest). No probe layout is used.
#   Sample numbers are assigned by ROLE, not alphabetically: the three sessions
#   that built the synthetic library are Samples 1-3, the held-out sessions
#   follow as Samples 4-5. Adding sessions therefore never renumbers the
#   existing ones, and the numbering itself shows the train / held-out split.
#   Held-out sessions are drawn dashed and marked with a dagger. Per-session
#   densities (D, F) are drawn only when n >= MIN_N_DENSITY; smaller samples
#   still appear in the median strip and in the pooled distributions.
#   The five-colour session palette was checked with a colour-vision-deficiency
#   validator: the worst pair sits in the 6-8 dE band, which is acceptable only
#   with a second encoding, and the dashed held-out style provides it.
#
# v5: session identifiers are no longer printed on the figure. Each recording
#   is displayed as "Sample N", assigned by sorting the session identifiers so
#   the mapping is stable across re-dumps of the pickle. The real identifiers
#   are still used to locate raw LFP on disk, and the Sample N -> session ID
#   key is printed to stdout for Methods. The in-figure suptitle is gone (the
#   figure number and title belong in the caption, per most journal styles),
#   and the figure is written as PNG, PDF and LZW-compressed TIFF.
#
# v4 addressed four review comments:
#
#   (B) "the Hilbert envelope is per SWR but the light blue line is the
#       average?"  The old panel plotted ONE channel's filtered trace and
#       ONE channel's envelope in the same colour, while the detector
#       actually thresholds an AGGREGATE across channels (SPW: mean of
#       per-channel envelopes; ripple: 75th percentile). Panel B now shows
#       thin per-channel envelopes, the bold aggregate the detector really
#       used, and the k x MAD threshold line, with a legend that names each.
#
#   (C) "Gold" is gone. Every label reads "high-confidence SWR subset".
#
#   (D) 1. Palette is seaborn 'colorblind' for sessions and a single warm
#          ramp for the nested ripple tiers, matched to the empirical red
#          used in Figure 2 (#e05a4b). No default tab10 anywhere.
#       2/3. The stacked histogram and its inset are gone. Per-session
#          DENSITIES are overlaid (comparable across sessions of different
#          n), and medians live in a strip UNDER the histogram sharing the
#          same lag axis — so a positive median cannot fall off a separate
#          inset scale.
#
#   (E) The coarse bins are NOT a plotting artifact. peak_freq is the argmax
#       of a Welch PSD with nperseg=128 at 1250 Hz, so it is quantised to
#       fs/nperseg = 9.77 Hz and only ~5 values are attainable inside the
#       150-200 Hz window. Panel E now marks the attainable grid explicitly,
#       and a new panel F shows the INDEPENDENT, non-quantised estimator the
#       detector already stores (frequency = n_cycles / duration), which
#       settles whether the chunking is measurement or signal.
#
#   Also retained from v3: thresholds read from data['config']; panel A
#   corrected to NREM+Intermediate; representative event chosen by rule;
#   panel C nested tiers; panel D binned on the admissible window.
#
#   Self-contained: reloads the pickle, no dependence on earlier cells.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from scipy import signal, stats
from pathlib import Path
import pickle

try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='paper')      # matches Figure 2
    _HAVE_SNS = True
except Exception:
    _HAVE_SNS = False

# Paths come from orexin_paths (see header). DIRS / FILES are already bound.
PKL_PATH = FILES['detector']

EXAMPLE_SESSION = 'Cicero_09102014'   # real ID: used to locate raw LFP on disk;
                                      # displayed as its Sample label in panel B

# Sessions that built the synthetic library come first (Samples 1-3); sessions
# never used by the generator follow (Samples 4-5). Any other session in the
# pickle is appended after these, alphabetically.
TRAINING_SESSIONS = ['Achilles_10252013', 'Cicero_09102014', 'Cicero_09172014']
HELD_OUT_SESSIONS = ['Buddy_06272013', 'Gatsby_08282013']
MIN_N_DENSITY     = 20                # per-session density drawn only above this n
WINDOW_MS       = 200                 # displayed window around the ripple peak
PAD_MS          = 250                 # loaded then discarded (filter edges)
BASELINE_SEC    = 60                  # local baseline for the threshold line

# Output. TIFF is what most journals want for production; LZW is lossless and
# universally accepted, unlike the JPEG-in-TIFF option. 600 dpi covers the
# common "600 dpi for line art / combination figures" requirement — drop to
# 300 if the file size becomes awkward for a submission portal.
SAVE_DPI      = 300                   # PNG (screen / preprint)
TIFF_DPI      = 600                   # TIFF (production)
TIFF_COMPRESS = 'tiff_lzw'            # None for uncompressed, if a journal insists

# ── Palette ─────────────────────────────────────────────────────────────────
# Taken from Figure 2 so the two figures read as one pair.
#   FIG2_EMP  #e05a4b  empirical           FIG2_SYN  #5b9bd5  synthetic
#   FIG2_SYND #1a5fa8  synthetic, dark
FIG2_EMP, FIG2_SYN, FIG2_SYND = '#e05a4b', '#5b9bd5', '#1a5fa8'

# Detection tiers (C, E): a sequential ramp built from the Figure 2 empirical
# red. Everything in Figure 1 is empirical, so the whole tier ladder stays in
# that hue and only lightness varies.
C_SPW,  C_SPW_E = '#aeb9c4', '#5c6b7a'          # neutral: a parallel stream
C_ALL,  C_ALL_E = '#f4b8ae', '#c0705f'          # tint  of FIG2_EMP
C_SWR,  C_SWR_E = FIG2_EMP,  '#8f2f24'          # =     FIG2_EMP
C_HC,   C_HC_E  = '#8f2f24', '#4a1712'          # shade of FIG2_EMP
C_RAWP          = '#5c6b7a'

# Sessions (D, F), fixed order, never cycled. Slots 1-3 are Figure 2's own
# colours (blue there means "synthetic"; Figure 1 has no synthetic data, so the
# reuse is cosmetic). Slots 4-5 were chosen to pass a colour-vision-deficiency
# check against slots 1-3; the worst pair (green vs red, protan) sits in the
# band that needs a second encoding, which the dashed held-out style supplies.
SESSION_PAL = [FIG2_EMP, FIG2_SYND, FIG2_SYN, '#3a8f3a', '#d4a017']
HELD_LS     = (0, (4, 2))             # dashed: held-out sessions (drawn as lines)

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'figure.dpi': 110, 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ── Load ─────────────────────────────────────────────────────────────────────
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
results = data['datasets']
CFG     = data.get('config', {})
if not CFG:
    raise KeyError("Pickle has no 'config'. This figure reads every printed "
                   "threshold from the detector CONFIG so the figure cannot "
                   "drift from Methods. Re-dump with {'config': CONFIG, ...}.")
print(f"Loaded {len(results)} sessions from {PKL_PATH.name}"
      f"{'' if _HAVE_SNS else '   [seaborn unavailable — fallback palette]'}")

# ── Session labels ──────────────────────────────────────────────────────────
# Figures and text refer to recordings as Sample 1..N, numbered by role:
# training sessions first, held-out sessions next, anything else after. The
# printed key below is what goes into Methods / data availability.
_present = {sess['name'] for sess in results}
SESSION_ORDER = ([s for s in TRAINING_SESSIONS if s in _present] +
                 [s for s in HELD_OUT_SESSIONS if s in _present] +
                 sorted(_present - set(TRAINING_SESSIONS) - set(HELD_OUT_SESSIONS)))
SESSION_LABELS = {nm: f"Sample {i + 1}" for i, nm in enumerate(SESSION_ORDER)}
if len(SESSION_ORDER) > len(SESSION_PAL):
    raise ValueError(f"{len(SESSION_ORDER)} sessions but {len(SESSION_PAL)} session "
                     f"colours. Add a validated colour rather than letting them cycle.")


def is_held_out(name):
    return name not in TRAINING_SESSIONS


def role(name):
    return 'training' if name in TRAINING_SESSIONS else 'held-out'


print("\nSession label key:")
for nm in SESSION_ORDER:
    print(f"  {SESSION_LABELS[nm]:<10} <-  {nm}   ({role(nm)})")


def cfg(key, default=None):
    return CFG.get(key, default)


def rng_str(key, unit=''):
    lo, hi = cfg(key)
    return f"{lo:g}–{hi:g}{unit}"


# ── Helpers ──────────────────────────────────────────────────────────────────
def bandpass(x, fs, lo, hi, order=4):
    hi = min(hi, fs / 2 - 1)
    sos = signal.butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x, axis=0)


def robust_threshold(x, k):
    """Identical to the detector's rule: median + k * 1.4826 * MAD."""
    x = np.asarray(x, float)
    return np.median(x) + k * 1.4826 * np.median(np.abs(x - np.median(x)))


def short(name):
    return SESSION_LABELS.get(name, name)


def step_outline(ax, values, bins, **kw):
    """Density histogram outline drawn as a LINE, not a patch.

    ax.hist(histtype='step') draws a patch, and some matplotlib versions ignore
    custom dash patterns on patches, which silently turned the dashed held-out
    outlines solid. A Line2D honours every dash style.
    """
    h, edges = np.histogram(np.asarray(values, float), bins=bins, density=True)
    xs = np.concatenate([[edges[0]], np.repeat(edges, 2)[1:-1], [edges[-1]]])
    ys = np.concatenate([[0.0], np.repeat(h, 2), [0.0]])
    return ax.plot(xs, ys, drawstyle='default', **kw)[0]


def tick_label(name):
    """Sample label, with a dagger on sessions not used to build the library."""
    return short(name) + ('†' if is_held_out(name) else '')


def mad_scale(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = np.median(np.abs(x - np.median(x)))
    return m if m > 0 else np.nan


def filterbank_tf(x, fs, f_lo=60, f_hi=300, n_bands=34, rel_bw=0.15):
    """Hilbert filter-bank time-frequency map (constant-Q).

    Welch on 128 samples gives ~9.8 Hz bins and destroys the time resolution
    that makes a ripple legible; a filter bank keeps both. Each band is
    z-scored over the padded segment so the display is contrast-normalised.
    """
    centres = np.geomspace(f_lo, min(f_hi, fs / 2 - 20), n_bands)
    out = np.empty((n_bands, len(x)), float)
    for i, fc in enumerate(centres):
        env = np.abs(signal.hilbert(bandpass(x, fs, fc * (1 - rel_bw),
                                             fc * (1 + rel_bw))))
        s = np.std(env)
        out[i] = (env - np.mean(env)) / (s if s > 0 else 1.0)
    return centres, out


# ── Per-session summary ──────────────────────────────────────────────────────
# Iterating over SESSION_ORDER (not the pickle order) keeps colour, legend
# position and Sample number locked to each other in every panel.
session_data = {}
for i, name in enumerate(SESSION_ORDER):
    sess     = next(r for r in results if r['name'] == name)
    swr      = sess['ripple_tiers']['ripples_swr']
    allrip   = sess['ripple_tiers']['ripples_all']
    raw_rip  = sess.get('ripple_events_raw', [])
    nrem_min = sess['quality']['nrem_duration_min']
    lags     = np.asarray(sess['coupling_swr']['coupling_lags_ms'], float)

    session_data[name] = {
        'color':         SESSION_PAL[i],
        'ls':            HELD_LS if is_held_out(name) else '-',
        'peak_freq':     [e['peak_freq'] for e in swr
                          if np.isfinite(e.get('peak_freq', np.nan))],
        'cyc_freq':      [e['frequency'] for e in swr
                          if np.isfinite(e.get('frequency', np.nan))],
        'freq_pairs':    [(e['peak_freq'], e['frequency']) for e in swr
                          if np.isfinite(e.get('peak_freq', np.nan))
                          and np.isfinite(e.get('frequency', np.nan))],
        'raw_freq':      [e['peak_freq'] for e in raw_rip
                          if np.isfinite(e.get('peak_freq', np.nan))],
        'all_freq':      [e['peak_freq'] for e in allrip
                          if np.isfinite(e.get('peak_freq', np.nan))],
        'duration_ms':   [e['duration_ms'] for e in swr],
        'purity':        [e.get('spectral_purity', np.nan) for e in swr],
        'coupling_lags': lags.tolist(),
        'nrem_min':      nrem_min,
        'spw_rate':      sess['quality']['spw_rate'],
        'all_rate':      sess['quality'].get(
                             'ripple_all_rate',
                             len(allrip) / nrem_min if nrem_min else np.nan),
        'swr_rate':      sess['quality']['swr_rate'],
        'hc_rate':       sess['gold']['n_ripple_gold'] / nrem_min if nrem_min else np.nan,
        'n_spw':         len(sess.get('spw_events', [])),
        'n_raw':         len(raw_rip),
        'n_all':         len(allrip),
        'n_swr':         len(swr),
        'n_hc':          sess['gold']['n_ripple_gold'],
        'n_coupled_spw': int(sess['coupling_swr']['n_coupled_spw']),
    }

names     = list(session_data.keys())
all_freq  = np.asarray([f for d in session_data.values() for f in d['peak_freq']], float)
cyc_freq  = np.asarray([f for d in session_data.values() for f in d['cyc_freq']], float)
raw_freq  = np.asarray([f for d in session_data.values() for f in d['raw_freq']], float)
alltier_f = np.asarray([f for d in session_data.values() for f in d['all_freq']], float)
all_lags  = np.asarray([l for d in session_data.values() for l in d['coupling_lags']], float)
all_dur   = np.asarray([x for d in session_data.values() for x in d['duration_ms']], float)
all_pur   = np.asarray([p for d in session_data.values() for p in d['purity']], float)

# ── CONSISTENCY CHECK: the "% of SPWs coupled" sentence ─────────────────────
tot_spw     = sum(d['n_spw'] for d in session_data.values())
tot_swr     = sum(d['n_swr'] for d in session_data.values())
tot_coupled = sum(d['n_coupled_spw'] for d in session_data.values())
tot_lags    = len(all_lags)
print("\n" + "-" * 74)
print("COUPLING BOOKKEEPING  (verifies the '% of SPWs coupled' sentence)")
print("-" * 74)
print(f"  {'session':<20}{'n_SPW':>8}{'n_coupled':>11}{'n_SWR-tier':>12}{'% SPW':>8}")
for nm in names:
    d = session_data[nm]
    print(f"  {short(nm):<20}{d['n_spw']:>8}{d['n_coupled_spw']:>11}{d['n_swr']:>12}"
          f"{100 * d['n_coupled_spw'] / max(1, d['n_spw']):>8.1f}")
print(f"  {'POOLED':<20}{tot_spw:>8}{tot_coupled:>11}{tot_swr:>12}"
      f"{100 * tot_coupled / max(1, tot_spw):>8.1f}")
print(f"  lags in panel D: {tot_lags}  (should equal n_coupled = {tot_coupled})")
if tot_coupled != tot_swr:
    print(f"  ⚠ n_coupled_SPW ({tot_coupled}) ≠ n_SWR-tier ({tot_swr}). Results "
          f"must read {100 * tot_coupled / max(1, tot_spw):.1f}% of SPWs, not "
          f"{100 * tot_swr / max(1, tot_spw):.1f}%.")
else:
    print("  ✓ counts coincide; the reported percentage is safe either way.")

# ── Representative event: chosen by rule, not by seed ────────────────────────
POOLED_MED = {'duration_ms': np.median(all_dur),
              'peak_freq': np.median(all_freq),
              'spectral_purity': np.nanmedian(all_pur)}
POOLED_MAD = {'duration_ms': mad_scale(all_dur),
              'peak_freq': mad_scale(all_freq),
              'spectral_purity': mad_scale(all_pur)}


def typicality(evt):
    tot = 0.0
    for k in POOLED_MED:
        v, s = evt.get(k, np.nan), POOLED_MAD[k]
        if not np.isfinite(v) or not np.isfinite(s):
            return np.inf
        tot += abs(v - POOLED_MED[k]) / s
    return tot


def pick_representative(sess):
    pool = sess['gold'].get('ripple_gold') or sess['ripple_tiers']['ripples_swr']
    tier = 'high-confidence' if sess['gold'].get('ripple_gold') else 'SWR-tier'
    if not pool:
        return None, None
    sc = [typicality(e) for e in pool]
    if not np.isfinite(np.min(sc)):
        return pool[0], tier
    return pool[int(np.argmin(sc))], tier


def load_example_traces(sess):
    """Return everything panel B needs, including the AGGREGATE envelopes the
    detector thresholded and a locally-estimated threshold line."""
    try:
        session_dir = DIRS['data_raw'] / sess['name']
        lfp_files = list(session_dir.glob('*.eeg')) + list(session_dir.glob('*.lfp'))
        if not lfp_files:
            return None
        n_ch = sess['lfp_params']['n_channels']
        fs   = sess['lfp_params']['fs']
        evt, tier = pick_representative(sess)
        if evt is None:
            return None
        spw_chs = list(sess['channel_info']['spw_channels'])
        rip_chs = list(sess['channel_info']['ripple_channels'])

        mmap = np.memmap(lfp_files[0], dtype=np.int16, mode='r').reshape(-1, n_ch)
        peak = int(evt['peak_sec'] * fs)
        halfb = int(BASELINE_SEC * fs / 2)
        b0, b1 = max(0, peak - halfb), min(mmap.shape[0], peak + halfb)
        raw_spw = mmap[b0:b1, spw_chs].astype(np.float32)
        raw_rip = mmap[b0:b1, rip_chs].astype(np.float32)
        del mmap

        spw_band = cfg('spw_band', (5, 40))
        rip_band = cfg('ripple_band', (150, 250))

        # SPW: detector aggregates by MEAN of per-channel Hilbert envelopes
        spw_env_ch = np.abs(signal.hilbert(bandpass(raw_spw, fs, *spw_band), axis=0))
        spw_agg    = np.mean(spw_env_ch, axis=1)
        spw_thr    = robust_threshold(spw_agg, cfg('spw_threshold_mad', 4.0))

        # Ripple: detector aggregates by the 75th PERCENTILE across channels
        rip_filt   = bandpass(raw_rip, fs, *rip_band)
        rip_env_ch = np.abs(signal.hilbert(rip_filt, axis=0))
        rip_agg    = np.percentile(rip_env_ch,
                                   cfg('ripple_envelope_percentile', 75), axis=1)
        rip_thr    = robust_threshold(rip_agg, cfg('ripple_threshold_mad', 5.0))

        half = int((WINDOW_MS / 2) * fs / 1000)
        pad  = int(PAD_MS * fs / 1000)
        c    = peak - b0
        tf_sl = slice(max(0, c - half - pad), min(raw_rip.shape[0], c + half + pad))
        f_c, tf_pad = filterbank_tf(np.mean(raw_rip[tf_sl], axis=1), fs)
        off = (c - half) - tf_sl.start
        tf  = tf_pad[:, off:off + 2 * half]

        sl   = slice(c - half, c + half)
        t_ms = (np.arange(2 * half) - half) / fs * 1000

        meta = {'dur': evt['duration_ms'], 'freq': evt.get('peak_freq', np.nan),
                'cyc': evt.get('frequency', np.nan), 'ncyc': evt.get('n_cycles', np.nan),
                'purity': evt.get('spectral_purity', np.nan), 'tier': tier,
                'nspw_ch': len(spw_chs), 'nrip_ch': len(rip_chs), 'fs': fs,
                'raw_ch': spw_chs[0], 'typ': typicality(evt)}
        return dict(t=t_ms, raw=raw_spw[sl, 0],
                    spw_ch=spw_env_ch[sl], spw_agg=spw_agg[sl], spw_thr=spw_thr,
                    rip_filt=np.mean(rip_filt[sl], axis=1),
                    rip_ch=rip_env_ch[sl], rip_agg=rip_agg[sl], rip_thr=rip_thr,
                    f_c=f_c, tf=tf, meta=meta)
    except Exception as ex:
        print(f"  [panel B] could not load raw traces: {ex}")
        return None


# =============================================================================
# FIGURE
# =============================================================================
fig = plt.figure(figsize=(11.4, 12.9), facecolor='white')
# top was 0.920 while a suptitle occupied the strip above the axes; with the
# title moved to the caption that strip is given back to the panels.
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.24, 1.0, 0.92],
                       width_ratios=[1, 1.15], hspace=0.40, wspace=0.24,
                       left=0.07, right=0.97, top=0.955, bottom=0.055)


def panel_label(ax, letter, dx=-0.12, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left')


def nogrid(*axes):
    for ax in axes:
        ax.grid(False)


# ── A: pipeline (all numbers from CONFIG) ────────────────────────────────────
axA = fig.add_subplot(gs[0, 0]); axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis('off')
nogrid(axA)
panel_label(axA, 'A', dx=-0.02, dy=1.08)
axA.set_title('Signal-processing pipeline', fontweight='bold', loc='left', x=0.04)

states = ' + '.join(cfg('target_states', ['NREM', 'Intermediate']))
steps = [
    (f"Raw LFP — POST sleep\n({states}; {cfg('hc11_lfp_fs', 1250)} Hz)", '#eceff2'),
    ("Channel selection (band power, roles mutually exclusive)\n"
     "SPW: top SW-band power  |  ripple: top ripple-band power of the rest",
     '#e3e9ef'),
    (f"SPW detect {rng_str('spw_band', ' Hz')}\n"
     f"mean of channel envelopes, {cfg('spw_threshold_mad')}×MAD, "
     f"≥{cfg('spw_min_channels')} ch, "
     f"{cfg('spw_min_duration_ms')}–{cfg('spw_max_duration_ms')} ms", '#dce9f7'),
    (f"Ripple detect {rng_str('ripple_band', ' Hz')}\n"
     f"p{cfg('ripple_envelope_percentile', 75)} of channel envelopes, "
     f"{cfg('ripple_threshold_mad')}×MAD, ≥{cfg('ripple_min_cycles')} cycles, "
     f"≥{cfg('ripple_min_channels')} ch\n"
     f"noise veto {rng_str('noise_band', ' Hz')} @ {cfg('noise_veto_threshold_mad')}×MAD",
     '#fbe0da'),
    (f"Tiers — all: purity ≥{cfg('ripple_all_purity_min')}, "
     f"{rng_str('ripple_all_peak_freq_range', ' Hz')}\n"
     f"SWR: coupled ({-cfg('coupling_pre_ms')} .. +{cfg('coupling_post_ms')} ms), "
     f"purity ≥{cfg('ripple_swr_purity_min')}, {rng_str('ripple_swr_peak_freq_range', ' Hz')}",
     '#f6c9c1'),
    (f"High-confidence SWR subset\nSPW {rng_str('spw_gold_duration_range_ms', ' ms')}, "
     f"top {cfg('spw_gold_min_amplitude_pct')}% amp.  |  "
     f"SWR {rng_str('ripple_gold_duration_range_ms', ' ms')}, "
     f"top {cfg('ripple_gold_min_power_pct')}% power", '#efb0a4'),
]
n = len(steps); box_h = 1.30; gap = (10 - n * box_h) / (n + 1)
y = 10 - gap - box_h; centres = []
for label, fc in steps:
    axA.add_patch(FancyBboxPatch((0.6, y), 8.8, box_h,
                                 boxstyle="round,pad=0.06,rounding_size=0.12",
                                 linewidth=0.9, edgecolor='#555', facecolor=fc))
    axA.text(5.0, y + box_h / 2, label, ha='center', va='center', fontsize=6.5)
    centres.append(y + box_h / 2); y -= (box_h + gap)
for y0, y1 in zip(centres[:-1], centres[1:]):
    axA.add_patch(FancyArrowPatch((5.0, y0 - box_h / 2 + 0.03),
                                  (5.0, y1 + box_h / 2 - 0.03),
                                  arrowstyle='-|>', mutation_scale=10,
                                  linewidth=1.0, color='#444'))

# ── B: representative event, with the aggregates the detector thresholded ───
gsB = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.16,
                                       height_ratios=[0.85, 1, 1, 1.2])
axB1 = fig.add_subplot(gsB[0])
axB2 = fig.add_subplot(gsB[1], sharex=axB1)
axB3 = fig.add_subplot(gsB[2], sharex=axB1)
axB4 = fig.add_subplot(gsB[3], sharex=axB1)
nogrid(axB1, axB2, axB3, axB4)
panel_label(axB1, 'B', dx=-0.10, dy=1.28)

ex_sess = next((r for r in results if r['name'] == EXAMPLE_SESSION), results[0])
T = load_example_traces(ex_sess)
if T is not None:
    t, m = T['t'], T['meta']
    axB1.set_title(
        f"Most typical {m['tier']} event — {short(ex_sess['name'])}\n"
        f"{m['dur']:.0f} ms, {m['freq']:.0f} Hz (PSD), purity {m['purity']:.1f}",
        fontsize=8.2, loc='right')

    axB1.plot(t, T['raw'] - np.mean(T['raw']), color='#333', lw=0.7)
    axB1.set_ylabel(f"Raw\n(ch {m['raw_ch']})", fontsize=7.0)

    # SPW row: thin = per-channel envelopes, bold = MEAN aggregate, dashed = threshold
    for j in range(T['spw_ch'].shape[1]):
        axB2.plot(t, T['spw_ch'][:, j], color='#a9c6e2', lw=0.45, alpha=0.8)
    axB2.plot(t, T['spw_agg'], color=FIG2_SYND, lw=1.6)
    axB2.axhline(T['spw_thr'], color=FIG2_SYND, lw=0.9, ls='--', alpha=0.9)
    axB2.set_ylabel(f"SPW env.\n{rng_str('spw_band', ' Hz')}", fontsize=7.0)
    axB2.legend(handles=[
        Line2D([], [], color='#a9c6e2', lw=0.8,
               label=f"single channels (n={m['nspw_ch']})"),
        Line2D([], [], color=FIG2_SYND, lw=1.6, label='detector aggregate = mean'),
        Line2D([], [], color=FIG2_SYND, lw=0.9, ls='--',
               label=f"{cfg('spw_threshold_mad')}×MAD threshold"),
    ], loc='upper left', fontsize=5.6, frameon=True, framealpha=0.9,
        handlelength=1.4, borderpad=0.25, labelspacing=0.22)

    # Ripple row: filtered mean trace + per-channel envelopes + p75 aggregate
    axB3.plot(t, T['rip_filt'], color='#9aa7b4', lw=0.6, alpha=0.9)
    for j in range(T['rip_ch'].shape[1]):
        axB3.plot(t, T['rip_ch'][:, j], color='#f0bdb4', lw=0.45, alpha=0.8)
    axB3.plot(t, T['rip_agg'], color=C_SWR, lw=1.6)
    axB3.axhline(T['rip_thr'], color=C_SWR, lw=0.9, ls='--', alpha=0.9)
    axB3.set_ylabel(f"Ripple\n{rng_str('ripple_band', ' Hz')}", fontsize=7.0)
    axB3.legend(handles=[
        Line2D([], [], color='#9aa7b4', lw=0.8, label='mean filtered trace'),
        Line2D([], [], color='#f0bdb4', lw=0.8,
               label=f"single-channel env. (n={m['nrip_ch']})"),
        Line2D([], [], color=C_SWR, lw=1.6,
               label=f"detector aggregate = p{cfg('ripple_envelope_percentile', 75)}"),
        Line2D([], [], color=C_SWR, lw=0.9, ls='--',
               label=f"{cfg('ripple_threshold_mad')}×MAD threshold"),
    ], loc='upper left', fontsize=5.6, frameon=True, framealpha=0.9,
        handlelength=1.4, borderpad=0.25, labelspacing=0.22)

    vmax = float(np.percentile(T['tf'], 99.5))
    axB4.pcolormesh(t, T['f_c'], T['tf'], cmap='rocket' if _HAVE_SNS else 'magma',
                    vmin=0, vmax=max(vmax, 1e-6), shading='auto', rasterized=True)
    axB4.set_yscale('log')
    axB4.set_yticks([60, 100, 150, 200, 300])
    axB4.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    axB4.set_ylabel('TF\n(Hz)', fontsize=7.0)
    for fb in cfg('ripple_band', (150, 250)):
        axB4.axhline(fb, color='w', lw=0.6, ls=':', alpha=0.75)
    axB4.text(0.985, 0.06, 'filter-bank envelope (z)', transform=axB4.transAxes,
              ha='right', va='bottom', fontsize=5.8, color='w')

    for ax in (axB1, axB2, axB3, axB4):
        ax.axvline(0, color='#b03030', lw=0.8, alpha=0.55)
    axB4.set_xlabel('Time relative to ripple peak (ms)')
    print(f"\n  [panel B] {short(ex_sess['name'])}: most typical event = {m['dur']:.0f} ms, "
          f"{m['freq']:.0f} Hz PSD, {m['cyc']:.0f} Hz cycle-based, "
          f"{m['ncyc']:.1f} cycles, purity {m['purity']:.2f} "
          f"(distance to pooled medians {m['typ']:.2f} MAD units)")
else:
    for ax in (axB1, axB2, axB3, axB4):
        ax.set_xticks([]); ax.set_yticks([])
    axB1.set_title('Representative SWR (raw LFP unavailable here)', fontsize=8.5)
    axB2.text(0.5, 0.5, "Raw .eeg/.lfp not reachable.\nPlace the session under\n"
              "data_raw/ to render example traces.", ha='center', va='center',
              transform=axB2.transAxes, fontsize=8, color='#888')
for ax in (axB1, axB2, axB3):
    plt.setp(ax.get_xticklabels(), visible=False)

# ── C: detection yield, nested ripple tiers beside the SPW stream ────────────
axC = fig.add_subplot(gs[1, 0]); panel_label(axC, 'C')
axC.set_title('Detection yield per session', fontweight='bold', loc='left', x=0.0)
x = np.arange(len(names)); dxL, dxR = -0.21, 0.21
_YTOP = max(max(session_data[k]['spw_rate'] for k in names),
            max(session_data[k]['all_rate'] for k in names)) * 1.32
for xi, nm in zip(x, names):
    d = session_data[nm]
    axC.bar(xi + dxL, d['spw_rate'], width=0.30, color=C_SPW, edgecolor=C_SPW_E,
            linewidth=0.8, zorder=2)
    axC.bar(xi + dxR, d['all_rate'], width=0.34, color=C_ALL, edgecolor=C_ALL_E,
            linewidth=0.8, zorder=2)
    axC.bar(xi + dxR, d['swr_rate'], width=0.21, color=C_SWR, edgecolor=C_SWR_E,
            linewidth=0.8, zorder=3)
    axC.bar(xi + dxR, d['hc_rate'], width=0.10, color=C_HC, edgecolor=C_HC_E,
            linewidth=0.6, zorder=4)
    _lbl = dict(fontsize=6.4, bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                        ec='none', alpha=0.85), zorder=6)
    axC.text(xi + dxL, d['spw_rate'] + 0.35, f"{d['spw_rate']:.1f}", ha='center',
             va='bottom', color=C_SPW_E, **_lbl)
    axC.text(xi + dxR, d['all_rate'] + 0.35, f"{d['all_rate']:.1f}", ha='center',
             va='bottom', color=C_ALL_E, **_lbl)
    if d['all_rate'] - d['swr_rate'] > 0.08 * _YTOP:
        axC.text(xi + dxR, d['swr_rate'] + 0.35, f"{d['swr_rate']:.1f}", ha='center',
                 va='bottom', color=C_SWR_E, **_lbl)
    else:   # too close to the all-tier label: put it beside the stack
        axC.text(xi + dxR + 0.19, d['swr_rate'], f"{d['swr_rate']:.1f}", ha='left',
                 va='center', color=C_SWR_E, **_lbl)
    if d['hc_rate'] > 0.10 * _YTOP:   # only where the label fits inside the bar
        axC.text(xi + dxR, d['hc_rate'] * 0.45, f"{d['hc_rate']:.1f}", ha='center',
                 va='center', fontsize=6.0, color='white', fontweight='bold',
                 rotation=90, zorder=5)
axC.set_xticks(x)
axC.set_xticklabels([f"{tick_label(k)}\n({session_data[k]['nrem_min']:.0f} min)"
                     for k in names], fontsize=7.2)
axC.set_ylabel('Event rate (events / min of POST NREM+Int.)')
_ytop = _YTOP
axC.set_ylim(0, _ytop)
_held_x = [xi for xi, nm in zip(x, names) if is_held_out(nm)]
if _held_x:
    axC.axvspan(min(_held_x) - 0.5, max(_held_x) + 0.5, color='#eef0f2', zorder=0, lw=0)
    axC.text(np.mean(_held_x), _ytop * 0.76, 'held out†', ha='center', va='bottom',
             fontsize=6.8, color='#5c6b7a', style='italic')
axC.set_xlim(-0.5, len(names) - 0.5)
axC.legend(handles=[
    Patch(fc=C_SPW, ec=C_SPW_E, label='SPW detections (parallel stream)'),
    Patch(fc=C_ALL, ec=C_ALL_E,
          label=f"Ripples, all tier (purity ≥{cfg('ripple_all_purity_min')})"),
    Patch(fc=C_SWR, ec=C_SWR_E, label='SWR tier (coupled + strict spectral)'),
    Patch(fc=C_HC, ec=C_HC_E, label='High-confidence SWR subset'),
], loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
    fontsize=6.3, handlelength=1.3, columnspacing=1.0, handletextpad=0.4)

# ── D: coupling lag — per-session densities + a median strip on the same axis ─
gsD = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.06,
                                       height_ratios=[2.6, 1.5])
axD = fig.add_subplot(gsD[0]); axDs = fig.add_subplot(gsD[1], sharex=axD)
panel_label(axD, 'D', dx=-0.10, dy=1.16)
axD.set_title('SPW→SWR temporal coupling', fontweight='bold', loc='left',
              x=0.0, pad=34)

W_LO, W_HI = -cfg('coupling_pre_ms', 20), cfg('coupling_post_ms', 50)
BIN_MS = 2.5
bins_lag = np.arange(W_LO, W_HI + BIN_MS, BIN_MS)
for nm in names:
    d = session_data[nm]
    n_l = len(d['coupling_lags'])
    if n_l >= MIN_N_DENSITY:
        step_outline(axD, d['coupling_lags'], bins_lag, lw=1.5, color=d['color'],
                     ls=d['ls'], label=f"{tick_label(nm)} (n={n_l})")
    else:
        # too few lags for a density; still listed so the reader sees why
        axD.plot([], [], color=d['color'], ls=d['ls'], lw=1.5,
                 label=f"{tick_label(nm)} (n={n_l}, strip only)")
pooled_median = float(np.median(all_lags))
pooled_q1, pooled_q3 = np.percentile(all_lags, [25, 75])
axD.axvline(0, color='#333', lw=1.1)
for edge in (W_LO, W_HI):
    axD.axvline(edge, color='#999', lw=1.0, ls=':')
axD.set_ylabel('Density (within session)')
axD.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
           fontsize=6.4, columnspacing=1.0, handlelength=1.3, handletextpad=0.4)
plt.setp(axD.get_xticklabels(), visible=False)

# median strip: same x axis, so a positive median cannot fall off scale
rows = names + ['POOLED']
for r, nm in enumerate(rows):
    lg = all_lags if nm == 'POOLED' else np.asarray(session_data[nm]['coupling_lags'], float)
    if lg.size == 0:
        continue
    col = '#333333' if nm == 'POOLED' else session_data[nm]['color']
    q1, med, q3 = np.percentile(lg, [25, 50, 75])
    p10, p90 = np.percentile(lg, [10, 90])
    yy = len(rows) - 1 - r
    axDs.plot([p10, p90], [yy, yy], color=col, lw=0.9, alpha=0.55, solid_capstyle='butt')
    axDs.plot([q1, q3], [yy, yy], color=col, lw=3.4, alpha=0.85, solid_capstyle='butt')
    axDs.plot([med], [yy], 'o', color='white', mec=col, mew=1.4, ms=6, zorder=4)
    small = nm != 'POOLED' and lg.size < MIN_N_DENSITY
    axDs.text(W_HI + 3.0, yy, f"{med:+.1f}" + (f" (n={lg.size})" if small else ''),
              va='center', ha='left', fontsize=6.0 if small else 6.4, color='#333333')
axDs.axvline(0, color='#333', lw=1.1)
for edge in (W_LO, W_HI):
    axDs.axvline(edge, color='#999', lw=1.0, ls=':')
axDs.set_yticks(range(len(rows)))
axDs.set_yticklabels([tick_label(k) if k != 'POOLED' else 'Pooled'
                      for k in rows[::-1]], fontsize=6.6)
axDs.set_ylim(-0.7, len(rows) - 0.3)
axDs.set_xlim(W_LO - 4, W_HI + 14)
axDs.set_xlabel(f'SPW→SWR lag (ms)   (negative = ripple precedes SPW peak; '
                f'admissible {W_LO:+g} .. {W_HI:+g} ms)', fontsize=7.6)

# ── E: PSD-based peak frequency (quantised) ─────────────────────────────────
axE = fig.add_subplot(gs[2, 0]); panel_label(axE, 'E')
axE.set_title('Peak frequency (Welch PSD argmax)', fontweight='bold', loc='left', x=0.0)

DET_LO, DET_HI = cfg('ripple_band', (150, 250))
SWR_LO, SWR_HI = cfg('ripple_swr_peak_freq_range', (150, 200))
_fs_ex  = results[0]['lfp_params']['fs']
NPERSEG = cfg('purity_nperseg', 128)
PSD_BIN = _fs_ex / NPERSEG
grid    = np.arange(0, DET_HI + PSD_BIN, PSD_BIN)
grid_in = grid[(grid >= DET_LO - 1e-9) & (grid <= DET_HI + 1e-9)]
n_attain = int(np.sum((grid >= SWR_LO) & (grid <= SWR_HI)))
edges   = np.concatenate([grid_in - PSD_BIN / 2, [grid_in[-1] + PSD_BIN / 2]])

mean_freq = float(np.mean(all_freq)); sd_freq = float(np.std(all_freq))
med_raw   = float(np.median(raw_freq)) if raw_freq.size else np.nan
frac_in   = (float(np.mean((raw_freq >= SWR_LO) & (raw_freq <= SWR_HI)))
             if raw_freq.size else np.nan)

axE.axvspan(SWR_LO, SWR_HI, color=C_SWR, alpha=0.07, zorder=0)
if raw_freq.size:
    axE.hist(raw_freq, bins=edges, density=True, histtype='step', lw=1.6,
             color=C_RAWP, zorder=3,
             label=f'Raw detections (n={raw_freq.size})')
axE.hist(alltier_f, bins=edges, density=True, color=C_ALL, alpha=0.55,
         edgecolor='white', linewidth=0.5, zorder=2,
         label=f'All tier (n={alltier_f.size})')
axE.hist(all_freq, bins=edges, density=True, color=C_SWR, alpha=0.60,
         edgecolor='white', linewidth=0.6, zorder=4,
         label=f'SWR tier (n={all_freq.size})')
axE.axvline(mean_freq, color=C_HC_E, ls='--', lw=1.4, zorder=5,
            label=f'SWR mean {mean_freq:.0f} Hz')
axE.axvline(DET_HI, color='#444', ls='-.', lw=1.0, zorder=5,
            label=f'Ceiling {DET_HI:g} Hz')
_y0, _y1 = axE.get_ylim(); axE.set_ylim(_y0, _y1 * 1.36)
for g in grid_in:
    axE.axvline(g, color='#4a4a4a', lw=0.5, ls=':', alpha=0.5, zorder=6)
axE.set_xlim(DET_LO - 1.5 * PSD_BIN, DET_HI + PSD_BIN)
axE.set_xlabel('Ripple peak frequency (Hz)')
axE.set_ylabel('Probability density')
axE.legend(loc='center right', bbox_to_anchor=(1.0, 0.40), frameon=True,
           framealpha=0.94, edgecolor='#ccc', fontsize=6.2, borderpad=0.4,
           labelspacing=0.3, handlelength=1.4)
axE.text(0.985, 0.985,
         f"raw median = {med_raw:.0f} Hz\n"
         f"{100 * frac_in:.0f}% of raw detections already fall\n"
         f"inside {SWR_LO:g}–{SWR_HI:g} Hz\n"
         f"SWR tier {mean_freq:.0f} ± {sd_freq:.0f} Hz\n"
         f"dotted: PSD grid ({PSD_BIN:.2f} Hz), {n_attain} attainable\n"
         f"values inside {SWR_LO:g}–{SWR_HI:g} Hz",
         transform=axE.transAxes, ha='right', va='top', fontsize=6.4, zorder=8,
         multialignment='left',
         bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.96))

# ── F: cycle-based frequency — the same events, without the PSD grid ─────────
axF = fig.add_subplot(gs[2, 1]); panel_label(axF, 'F', dx=-0.10)
axF.set_title('Same events, cycle-based estimate (not quantised)',
              fontweight='bold', loc='left', x=0.0)
if cyc_freq.size:
    fb = np.linspace(max(50, np.percentile(cyc_freq, 0.5)),
                     np.percentile(cyc_freq, 99.5), 46)
    for nm in names:
        v = np.asarray(session_data[nm]['cyc_freq'], float)
        if v.size >= MIN_N_DENSITY:
            step_outline(axF, v, fb, lw=1.4, color=session_data[nm]['color'],
                         ls=session_data[nm]['ls'], label=tick_label(nm))
        elif v.size:
            axF.plot([], [], color=session_data[nm]['color'], ls=session_data[nm]['ls'],
                     lw=1.4, label=f"{tick_label(nm)} (n={v.size}, pooled only)")
    axF.hist(cyc_freq, bins=fb, density=True, color='#cfd6dd', alpha=0.85,
             edgecolor='none', zorder=0, label='Pooled SWR tier')
    med_cyc = float(np.median(cyc_freq))
    axF.axvline(med_cyc, color=C_HC_E, ls='--', lw=1.4,
                label=f'median {med_cyc:.0f} Hz')
    axF.set_xlabel('Cycle-based frequency,  n$_{cycles}$ / duration  (Hz)')
    axF.set_ylabel('Probability density')
    _y0, _y1 = axF.get_ylim(); axF.set_ylim(_y0, _y1 * 1.30)
    axF.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=6.0,
               ncol=2, columnspacing=0.9, handlelength=1.6, labelspacing=0.3)

    # PSD and cycle-based estimates paired event by event (both finite)
    _pairs = np.asarray([p for d in session_data.values() for p in d['freq_pairs']], float)
    rho, pval = (stats.spearmanr(_pairs[:, 0], _pairs[:, 1])
                 if len(_pairs) > 10 else (np.nan, np.nan))
    axF.text(0.015, 0.955,
             f"median {med_cyc:.0f} Hz  (PSD: {np.median(all_freq):.0f} Hz)\n"
             f"Spearman ρ vs PSD estimate = {rho:+.2f}\n"
             f"n = {cyc_freq.size}",
             transform=axF.transAxes, ha='left', va='top', fontsize=6.6,
             bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.92))
else:
    axF.axis('off')
    axF.text(0.5, 0.5, "No 'frequency' field on SWR-tier events.\n"
                       "Re-dump the detector pickle to enable this panel.",
             ha='center', va='center', transform=axF.transAxes, fontsize=8,
             color='#888')
    rho = np.nan; med_cyc = np.nan

# ── save ─────────────────────────────────────────────────────────────────────
# No suptitle: the figure number and title live in the caption, which is what
# nearly every journal asks for in the submitted artwork file.
png_path  = DIRS['figures'] / 'fig1_swr_detection.png'
pdf_path  = DIRS['figures'] / 'fig1_swr_detection.pdf'
tiff_path = DIRS['figures'] / 'fig1_swr_detection.tiff'

fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

# TIFF goes through Pillow. compression='tiff_lzw' is lossless; a few portals
# reject compressed TIFF, in which case set TIFF_COMPRESS = None above.
_tiff_kw = {'compression': TIFF_COMPRESS} if TIFF_COMPRESS else {}
try:
    fig.savefig(tiff_path, dpi=TIFF_DPI, bbox_inches='tight', facecolor='white',
                pil_kwargs=_tiff_kw)
    _tiff_mb = tiff_path.stat().st_size / 1e6
    _tiff_msg = (f"{tiff_path}   [{TIFF_DPI} dpi, "
                 f"{TIFF_COMPRESS or 'uncompressed'}, {_tiff_mb:.1f} MB]")
except Exception as ex:
    # Pillow missing, or too old for this compression keyword.
    _tiff_msg = f"TIFF not written: {ex}   (pip install -U Pillow)"

try:
    from IPython.display import display; display(fig)
except Exception:
    pass
plt.show()

# =============================================================================
# CAPTION NUMBERS — read these into the caption, do not retype from memory
# =============================================================================
sess_lbl = ', '.join(tick_label(k) for k in names)
nrem_lbl = ', '.join(f"{session_data[k]['nrem_min']:.1f}" for k in names)
rate_lbl = {r: ', '.join(f"{session_data[k][r]:.1f}" for k in names)
            for r in ('spw_rate', 'all_rate', 'swr_rate', 'hc_rate')}
print("\n" + "=" * 74)
print("FIGURE 1 CAPTION NUMBERS")
print("=" * 74)
print(f"  sessions            {len(names)}  ({sess_lbl}); "
      f"training {sum(not is_held_out(k) for k in names)}, "
      f"held out {sum(is_held_out(k) for k in names)}")
print(f"  POST NREM+Int.      "
      f"{sum(session_data[k]['nrem_min'] for k in names):.1f} min ({nrem_lbl})")
print(f"  C  SPW              {tot_spw}; rates {rate_lbl['spw_rate']} /min")
print(f"     ripples all tier {sum(session_data[k]['n_all'] for k in names)}; "
      f"rates {rate_lbl['all_rate']} /min")
print(f"     SWR tier         {tot_swr}; rates {rate_lbl['swr_rate']} /min")
print(f"     high-confidence  {sum(session_data[k]['n_hc'] for k in names)}; "
      f"rates {rate_lbl['hc_rate']} /min")
print(f"  D  lags             n={tot_lags}; pooled median {pooled_median:+.1f} ms "
      f"[IQR {pooled_q1:+.1f}, {pooled_q3:+.1f}]")
for nm in names:
    lg = np.asarray(session_data[nm]['coupling_lags'], float)
    if lg.size:
        print(f"       {short(nm):<14} median {np.median(lg):+.1f} "
              f"[IQR {np.percentile(lg, 25):+.1f}, {np.percentile(lg, 75):+.1f}] ms, "
              f"n={lg.size}")
print(f"     SPWs coupled     {100 * tot_coupled / max(1, tot_spw):.1f}% "
      f"({tot_coupled}/{tot_spw})")
print(f"  E  raw detections   n={raw_freq.size}, median {med_raw:.0f} Hz, "
      f"{100 * frac_in:.1f}% inside {SWR_LO:g}–{SWR_HI:g} Hz")
print(f"     SWR tier         {mean_freq:.0f} ± {sd_freq:.0f} Hz (n={all_freq.size})")
print(f"     PSD grid         {PSD_BIN:.2f} Hz (fs {_fs_ex:g} / nperseg {NPERSEG}); "
      f"{n_attain} attainable values in {SWR_LO:g}–{SWR_HI:g} Hz")
print(f"  F  cycle-based      median {med_cyc:.0f} Hz, Spearman ρ vs PSD "
      f"= {rho:+.2f} (n={len(_pairs)} events with both estimates)")
print("\n" + "=" * 74)
print("SESSION KEY  (for Methods / data availability, not for the figure)")
print("=" * 74)
for nm in SESSION_ORDER:
    print(f"  {SESSION_LABELS[nm]:<10} = {nm}   ({role(nm)})")
print("\n" + "=" * 74)
print("CAPTION SENTENCES  (these were in-figure notes; they belong in the caption)")
print("=" * 74)
print("  Figure 1.  SWR detection and physiological validation.  <- title, now\n"
      "     caption-only; the artwork file carries no title of its own.")
print("  \u2020  Held-out samples were not used to build or calibrate the synthetic\n"
      "     library (Figure 2); they are drawn dashed.")
print("  A  All detection thresholds shown are read from the stored detector\n"
      "     configuration, not transcribed.")
print(f"  B  Thin traces are individual channel envelopes; the bold trace is the\n"
      f"     aggregate the detector thresholds (SPW: mean across "
      f"{cfg('spw_min_channels') and ''}channels; ripple:\n"
      f"     {cfg('ripple_envelope_percentile', 75)}th percentile). The dashed line is the "
      f"{cfg('spw_threshold_mad')}x / {cfg('ripple_threshold_mad')}x MAD threshold,\n"
      f"     reconstructed from a {BASELINE_SEC} s local baseline around the event "
      f"(detection used\n     {cfg('chunk_sec', 120)} s chunks). The event is the SWR-tier "
      f"event closest to the pooled\n     medians of duration, peak frequency and purity.")
print("  C  Ripple tiers are nested subsets (bar widths shrink inward); SPWs are a\n"
      "     separate detection stream, not a superset.")
print(f"  D  Densities are normalised within session and drawn only for n \u2265 "
      f"{MIN_N_DENSITY} lags;\n     smaller samples appear in the strip only. Strip: "
      f"open circle = median,\n     thick bar = IQR, thin line = 10-90%, on the same lag axis.")
print(f"  E  Peak frequency is the argmax of a Welch PSD (nperseg={NPERSEG} at "
      f"{_fs_ex:g} Hz) and is\n     therefore quantised to a {PSD_BIN:.2f} Hz grid, with "
      f"{n_attain} attainable values inside\n     {SWR_LO:g}-{SWR_HI:g} Hz. The "
      f"{DET_LO:g} Hz edge is a construction floor, not a result.")
print("  F  Cycle-based frequency (n_cycles / duration) is independent of the PSD\n"
      "     grid. It averages over the whole event, whereas the PSD argmax tracks the\n"
      "     dominant component, so the two estimators are not interchangeable.")
print(f"\nSaved:\n  {png_path}\n  {pdf_path}\n  {_tiff_msg}")
