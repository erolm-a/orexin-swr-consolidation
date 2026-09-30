"""
Stage 2 - synthetic SPW/ripple library generation.

Reads:  data_processed/swr_detection.pkl
Writes: models/synthetic_libraries.pkl    agent libraries, 200 trials
        models/generator_config.pkl       baselines and calibration
        models/validation_null.pkl        null model, 5000 trials

Fits per-variable distributions to the detected events, applies the
receptor-specific modulation models, and samples synthetic event
libraries across the OX2R grid.

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

from orexin_paths import FILES, ensure_dirs, load_artifact, stamp  # noqa: E402
from orexin_keys import ox_key  # noqa: E402
from common.utils import (  # noqa: E402
    compute_isis_within_intervals,
    compute_min_duration_ms,
    ramp_to_one,
    safe_concat,
    sample_trunc_norm,
    stable_seed,
)

ensure_dirs()

# =============================================================================
# SYNTHETIC SPW & RIPPLE GENERATOR
# =============================================================================
"""
Synthetic SPW-ripple generator.

Distribution families
---------------------
  SPW amplitude     lognormal. Supported both by the literature on lognormal
                    statistics for hippocampal magnitudes (Mizuseki and
                    Buzsaki 2013; Buzsaki and Mizuseki 2014) and by AIC
                    comparison on the detected events.
  Ripple power      whichever of gamma and lognormal fits better by AIC. No
                    literature precedent mandates a family here, so the
                    comparison is run and the winner used. Both fits are
                    stored for reporting.
  Durations, ISIs   gamma, as waiting-time processes.

FittedDist gives all of them one sampling interface. Gamma fits are kept in
the pickle metadata for every variable, so the robustness reporting can use
them.

Training and held-out sessions
------------------------------
The detector pickle holds every session it processed, including the ones held
out of library construction. The generator reads only GENERATOR_SESSIONS, so
the library never learns from a session Figure 2 treats as held out. Both
output pickles record 'source_sessions', and Figure 2 stops if a held-out
session appears there.

Outputs
-------
  synthetic_libraries.pkl   agent libraries, 200 trials per condition
  generator_config.pkl      baselines and calibration
  validation_null.pkl       null model at ox = 1.0, 5000 trials, seed 777,
                            for Figure 2 only

Any cached Table 1 calibration library is removed at the end of the run, since
it was drawn from the previous generator state; Table 1 rebuilds it on demand.

Grid: 15 OX2R levels matching the agent sweep. SEQUENCE_LENGTH = 16.
"""

# =============================================================================
# SETUP
# =============================================================================
import numpy as np
from scipy import stats
from pathlib import Path
import pickle
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SYNTHETIC SPW-RIPPLE GENERATOR")
print("="*70)

# Paths come from orexin_paths (see header). DIRS / FILES are already bound.

# =============================================================================
# CANONICAL KEY FORMATTER — used in both generator AND agent
# =============================================================================
# ox_key itself lives in orexin_keys so the generator and the agent cannot
# disagree about how a float becomes a string.

# Smoke-test the formatter
_key_checks = [(0.3, "0.3"), (1.0, "1"), (0.85, "0.85"), (1.05, "1.05"),
               (0.7, "0.7"), (1.15, "1.15"), (1.8, "1.8")]
for val, expected in _key_checks:
    assert ox_key(val) == expected, f"ox_key({val}) = {ox_key(val)!r}, expected {expected!r}"
print(f"✓ ox_key formatter verified: {[f'{v}→{ox_key(v)}' for v, _ in _key_checks]}")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
MIN_RIPPLE_CYCLES          = 3
UNCOUPLED_RATE_FRACTION    = 0.5
MAX_RIPPLE_DURATION_MS     = 250.0
USE_EMPIRICAL_FREQ_SAMPLING = True

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# stable_seed, safe_concat, sample_trunc_norm, compute_min_duration_ms,
# compute_isis_within_intervals and ramp_to_one come from common.utils.

# =============================================================================
# DISTRIBUTION COMPARISON: GAMMA vs LOGNORMAL
# =============================================================================
# Buzsáki (2015) and Mizuseki & Buzsáki (2013) report lognormal statistics
# for SPW-R magnitudes. We fit both gamma and lognormal to each positive-valued
# feature and select the better model by AIC. For durations and ISIs (waiting
# times), gamma remains the default because the generative process is better
# described by a rate-and-shape parameterisation.

class FittedDist:
    """
    Wrapper that stores a fitted distribution (gamma or lognormal) and
    provides a uniform sampling interface so the rest of the generator
    doesn't need to know which family was selected.

    Attributes
    ----------
    family : str          'gamma' or 'lognorm'
    params : tuple        scipy params (shape, loc, scale)
    aic    : float        AIC of the fit
    bic    : float        BIC of the fit
    ks_stat : float       KS statistic vs empirical data
    ks_p    : float       KS p-value
    """
    def __init__(self, family: str, params: tuple,
                 aic: float, bic: float, ks_stat: float, ks_p: float):
        self.family  = family
        self.params  = params
        self.aic     = aic
        self.bic     = bic
        self.ks_stat = ks_stat
        self.ks_p    = ks_p

    def rvs(self, size=None, random_state=None):
        """Sample from the fitted distribution."""
        if self.family == 'gamma':
            return stats.gamma.rvs(*self.params, size=size, random_state=random_state)
        else:
            return stats.lognorm.rvs(*self.params, size=size, random_state=random_state)

    def __repr__(self):
        return (f"FittedDist({self.family}, AIC={self.aic:.1f}, "
                f"BIC={self.bic:.1f}, KS={self.ks_stat:.4f}, p={self.ks_p:.4f})")


def compare_gamma_lognormal(data: np.ndarray, label: str = "",
                             prefer: str = "aic") -> dict:
    """
    Fit gamma and lognormal to positive-valued data, compare via AIC, BIC,
    and KS test, and return a dict with both FittedDist objects and the winner.

    Parameters
    ----------
    data   : 1-D array of positive observations
    label  : human-readable name for printing
    prefer : 'aic' or 'bic' — criterion for selecting winner

    Returns
    -------
    dict with keys: 'gamma', 'lognorm', 'winner', 'winner_name', 'delta_aic',
                    'delta_bic', 'label'
    """
    x = data[data > 0].copy()
    n = len(x)
    k_params = 2  # both families have 2 free params when loc is fixed

    # --- Gamma (loc fixed at 0) ---
    g_params = stats.gamma.fit(x, floc=0)
    g_ll     = np.sum(stats.gamma.logpdf(x, *g_params))
    g_aic    = 2 * k_params - 2 * g_ll
    g_bic    = k_params * np.log(n) - 2 * g_ll
    g_ks, g_p = stats.kstest(x, 'gamma', args=g_params)

    # --- Lognormal (loc fixed at 0) ---
    ln_params = stats.lognorm.fit(x, floc=0)
    ln_ll     = np.sum(stats.lognorm.logpdf(x, *ln_params))
    ln_aic    = 2 * k_params - 2 * ln_ll
    ln_bic    = k_params * np.log(n) - 2 * ln_ll
    ln_ks, ln_p = stats.kstest(x, 'lognorm', args=ln_params)

    gamma_fit = FittedDist('gamma',   g_params,  g_aic,  g_bic,  g_ks,  g_p)
    ln_fit    = FittedDist('lognorm', ln_params, ln_aic, ln_bic, ln_ks, ln_p)

    criterion = prefer.lower()
    if criterion == 'bic':
        winner = ln_fit if ln_bic < g_bic else gamma_fit
    else:  # default AIC
        winner = ln_fit if ln_aic < g_aic else gamma_fit

    delta_aic = g_aic - ln_aic   # positive ⇒ lognormal better
    delta_bic = g_bic - ln_bic

    result = {
        'gamma': gamma_fit, 'lognorm': ln_fit,
        'winner': winner, 'winner_name': winner.family,
        'delta_aic': delta_aic, 'delta_bic': delta_bic,
        'label': label, 'n': n,
    }

    # Print comparison
    tag = f"  [{label}]" if label else ""
    print(f"{tag}  n={n}")
    print(f"    Gamma    — AIC={g_aic:>12.1f}  BIC={g_bic:>12.1f}  "
          f"KS={g_ks:.4f} (p={g_p:.4f})  params=({g_params[0]:.3f}, loc={g_params[1]:.3f}, scale={g_params[2]:.3f})")
    print(f"    Lognorm  — AIC={ln_aic:>12.1f}  BIC={ln_bic:>12.1f}  "
          f"KS={ln_ks:.4f} (p={ln_p:.4f})  params=(s={ln_params[0]:.3f}, loc={ln_params[1]:.3f}, scale={ln_params[2]:.3f})")
    arrow = "◀" if winner.family == 'lognorm' else "▶"
    print(f"    ΔAIC={delta_aic:+.1f} (pos⇒lognorm better)  "
          f"ΔBIC={delta_bic:+.1f}  →  WINNER: {winner.family} {arrow}")

    return result

# =============================================================================
# LOAD DETECTOR OUTPUT
# =============================================================================
print("\n[1/7] Loading SWR detection data...")

data_file = FILES['detector']
if not data_file.exists():
    raise FileNotFoundError(f"Data file not found: {data_file}\nRun the detection stage first.")

# Version-checked load: a detector pickle from an older release raises here
# rather than being fitted as if it came from the current selection rule.
data = load_artifact('detector')

datasets_raw = data['datasets']

# The library is built from the TRAINING sessions only. Held-out and excluded
# sessions stay in the detector pickle (Figure 1 shows them) but must never
# reach the generator, or the held-out validation in Figure 2 is circular.
GENERATOR_SESSIONS = ['Achilles_10252013', 'Cicero_09102014', 'Cicero_09172014']
_all_names = [s['name'] for s in datasets_raw]
_missing = [s for s in GENERATOR_SESSIONS if s not in _all_names]
if _missing:
    raise KeyError(f"Training sessions missing from the detector pickle: {_missing}")
datasets_raw = [s for s in datasets_raw if s['name'] in GENERATOR_SESSIONS]
print(f"  Library built from {len(datasets_raw)} training sessions: {GENERATOR_SESSIONS}")
print(f"  Not used by the generator: {sorted(set(_all_names) - set(GENERATOR_SESSIONS))}")
# Provenance: the detector stamps itself with PIPELINE_VERSION, which
# load_artifact() has already verified above. Record that, not a guess.
detector_version = data.get('pipeline_version', '<unstamped>')
config       = data.get('config', {})
print(f"  ✓ Loaded {len(datasets_raw)} sessions (detector {detector_version})")

# =============================================================================
# EXTRACT FEATURES
# =============================================================================
print("\n[2/7] Extracting features from detector output...")

datasets = []
for sess in datasets_raw:
    name = sess['name']
    nrem_intervals  = sess['nrem_intervals']
    total_duration_sec = nrem_intervals['total_sec']

    if 'intervals_sec' in nrem_intervals:
        interval_tuples = nrem_intervals['intervals_sec']
    elif 'intervals' in nrem_intervals:
        interval_tuples = nrem_intervals['intervals']
    elif 'start_times' in nrem_intervals and 'end_times' in nrem_intervals:
        interval_tuples = list(zip(nrem_intervals['start_times'], nrem_intervals['end_times']))
    else:
        interval_tuples = [(0.0, total_duration_sec)]

    spw_events     = sess['spw_events']
    n_spw          = len(spw_events)
    spw_durations  = np.array([e['duration_ms']  for e in spw_events])
    spw_amplitudes = np.array([e['amplitude']     for e in spw_events])
    spw_times      = np.array([e['peak_sec']      for e in spw_events])
    spw_isis       = compute_isis_within_intervals(spw_times, interval_tuples)

    swr_events       = sess['ripple_tiers']['ripples_swr']
    n_swr            = len(swr_events)
    swr_durations    = np.array([e['duration_ms']          for e in swr_events])
    swr_powers       = np.array([e['power']                for e in swr_events])
    swr_frequencies  = np.array([e.get('peak_freq', np.nan) for e in swr_events])

    swr_cycles = []
    swr_cycles_needs_patch = []
    for e in swr_events:
        if 'n_cycles' in e and e['n_cycles'] is not None:
            swr_cycles.append(e['n_cycles']); swr_cycles_needs_patch.append(False)
        else:
            pf = e.get('peak_freq', np.nan)
            if np.isfinite(pf):
                swr_cycles.append(e['duration_ms'] / 1000 * pf); swr_cycles_needs_patch.append(False)
            else:
                swr_cycles.append(np.nan); swr_cycles_needs_patch.append(True)
    swr_cycles = np.array(swr_cycles, dtype=float)
    swr_cycles_needs_patch = np.array(swr_cycles_needs_patch, dtype=bool)

    swr_times   = np.array([e['peak_sec'] for e in swr_events])
    ripple_isis = compute_isis_within_intervals(swr_times, interval_tuples)

    all_ripples    = sess['ripple_tiers']['ripples_all']
    n_ripples_all  = len(all_ripples)

    coupling_info     = sess['coupling_swr']
    coupling_lags     = coupling_info['coupling_lags_ms']
    n_coupled_spw     = coupling_info['n_coupled_spw']
    coupled_spw_mask  = np.asarray(coupling_info['coupled_spw_mask'], dtype=bool)

    if coupled_spw_mask.shape[0] != spw_amplitudes.shape[0]:
        coupled_spw_amplitudes   = np.array([])
        uncoupled_spw_amplitudes = np.array([])
    else:
        coupled_spw_amplitudes   = spw_amplitudes[ coupled_spw_mask]
        uncoupled_spw_amplitudes = spw_amplitudes[~coupled_spw_mask]

    coupling_pre_ms  = config.get('coupling_pre_ms',  20)
    coupling_post_ms = config.get('coupling_post_ms', 50)

    ripples_per_spw = []
    for spw in spw_events:
        spw_peak     = spw['peak_sec']
        window_start = spw_peak - coupling_pre_ms  / 1000
        window_end   = spw_peak + coupling_post_ms / 1000
        n_in_window  = np.sum((swr_times >= window_start) & (swr_times <= window_end))
        if n_in_window > 0:
            ripples_per_spw.append(n_in_window)
    ripples_per_spw = np.array(ripples_per_spw) if ripples_per_spw else np.array([])

    datasets.append({
        'name': name, 'total_duration_sec': total_duration_sec,
        'interval_tuples': interval_tuples,
        'n_spw': n_spw, 'spw_durations': spw_durations,
        'spw_amplitudes': spw_amplitudes, 'spw_times': spw_times, 'spw_isis': spw_isis,
        'n_swr_ripples': n_swr, 'swr_ripple_durations': swr_durations,
        'swr_ripple_powers': swr_powers, 'swr_ripple_frequencies': swr_frequencies,
        'swr_ripple_cycles': swr_cycles, 'swr_cycles_needs_patch': swr_cycles_needs_patch,
        'swr_ripple_times': swr_times, 'ripple_isis': ripple_isis,
        'n_ripples': n_ripples_all, 'coupling_lags': coupling_lags,
        'n_coupled_spw': n_coupled_spw,
        'ripples_per_spw_distribution': ripples_per_spw,
        'coupled_spw_amplitudes': coupled_spw_amplitudes,
        'uncoupled_spw_amplitudes': uncoupled_spw_amplitudes,
    })
    print(f"  {name}: {n_spw} SPW, {n_swr} SWR, "
          f"ISIs: {len(spw_isis)} (from {len(interval_tuples)} intervals)")

print(f"\n  ✓ Extracted features from {len(datasets)} sessions")

# =============================================================================
# POOL DATA AND COMPUTE BASELINES
# =============================================================================
print("\n[3/7] Computing baseline statistics...")

all_spw_durations  = safe_concat([d['spw_durations']  for d in datasets])
all_spw_amplitudes = safe_concat([d['spw_amplitudes'] for d in datasets])
all_spw_isis       = safe_concat([d['spw_isis']       for d in datasets], allow_empty=True)
if len(all_spw_isis) == 0:
    print("  ⚠ No SPW ISIs, using fallback [1.0]")
    all_spw_isis = np.array([1.0])

all_swr_durations  = safe_concat([d['swr_ripple_durations']  for d in datasets])
all_swr_powers     = safe_concat([d['swr_ripple_powers']     for d in datasets])
all_swr_frequencies = safe_concat([d['swr_ripple_frequencies'] for d in datasets])
all_ripple_isis    = safe_concat([d['ripple_isis']            for d in datasets], allow_empty=True)
if len(all_ripple_isis) == 0:
    all_ripple_isis = np.array([1.0])

all_coupling_lags     = safe_concat([d['coupling_lags']           for d in datasets], allow_empty=True)
all_coupled_spw_amps  = safe_concat([d['coupled_spw_amplitudes']  for d in datasets], allow_empty=True)
all_uncoupled_spw_amps = safe_concat([d['uncoupled_spw_amplitudes'] for d in datasets], allow_empty=True)

total_sec         = sum(d['total_duration_sec'] for d in datasets)
total_spw         = sum(d['n_spw']            for d in datasets)
total_ripples     = sum(d['n_ripples']         for d in datasets)
total_swr         = sum(d['n_swr_ripples']     for d in datasets)
total_coupled_spw = sum(d['n_coupled_spw']     for d in datasets)

BASELINE_SPW_RATE          = total_spw     / total_sec
BASELINE_RIPPLE_RATE       = total_ripples / total_sec
BASELINE_SWR_RATE          = total_swr     / total_sec
BASELINE_P_SPW_HAS_RIPPLE  = total_coupled_spw / total_spw if total_spw > 0 else 0.5

ISI_MEAN   = np.mean(all_spw_isis)
ISI_MEDIAN = np.median(all_spw_isis)
ISI_STD    = np.std(all_spw_isis)

EMPIRICAL_RPS          = safe_concat([d['ripples_per_spw_distribution'] for d in datasets], allow_empty=True)
if len(EMPIRICAL_RPS) == 0:
    EMPIRICAL_RPS = np.array([1])
BASELINE_RIPPLES_PER_SPW = np.mean(EMPIRICAL_RPS)
BASELINE_RPS_P_MULTI     = np.mean(EMPIRICAL_RPS > 1)

valid_freq_all    = all_swr_frequencies[np.isfinite(all_swr_frequencies)]
valid_freq_invivo = valid_freq_all[(valid_freq_all >= 150) & (valid_freq_all <= 250)]
BASELINE_FREQ_MEAN = np.mean(valid_freq_invivo) if len(valid_freq_invivo) > 0 else 175.0
BASELINE_FREQ_STD  = np.std(valid_freq_invivo)  if len(valid_freq_invivo) > 0 else 15.0

EMPIRICAL_FREQS_INVIVO = valid_freq_invivo.copy()
valid_freq_paper       = valid_freq_all[(valid_freq_all >= 100) & (valid_freq_all <= 175)]
EMPIRICAL_FREQS_PAPER  = valid_freq_paper.copy() if len(valid_freq_paper) > 0 else valid_freq_invivo.copy()

print(f"  Frequency: {BASELINE_FREQ_MEAN:.1f} ± {BASELINE_FREQ_STD:.1f} Hz")

# Patch cycles
print("\n  Patching cycles...")
n_patched_total = 0
for d in datasets:
    cyc = d['swr_ripple_cycles'].copy()
    needs_patch = d['swr_cycles_needs_patch']
    if np.any(needs_patch):
        cyc[needs_patch] = (d['swr_ripple_durations'][needs_patch] / 1000) * BASELINE_FREQ_MEAN
        n_patched_total += np.sum(needs_patch)
    d['swr_ripple_cycles'] = cyc
print(f"  ✓ Patched {n_patched_total} cycle values")

all_swr_cycles = safe_concat([d['swr_ripple_cycles'] for d in datasets])
valid_cycles   = all_swr_cycles[all_swr_cycles >= MIN_RIPPLE_CYCLES]
BASELINE_CYCLES_MEAN = np.mean(valid_cycles) if len(valid_cycles) > 0 else 6.0
BASELINE_CYCLES_STD  = np.std(valid_cycles)  if len(valid_cycles) > 0 else 2.0

if len(all_coupled_spw_amps) == 0 or len(all_uncoupled_spw_amps) == 0:
    COUPLING_AMP_SENSITIVITY = 0.0
else:
    amps_combined = np.concatenate([all_coupled_spw_amps, all_uncoupled_spw_amps])
    labels        = np.concatenate([np.ones(len(all_coupled_spw_amps)),
                                    np.zeros(len(all_uncoupled_spw_amps))])
    amp_median    = np.median(amps_combined)
    frac_high     = np.mean(labels[amps_combined >  amp_median])
    frac_low      = np.mean(labels[amps_combined <= amp_median])
    COUPLING_AMP_SENSITIVITY = (frac_high - frac_low) / amp_median

EMP_SWR_DUR_MEAN       = np.mean(all_swr_durations)
EMP_SWR_CYCLES_MEAN    = np.mean(valid_cycles) if len(valid_cycles) > 0 else BASELINE_CYCLES_MEAN
EMP_COUPLING_LAG_MEAN  = np.mean(all_coupling_lags) if len(all_coupling_lags) > 0 else 0

print(f"\n  Total NREM: {total_sec/60:.1f} min")
print(f"  SPW: {total_spw} events, {BASELINE_SPW_RATE*60:.2f}/min")
print(f"  SPW ISI (interval-aware): mean={ISI_MEAN:.3f}s, n={len(all_spw_isis)}")
print(f"  SWR: {total_swr} ({BASELINE_SWR_RATE*60:.2f}/min)")
print(f"  P(SPW→ripple): {BASELINE_P_SPW_HAS_RIPPLE:.1%}")
print(f"  Coupling amp sensitivity: {COUPLING_AMP_SENSITIVITY:.6f}")

# =============================================================================
# MODULATION CLASSES
# =============================================================================
@dataclass
class SPWMod:
    rate: float = 1.0; amplitude: float = 1.0
    duration: float = 1.0; isi_regularity: float = 1.0

@dataclass
class RippleMod:
    rate: float = 1.0; duration: float = 1.0; power: float = 1.0
    frequency: float = 1.0; multi_ripple_tendency: float = 1.0

@dataclass
class CouplingMod:
    coupling_prob: float = 1.0; amplitude_sensitivity: float = 1.0; lag_precision: float = 1.0

# =============================================================================
# OREXIN MODELS
# =============================================================================
print("\n[4/7] Defining orexin models...")

class OrexinModel:
    def __init__(self, name, description, paper_aligned=False):
        self.name = name; self.description = description; self.paper_aligned = paper_aligned
    def get_spw(self, ox): raise NotImplementedError
    def get_ripple(self, ox): raise NotImplementedError
    def get_coupling(self, ox): raise NotImplementedError

class ModelSB_OX1R(OrexinModel):
    def __init__(self):
        super().__init__("SB-OX1R",
            "OX1R antagonist (SB-334867): ↓SPW rate/amp/dur, ↓ripple incidence, dur unchanged",
            paper_aligned=True)
        self.spw_incidence_at_low = 0.71; self.spw_amplitude_at_low = 0.59
        self.spw_duration_at_low  = 0.91; self.coupling_prob_at_low  = 0.86
    def get_spw(self, ox):
        t = np.clip((ox - 0.3) / 0.7, 0, 1)
        return SPWMod(rate=self.spw_incidence_at_low + t*(1-self.spw_incidence_at_low),
                      amplitude=self.spw_amplitude_at_low + t*(1-self.spw_amplitude_at_low),
                      duration=self.spw_duration_at_low   + t*(1-self.spw_duration_at_low))
    def get_ripple(self, ox):
        t = np.clip((ox - 0.3) / 0.7, 0, 1)
        return RippleMod(rate=1.0, duration=1.0, power=0.7+0.3*t, frequency=1.0,
                         multi_ripple_tendency=0.8+0.2*t)
    def get_coupling(self, ox):
        return CouplingMod(coupling_prob=ramp_to_one(self.coupling_prob_at_low, ox),
                           amplitude_sensitivity=1.0, lag_precision=1.0)

class ModelEMPA_OX2R(OrexinModel):
    def __init__(self):
        super().__init__("EMPA-OX2R",
            "OX2R antagonist (EMPA): ↓↓SPW rate/amp/dur, ↓↓ripple incidence, ↑ripple dur",
            paper_aligned=True)
        self.spw_incidence_at_low = 0.50; self.spw_amplitude_at_low  = 0.58
        self.spw_duration_at_low  = 0.89; self.ripple_duration_at_low = 1.12
        self.coupling_prob_at_low = 0.83
    def get_spw(self, ox):
        t = np.clip((ox - 0.3) / 0.7, 0, 1)
        return SPWMod(rate=self.spw_incidence_at_low + t*(1-self.spw_incidence_at_low),
                      amplitude=self.spw_amplitude_at_low + t*(1-self.spw_amplitude_at_low),
                      duration=self.spw_duration_at_low   + t*(1-self.spw_duration_at_low))
    def get_ripple(self, ox):
        t = np.clip((ox - 0.3) / 0.7, 0, 1)
        dur = self.ripple_duration_at_low + t*(1-self.ripple_duration_at_low)
        return RippleMod(rate=1.0, duration=dur, power=0.6+0.4*t, frequency=1.0,
                         multi_ripple_tendency=0.6+0.4*t)
    def get_coupling(self, ox):
        t = np.clip((ox - 0.3) / 0.7, 0, 1)
        return CouplingMod(coupling_prob=ramp_to_one(self.coupling_prob_at_low, ox),
                           amplitude_sensitivity=1.0, lag_precision=0.8+0.2*t)

class ModelDORA(OrexinModel):
    def __init__(self):
        super().__init__("DORA",
            "Dual OX1R/OX2R antagonist (TCS-1102): ↓↓↓SPW rate, amp/dur unchanged, ↑↑ripple dur",
            paper_aligned=True)
        self.spw_incidence_at_low   = 0.40; self.ripple_duration_at_low = 1.26
        self.coupling_prob_at_low   = 1.29
    def get_spw(self, ox):
        t = np.clip((ox - 0.3) / 0.7, 0, 1)
        return SPWMod(rate=self.spw_incidence_at_low + t*(1-self.spw_incidence_at_low),
                      amplitude=1.0, duration=1.0)
    def get_ripple(self, ox):
        t = np.clip((ox - 0.3) / 0.7, 0, 1)
        dur = self.ripple_duration_at_low + t*(1-self.ripple_duration_at_low)
        return RippleMod(rate=1.0, duration=dur, power=0.7+0.3*t, frequency=1.0,
                         multi_ripple_tendency=0.7+0.3*t)
    def get_coupling(self, ox):
        t = np.clip((ox - 0.3) / 0.7, 0, 1)
        return CouplingMod(coupling_prob=ramp_to_one(self.coupling_prob_at_low, ox),
                           amplitude_sensitivity=1.0, lag_precision=0.7+0.3*t)

class ModelNull(OrexinModel):
    def __init__(self): super().__init__("Null", "No orexin effect (control, in-vivo band)")
    def get_spw(self, ox): return SPWMod()
    def get_ripple(self, ox): return RippleMod()
    def get_coupling(self, ox): return CouplingMod()

class ModelNullPaperBand(OrexinModel):
    def __init__(self): super().__init__("Null-PaperBand",
                                         "No orexin effect (control, paper band)", paper_aligned=True)
    def get_spw(self, ox): return SPWMod()
    def get_ripple(self, ox): return RippleMod()
    def get_coupling(self, ox): return CouplingMod()

class ModelCouplingFidelity(OrexinModel):
    def __init__(self):
        super().__init__("Coupling-Fidelity",
                         "Inverted-U: optimal coupling at intermediate orexin (EXPLORATORY)")
        self.opt = 1.0; self.width = 0.5
    def get_spw(self, ox): return SPWMod(rate=0.5+0.5*ox, amplitude=0.6+0.4*ox)
    def get_ripple(self, ox):
        g = np.exp(-((ox-self.opt)**2)/(2*self.width**2))
        return RippleMod(rate=0.5+0.4*ox, duration=0.7+0.5*g, power=0.6+0.4*ox,
                         multi_ripple_tendency=0.7+0.5*g)
    def get_coupling(self, ox):
        g = np.exp(-((ox-self.opt)**2)/(2*self.width**2))
        return CouplingMod(coupling_prob=0.5+0.6*g, amplitude_sensitivity=0.8+0.4*g,
                           lag_precision=0.6+0.4*g)

class ModelDualReceptor(OrexinModel):
    def __init__(self):
        super().__init__("Dual-Receptor", "OX1R + OX2R combined effects (EXPLORATORY)")
        self._sb = ModelSB_OX1R(); self._empa = ModelEMPA_OX2R()
    def get_spw(self, ox):
        sb = self._sb.get_spw(ox); em = self._empa.get_spw(ox)
        return SPWMod(rate=0.5*sb.rate+0.5*em.rate,
                      amplitude=0.5*sb.amplitude+0.5*em.amplitude,
                      duration=0.5*sb.duration+0.5*em.duration)
    def get_ripple(self, ox):
        sb = self._sb.get_ripple(ox); em = self._empa.get_ripple(ox)
        return RippleMod(rate=1.0, duration=0.5*sb.duration+0.5*em.duration,
                         power=0.5*sb.power+0.5*em.power,
                         multi_ripple_tendency=0.5*sb.multi_ripple_tendency+0.5*em.multi_ripple_tendency)
    def get_coupling(self, ox):
        sb = self._sb.get_coupling(ox); em = self._empa.get_coupling(ox)
        return CouplingMod(coupling_prob=0.5*sb.coupling_prob+0.5*em.coupling_prob,
                           amplitude_sensitivity=1.0,
                           lag_precision=0.5*sb.lag_precision+0.5*em.lag_precision)

OREXIN_MODELS = {
    'sb_ox1r': ModelSB_OX1R(), 'empa_ox2r': ModelEMPA_OX2R(),
    'dora': ModelDORA(), 'null': ModelNull(), 'null_paper_band': ModelNullPaperBand(),
    'coupling_fidelity': ModelCouplingFidelity(), 'dual_receptor': ModelDualReceptor(),
}
print(f"  ✓ {len(OREXIN_MODELS)} models defined")

# =============================================================================
# GENERATOR CLASS
# =============================================================================
print("\n[5/7] Building generator...")

class SyntheticGenerator:
    """Dense-grid library keyed by ox_key()."""
    COUPLING_LAG_MIN_MS = -20
    COUPLING_LAG_MAX_MS =  50

    def __init__(self, datasets, sequence_length=16, seed=42):
        self.rng = np.random.RandomState(seed)
        self.sequence_length = sequence_length
        self._fit(datasets)

    def _fit(self, datasets):
        spw_durs = safe_concat([d['spw_durations']  for d in datasets])
        spw_amps = safe_concat([d['spw_amplitudes'] for d in datasets])
        rip_durs = safe_concat([d['swr_ripple_durations']  for d in datasets])
        rip_pows = safe_concat([d['swr_ripple_powers']     for d in datasets])
        rip_freqs = safe_concat([d['swr_ripple_frequencies'] for d in datasets])
        coup_lags = safe_concat([d['coupling_lags'] for d in datasets], allow_empty=True)

        # --- Durations & ISIs: gamma (waiting-time processes) ---
        self.spw_dur_params = stats.gamma.fit(spw_durs[spw_durs > 0], floc=0)
        self.spw_isi_params = stats.gamma.fit(all_spw_isis[all_spw_isis > 0], floc=0)
        self.rip_dur_params = stats.gamma.fit(rip_durs[rip_durs > 0], floc=0)

        # --- Magnitudes: per-variable distribution selection ---
        # SPW amplitude: lognormal committed (Mizuseki & Buzsáki 2013;
        # Buzsáki & Mizuseki 2014 — established lognormal statistics;
        # also strongly supported by empirical AIC comparison).
        # Ripple power: AIC winner (let the data decide; no literature
        # precedent specifically for ripple power distribution family).
        print("\n    ── Distribution comparison: Gamma vs Lognormal ──")
        self.spw_amp_comparison = compare_gamma_lognormal(
            spw_amps, label="SPW amplitude")
        self.rip_pow_comparison = compare_gamma_lognormal(
            rip_pows, label="Ripple power")

        # SPW amplitude: COMMIT to lognormal (literature-justified + empirical)
        self.spw_amp_dist = self.spw_amp_comparison['lognorm']

        # Ripple power: use AIC winner (data-driven, no override)
        self.rip_pow_dist = self.rip_pow_comparison['winner']

        # Backward-compatible aliases (gamma params, kept for validation)
        self.spw_amp_params = self.spw_amp_comparison['gamma'].params
        self.rip_pow_params = self.rip_pow_comparison['gamma'].params

        print(f"\n    Active distributions:")
        print(f"      SPW amplitude : lognorm  (committed — literature + ΔAIC={self.spw_amp_comparison['delta_aic']:+.1f})")
        print(f"      Ripple power  : {self.rip_pow_dist.family:<8} (AIC-selected — ΔAIC={self.rip_pow_comparison['delta_aic']:+.1f})")
        print(f"      SPW duration  : gamma  (waiting-time process)")
        print(f"      Ripple duration: gamma  (waiting-time process)")
        print(f"      ISIs          : gamma  (inter-event intervals)")
        print("    ── End distribution comparison ──\n")

        rip_freqs_valid   = rip_freqs[np.isfinite(rip_freqs)]
        vf_invivo = rip_freqs_valid[(rip_freqs_valid >= 150) & (rip_freqs_valid <= 250)]
        vf_paper  = rip_freqs_valid[(rip_freqs_valid >= 100) & (rip_freqs_valid <= 175)]
        self.rip_freq_mean_invivo = np.mean(vf_invivo) if len(vf_invivo) > 0 else BASELINE_FREQ_MEAN
        self.rip_freq_std_invivo  = np.std(vf_invivo)  if len(vf_invivo) > 0 else BASELINE_FREQ_STD
        self.rip_freq_mean_paper  = np.mean(vf_paper)  if len(vf_paper)  > 0 else 140.0
        self.rip_freq_std_paper   = np.std(vf_paper)   if len(vf_paper)  > 0 else 15.0
        self.empirical_freqs_invivo = vf_invivo.copy() if len(vf_invivo) > 0 else np.array([175.0])
        self.empirical_freqs_paper  = vf_paper.copy()  if len(vf_paper)  > 0 else np.array([140.0])

        vc = all_swr_cycles[all_swr_cycles >= MIN_RIPPLE_CYCLES]
        self.rip_cycles_mean = np.mean(vc) if len(vc) > 0 else BASELINE_CYCLES_MEAN
        self.rip_cycles_std  = np.std(vc)  if len(vc) > 0 else BASELINE_CYCLES_STD
        self.rip_isi_params  = stats.gamma.fit(all_ripple_isis[all_ripple_isis > 0], floc=0)

        self.empirical_lags   = coup_lags[np.isfinite(coup_lags)] if len(coup_lags) > 0 else np.array([0.0])
        self.coup_lag_mean    = np.mean(self.empirical_lags)
        self.coup_lag_std     = np.std(self.empirical_lags) if len(self.empirical_lags) > 1 else 10.0
        self.empirical_rps    = EMPIRICAL_RPS.copy()
        self.baseline_spw_rate = BASELINE_SPW_RATE

        # Calibrate coupling with the SELECTED amplitude distribution
        cal_amps = self.spw_amp_dist.rvs(size=50000,
                                          random_state=np.random.RandomState(42))
        self.spw_amp_median     = np.median(cal_amps)
        self._expected_amp_effect = np.mean(
            (cal_amps - self.spw_amp_median) * COUPLING_AMP_SENSITIVITY)
        print(f"    Fitted distributions | sequence_length={self.sequence_length}")

    def _sample_ripples_per_spw(self, rng, multi_ripple_tendency):
        if len(self.empirical_rps) == 0 or np.all(self.empirical_rps == 1):
            return 1
        w = np.exp((self.empirical_rps - 1) * (multi_ripple_tendency - 1) * 0.5)
        w = w / w.sum()
        return int(rng.choice(self.empirical_rps, p=w))

    def _compute_coupling_prob(self, spw_amplitude, coup_mod):
        base_p     = BASELINE_P_SPW_HAS_RIPPLE * coup_mod.coupling_prob
        amp_effect = COUPLING_AMP_SENSITIVITY * (spw_amplitude - self.spw_amp_median) \
                     * coup_mod.amplitude_sensitivity
        return np.clip(base_p + amp_effect - self._expected_amp_effect
                       * coup_mod.amplitude_sensitivity, 1e-6, 1 - 1e-6)

    def _sample_lag(self, rng, lag_precision):
        if len(self.empirical_lags) == 0:
            return 0.0
        base_lag = rng.choice(self.empirical_lags)
        jitter   = rng.normal(0, self.coup_lag_std / max(lag_precision, 1e-6) * 0.3)
        return np.clip(base_lag + jitter, self.COUPLING_LAG_MIN_MS, self.COUPLING_LAG_MAX_MS)

    def _generate_spw_times_poisson(self, rng, duration_sec, spw_mod):
        rate = self.baseline_spw_rate * spw_mod.rate
        if rate <= 0:
            return np.array([])
        n = rng.poisson(rate * duration_sec)
        return np.sort(rng.uniform(0, duration_sec, n)) if n > 0 else np.array([])

    def _sample_frequency(self, rng, freq_mean_base, freq_std_base,
                           freq_min, freq_max, rip_mod, use_empirical):
        if use_empirical:
            emp = self.empirical_freqs_paper if freq_max <= 175 else self.empirical_freqs_invivo
            return np.clip(rng.choice(emp) * rip_mod.frequency, freq_min, freq_max)
        return sample_trunc_norm(rng, freq_mean_base * rip_mod.frequency,
                                  freq_std_base * rip_mod.frequency, freq_min, freq_max)

    def _generate_ripple(self, rng, freq_mean_base, freq_std_base,
                          freq_min, freq_max, rip_mod):
        freq    = self._sample_frequency(rng, freq_mean_base, freq_std_base,
                                          freq_min, freq_max, rip_mod, USE_EMPIRICAL_FREQ_SAMPLING)
        n_cyc   = max(MIN_RIPPLE_CYCLES,
                      rng.normal(self.rip_cycles_mean, self.rip_cycles_std) * rip_mod.duration)
        dur     = np.clip((n_cyc / freq) * 1000.0,
                          compute_min_duration_ms(freq, MIN_RIPPLE_CYCLES), MAX_RIPPLE_DURATION_MS)
        n_cyc   = (dur / 1000.0) * freq
        power   = self.rip_pow_dist.rvs(random_state=rng) * rip_mod.power
        return dur, freq, n_cyc, power

    def generate(self, model, duration_sec, orexin, context, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        spw_mod = model.get_spw(orexin)
        rip_mod = model.get_ripple(orexin)
        coup_mod = model.get_coupling(orexin)

        if model.paper_aligned:
            freq_min, freq_max       = 100, 175
            freq_mean_base, freq_std_base = self.rip_freq_mean_paper, self.rip_freq_std_paper
        else:
            freq_min, freq_max       = 150, 250
            freq_mean_base, freq_std_base = self.rip_freq_mean_invivo, self.rip_freq_std_invivo

        spw_times = self._generate_spw_times_poisson(rng, duration_sec, spw_mod)
        n_spw     = len(spw_times)

        if n_spw == 0:
            spw_durations = spw_amplitudes = np.array([])
            spw_has_ripple = np.array([], dtype=bool)
            ripples_per_spw = np.array([], dtype=int)
        else:
            spw_durations = np.clip(
                stats.gamma.rvs(*self.spw_dur_params, size=n_spw, random_state=rng) * spw_mod.duration,
                30, 200)
            spw_amplitudes  = self.spw_amp_dist.rvs(size=n_spw, random_state=rng) * spw_mod.amplitude
            spw_has_ripple  = np.zeros(n_spw, dtype=bool)
            ripples_per_spw = np.zeros(n_spw, dtype=int)

        crt = []; crd = []; crp = []; crf = []; crc = []; clags = []
        if n_spw > 0:
            for i, (t_spw, amp) in enumerate(zip(spw_times, spw_amplitudes)):
                if rng.random() < self._compute_coupling_prob(amp, coup_mod):
                    spw_has_ripple[i] = True
                    k = self._sample_ripples_per_spw(rng, rip_mod.multi_ripple_tendency)
                    ripples_per_spw[i] = k
                    for rip_idx in range(k):
                        lag_ms = self._sample_lag(rng, coup_mod.lag_precision)
                        if rip_idx > 0:
                            lag_ms = np.clip(lag_ms + rip_idx * rng.uniform(5, 15),
                                             self.COUPLING_LAG_MIN_MS, self.COUPLING_LAG_MAX_MS)
                        clags.append(lag_ms)
                        crt.append(t_spw + lag_ms / 1000)
                        dur, freq, nc, power = self._generate_ripple(
                            rng, freq_mean_base, freq_std_base, freq_min, freq_max, rip_mod)
                        crd.append(dur); crf.append(freq); crc.append(nc); crp.append(power)

        n_coup = len(crt)
        n_spw_with_ripple = int(np.sum(spw_has_ripple))

        crt  = np.array(crt);  crd  = np.array(crd);  crp  = np.array(crp)
        crf  = np.array(crf);  crc  = np.array(crc);  clags = np.array(clags)

        uncoupled_rate = max(0, (BASELINE_RIPPLE_RATE - BASELINE_SWR_RATE)
                               * rip_mod.rate * UNCOUPLED_RATE_FRACTION)
        n_unc = rng.poisson(uncoupled_rate * duration_sec)
        if n_unc > 0:
            utimes = rng.uniform(0, duration_sec, n_unc)
            ud, uf, uc, up = [], [], [], []
            for _ in range(n_unc):
                dur, freq, nc, power = self._generate_ripple(
                    rng, freq_mean_base, freq_std_base, freq_min, freq_max, rip_mod)
                ud.append(dur); uf.append(freq); uc.append(nc); up.append(power)
            ud = np.array(ud); uf = np.array(uf); uc = np.array(uc); up = np.array(up)
        else:
            utimes = ud = uf = uc = up = np.array([])

        all_rt  = np.concatenate([crt, utimes])
        all_rd  = np.concatenate([crd, ud])
        all_rp  = np.concatenate([crp, up])
        all_rf  = np.concatenate([crf, uf])
        all_rc  = np.concatenate([crc, uc])
        n_rip   = len(all_rt)

        if n_rip > 0:
            idx = np.argsort(all_rt)
            all_rt = all_rt[idx]; all_rd = all_rd[idx]
            all_rp = all_rp[idx]; all_rf = all_rf[idx]; all_rc = all_rc[idx]

        total_cyc    = np.sum(crc)
        fidelity     = np.clip(total_cyc / self.sequence_length, 0, 1) if n_coup > 0 else 0
        max_comp     = np.max(np.clip(crc / self.sequence_length, 0, 1)) if n_coup > 0 else 0
        mean_comp    = np.mean(np.clip(crc / self.sequence_length, 0, 1)) if n_coup > 0 else 0
        base_react   = 0.75 if context == 'Context_A' else 0.25
        ctx_react    = base_react * fidelity + 0.5 * (1 - fidelity)
        p_has_rip    = n_spw_with_ripple / n_spw if n_spw > 0 else 0

        return {
            'orexin_level': orexin, 'context': context,
            'context_id': 0 if context == 'Context_A' else 1,
            'duration_sec': duration_sec, 'sequence_length': self.sequence_length,
            'model_name': model.name, 'paper_aligned': model.paper_aligned,
            'n_spw': n_spw, 'spw_rate': n_spw / duration_sec,
            'spw_mean_duration': float(np.mean(spw_durations)) if n_spw > 0 else 0,
            'spw_mean_amplitude': float(np.mean(spw_amplitudes)) if n_spw > 0 else 0,
            'spw_isi_cv': (np.std(np.diff(spw_times)) / np.mean(np.diff(spw_times))
                           if n_spw > 1 else 0),
            'n_ripples': n_rip, 'ripple_rate': n_rip / duration_sec,
            'n_coupled': n_coup, 'n_spw_with_ripple': n_spw_with_ripple,
            'swr_ripple_rate': n_coup / duration_sec,
            'swr_mean_duration': float(np.mean(crd)) if n_coup > 0 else 0,
            'swr_mean_frequency': float(np.mean(crf)) if n_coup > 0 else 0,
            'swr_mean_cycles': float(np.mean(crc)) if n_coup > 0 else 0,
            'swr_total_cycles': float(total_cyc),
            'p_spw_has_ripple': p_has_rip,
            'coupling_lag_mean': float(np.mean(clags)) if len(clags) > 0 else 0,
            'coupling_lag_std':  float(np.std(clags))  if len(clags) > 1 else 0,
            'content_fidelity': fidelity, 'max_replay_completeness': max_comp,
            'mean_replay_completeness': mean_comp, 'context_reactivation': ctx_react,
            'mean_ripples_per_spw': (float(np.mean(ripples_per_spw[spw_has_ripple]))
                                      if n_spw_with_ripple > 0 else 0),
            # Per-SPW arrays (for agent plan builder)
            'spw_times': spw_times, 'spw_durations': spw_durations,
            'spw_amplitudes': spw_amplitudes,
            'spw_has_ripple': spw_has_ripple,
            'ripples_per_spw': ripples_per_spw,
            # Per-ripple arrays
            'swr_cycles': crc, 'coupling_lags': clags,
            'swr_durations': crd, 'swr_frequencies': crf,
        }

    def generate_library(self, model, orexin_levels, n_trials=200,
                          duration_sec=3.0, seed=42):
        """
        Generate a trial library keyed by ox_key().
        Key format: f"{context}_ox{ox_key(level)}"
        """
        library = {}
        for ctx in ['Context_A', 'Context_B']:
            for ox in orexin_levels:
                key = f"{ctx}_ox{ox_key(ox)}"   # canonical formatter
                library[key] = []
                for trial in range(n_trials):
                    s = stable_seed(seed, f"{ctx}_{ox_key(ox)}_{trial}")
                    library[key].append(self.generate(model, duration_sec, ox, ctx, s))
        return library


generator = SyntheticGenerator(datasets, sequence_length=16, seed=42)
print(f"\n  ✓ Generator ready (sequence_length=16)")

# Calibration check
print("\n[Calibration check]")
_cal_amps = generator.spw_amp_dist.rvs(size=10000,
                              random_state=np.random.RandomState(99))
_null_cm  = CouplingMod(coupling_prob=1.0, amplitude_sensitivity=1.0, lag_precision=1.0)
_null_eff = np.mean([generator._compute_coupling_prob(a, _null_cm) for a in _cal_amps])
print(f"  Null effective coupling: {_null_eff:.4f}  (target: {BASELINE_P_SPW_HAS_RIPPLE:.4f})")
if abs(_null_eff - BASELINE_P_SPW_HAS_RIPPLE) < 0.001:
    print(f"  ✓ Calibration anchored correctly")

# =============================================================================
# GENERATE ALL LIBRARIES
# =============================================================================
print("\n[6/7] Generating libraries...")

# Dense grid — matches agent sweep exactly
OREXIN_LEVELS = [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
                 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.8]
N_TRIALS      = 200          # 15 levels × 200 × 2 contexts = 6000 total
DURATION_SEC  = 3.0

print(f"  Grid:    {OREXIN_LEVELS}")
print(f"  Trials per (model × context × level): {N_TRIALS}")
print(f"  Key format: 'Context_A_ox{ox_key(OREXIN_LEVELS[0])}' … 'Context_B_ox{ox_key(OREXIN_LEVELS[-1])}'")

all_libraries = {}
for model_name, model in OREXIN_MODELS.items():
    aligned = "📄" if model.paper_aligned else "🔬"
    print(f"  {aligned} {model_name}...", end=" ", flush=True)
    all_libraries[model_name] = generator.generate_library(
        model, OREXIN_LEVELS, N_TRIALS, DURATION_SEC, seed=42)
    print("✓")

print(f"\n  ✓ All {len(OREXIN_MODELS)} models complete")

# =============================================================================
# SAVE
# =============================================================================
print("\n[7/7] Saving outputs...")

libraries_path = FILES['libraries']
with open(libraries_path, 'wb') as f:
    pickle.dump({
        'libraries':           all_libraries,
        'orexin_levels':       OREXIN_LEVELS,
        'n_trials':            N_TRIALS,
        'duration_sec':        DURATION_SEC,
        'sequence_length':     generator.sequence_length,
        'key_format':          'f"{context}_ox{ox_key(level)}"',
        'ox_key_source':       'orexin_keys.ox_key -> f"{float(x):.4f}".rstrip("0").rstrip(".")',
        'coupling_window_ms':  (generator.COUPLING_LAG_MIN_MS, generator.COUPLING_LAG_MAX_MS),
        'min_ripple_cycles':   MIN_RIPPLE_CYCLES,
        'max_ripple_duration_ms': MAX_RIPPLE_DURATION_MS,
        'uncoupled_rate_fraction': UNCOUPLED_RATE_FRACTION,
        'use_empirical_freq_sampling': USE_EMPIRICAL_FREQ_SAMPLING,
        'model_names':         list(OREXIN_MODELS.keys()),
        'paper_aligned_models': [n for n, m in OREXIN_MODELS.items() if m.paper_aligned],
        'baseline_spw_rate':   BASELINE_SPW_RATE,
        'baseline_swr_rate':   BASELINE_SWR_RATE,
        'baseline_p_spw_has_ripple': BASELINE_P_SPW_HAS_RIPPLE,
        'baseline_ripples_per_spw':  BASELINE_RIPPLES_PER_SPW,
        'baseline_cycles_mean': BASELINE_CYCLES_MEAN,
        'baseline_freq_mean':  BASELINE_FREQ_MEAN,
        'coupling_amp_sensitivity': COUPLING_AMP_SENSITIVITY,
        'isi_mean': ISI_MEAN, 'isi_median': ISI_MEDIAN,
        'drug_coupling_calibration': {
            'sb_ox1r':   ModelSB_OX1R().coupling_prob_at_low,
            'empa_ox2r': ModelEMPA_OX2R().coupling_prob_at_low,
            'dora':      ModelDORA().coupling_prob_at_low,
        },
        'frequency_bands': {'invivo': (150, 250), 'paper': (100, 175)},
        'distribution_comparison': {
            'spw_amplitude': {
                'active': 'lognorm',
                'selection': 'literature-committed (Mizuseki & Buzsáki 2013)',
                'gamma_aic':   generator.spw_amp_comparison['gamma'].aic,
                'lognorm_aic': generator.spw_amp_comparison['lognorm'].aic,
                'delta_aic':   generator.spw_amp_comparison['delta_aic'],
                'gamma_ks':    generator.spw_amp_comparison['gamma'].ks_stat,
                'lognorm_ks':  generator.spw_amp_comparison['lognorm'].ks_stat,
                'gamma_params':   generator.spw_amp_comparison['gamma'].params,
                'lognorm_params': generator.spw_amp_comparison['lognorm'].params,
            },
            'ripple_power': {
                'active': generator.rip_pow_dist.family,
                'selection': 'AIC-selected (data-driven)',
                'gamma_aic':   generator.rip_pow_comparison['gamma'].aic,
                'lognorm_aic': generator.rip_pow_comparison['lognorm'].aic,
                'delta_aic':   generator.rip_pow_comparison['delta_aic'],
                'gamma_ks':    generator.rip_pow_comparison['gamma'].ks_stat,
                'lognorm_ks':  generator.rip_pow_comparison['lognorm'].ks_stat,
                'gamma_params':   generator.rip_pow_comparison['gamma'].params,
                'lognorm_params': generator.rip_pow_comparison['lognorm'].params,
            },
            'note': 'SPW amplitude: lognormal committed (literature + empirical). '
                    f'Ripple power: {generator.rip_pow_dist.family} (AIC-selected). '
                    'Durations and ISIs: gamma (waiting-time process).',
        },
        'created': datetime.now().isoformat(),
        'detector_version': detector_version,
        'source_sessions':  [d['name'] for d in datasets],
        'channel_rule':     sorted({s.get('channel_info', {}).get('rule', 'untagged')
                                    for s in datasets_raw}),
        **stamp(),
        'methodology': [
            'Per-variable distribution family selection',
            f'Grid: {OREXIN_LEVELS}',
            'Key format: context_ox{ox_key(level)} — canonical float formatter',
            'Calibrated coupling for drug phenotypes',
            'Paper-aligned: Kostansek et al. 2023',
            f'Min {MIN_RIPPLE_CYCLES} cycles GUARANTEED',
            'Poisson process for SPW timing',
            'SPW amplitude: lognormal (Mizuseki & Buzsáki 2013; Buzsáki & Mizuseki 2014)',
            f'Ripple power: {generator.rip_pow_dist.family} (AIC-selected, data-driven)',
            'Durations/ISIs: gamma (waiting-time process)',
            'Both gamma and lognormal fits stored for reporting',
        ],
    }, f)
print(f"  ✓ {libraries_path}")

config_path = FILES['generator_config']
with open(config_path, 'wb') as f:
    pickle.dump({
        'baseline_spw_rate':        BASELINE_SPW_RATE,
        'baseline_swr_rate':        BASELINE_SWR_RATE,
        'baseline_p_spw_has_ripple': BASELINE_P_SPW_HAS_RIPPLE,
        'baseline_ripples_per_spw': BASELINE_RIPPLES_PER_SPW,
        'baseline_cycles_mean':     BASELINE_CYCLES_MEAN,
        'baseline_freq_mean':       BASELINE_FREQ_MEAN,
        'baseline_freq_std':        BASELINE_FREQ_STD,
        'coupling_amp_sensitivity': COUPLING_AMP_SENSITIVITY,
        'isi_mean': ISI_MEAN, 'isi_median': ISI_MEDIAN, 'isi_std': ISI_STD,
        'drug_coupling_calibration': {
            'sb_ox1r':   ModelSB_OX1R().coupling_prob_at_low,
            'empa_ox2r': ModelEMPA_OX2R().coupling_prob_at_low,
            'dora':      ModelDORA().coupling_prob_at_low,
        },
        'sequence_length':       generator.sequence_length,
        'orexin_levels':         OREXIN_LEVELS,
        'key_format':            'f"{context}_ox{ox_key(level)}"',
        'ox_key_source':         'orexin_keys.ox_key -> f"{float(x):.4f}".rstrip("0").rstrip(".")',
        'coupling_window_ms':    (generator.COUPLING_LAG_MIN_MS, generator.COUPLING_LAG_MAX_MS),
        'min_ripple_cycles':     MIN_RIPPLE_CYCLES,
        'max_ripple_duration_ms': MAX_RIPPLE_DURATION_MS,
        'spw_amp_distribution':  generator.spw_amp_dist.family,
        'rip_pow_distribution':  generator.rip_pow_dist.family,
        'spw_amp_dist_params':   generator.spw_amp_dist.params,
        'rip_pow_dist_params':   generator.rip_pow_dist.params,
        'created':               datetime.now().isoformat(),
        'detector_version':      detector_version,
        'source_sessions':       [d['name'] for d in datasets],
        **stamp(),
    }, f)
print(f"  ✓ {config_path}")

# =============================================================================
# VALIDATION
# =============================================================================
print("\n" + "="*70)
print("VALIDATION: Null Model @ ox=1.0")
print("="*70)

_null_key   = f"Context_A_ox{ox_key(1.0)}"
null_trials = all_libraries['null'][_null_key]

synth_spw_rate = np.mean([t['spw_rate'] * 60 for t in null_trials])
synth_swr_rate = np.mean([t['swr_ripple_rate'] * 60 for t in null_trials])
tot_sp = sum(t['n_spw']           for t in null_trials)
tot_cp = sum(t['n_spw_with_ripple'] for t in null_trials)
synth_p_coup = tot_cp / tot_sp if tot_sp > 0 else 0

all_synth_cyc = np.concatenate([t['swr_cycles'] for t in null_trials if len(t['swr_cycles']) > 0])
min_cyc = np.min(all_synth_cyc) if len(all_synth_cyc) > 0 else 0
viol    = int(np.sum(all_synth_cyc < MIN_RIPPLE_CYCLES)) if len(all_synth_cyc) > 0 else 0

print(f"\n{'Metric':<25} {'Empirical':<12} {'Synthetic':<12} {'Match'}")
print("-"*55)
print(f"{'SPW rate (/min)':<25} {BASELINE_SPW_RATE*60:<12.2f} {synth_spw_rate:<12.2f} "
      f"{'✓' if abs(synth_spw_rate - BASELINE_SPW_RATE*60) < 3 else '✗'}")
print(f"{'SWR rate (/min)':<25} {BASELINE_SWR_RATE*60:<12.2f} {synth_swr_rate:<12.2f} "
      f"{'✓' if abs(synth_swr_rate - BASELINE_SWR_RATE*60) < 2 else '✗'}")
print(f"{'P(couple)':<25} {BASELINE_P_SPW_HAS_RIPPLE:<12.1%} {synth_p_coup:<12.1%} "
      f"{'✓' if abs(synth_p_coup - BASELINE_P_SPW_HAS_RIPPLE) < 0.05 else '✗'}")
print(f"\n  ≥3 cycles: Min={min_cyc:.2f}, Violations={viol}")

print("\n" + "="*70)
print("VALIDATION: Drug Phenotypes @ ox=0.3")
print("="*70)
rng_val = np.random.RandomState(0)
_cal2   = generator.spw_amp_dist.rvs(size=10000, random_state=rng_val)
_null_cm = CouplingMod(coupling_prob=1.0, amplitude_sensitivity=1.0, lag_precision=1.0)
_base_p  = np.mean([generator._compute_coupling_prob(a, _null_cm) for a in _cal2])
print(f"\nBaseline effective coupling: {_base_p:.1%}")

paper_targets = {'sb_ox1r': 0.55, 'empa_ox2r': 0.38, 'dora': 0.51}
print(f"\n{'Model':<12} {'SPW rate':<10} {'Amp':<10} {'Coup_p':<10} {'Predicted':<10} {'Paper':<10} {'Match'}")
print("-"*75)
for mn in ['sb_ox1r', 'empa_ox2r', 'dora']:
    model     = OREXIN_MODELS[mn]
    spw_mod   = model.get_spw(0.3)
    coup_mod  = model.get_coupling(0.3)
    mod_amps  = _cal2 * spw_mod.amplitude
    eff_p     = np.mean([generator._compute_coupling_prob(a, coup_mod) for a in mod_amps])
    predicted = spw_mod.rate * (eff_p / _base_p) if _base_p > 0 else spw_mod.rate
    paper     = paper_targets[mn]
    print(f"{mn:<12} {spw_mod.rate:<10.2f} {spw_mod.amplitude:<10.2f} "
          f"{coup_mod.coupling_prob:<10.2f} {predicted:<10.2f} {paper:<10.2f} "
          f"{'✓' if abs(predicted - paper) < 0.08 else '✗'}")

# Key format smoke test
print("\n" + "="*70)
print("KEY FORMAT SMOKE TEST")
print("="*70)
print("\n  Verifying all expected keys exist in empa_ox2r library:")
for ox in OREXIN_LEVELS:
    for ctx in ['Context_A', 'Context_B']:
        k = f"{ctx}_ox{ox_key(ox)}"
        present = k in all_libraries['empa_ox2r']
        print(f"  {'✓' if present else '✗'} {k!r}")

print("\n" + "="*70)
print("GENERATION COMPLETE")
print("="*70)
print(f"\n  Grid:    {len(OREXIN_LEVELS)} levels, {N_TRIALS} trials each")
print(f"  Keys:    f\"{{context}}_ox{{ox_key(level)}}\"  (canonical formatter)")
print(f"  Library: synthetic_libraries.pkl")
print(f"  Config:  generator_config.pkl")

print(f"\n  ── Distribution Summary ──")
_amp_d = generator.spw_amp_dist
_pow_d = generator.rip_pow_dist
print(f"  SPW amplitude : {_amp_d.family:<10}"
      f"  params={_amp_d.params[0]:.3f}, scale={_amp_d.params[2]:.1f}"
      f"  ΔAIC={generator.spw_amp_comparison['delta_aic']:+.1f}"
      f"  [committed — literature + empirical]")
print(f"  Ripple power  : {_pow_d.family:<10}"
      f"  params={_pow_d.params[0]:.3f}, scale={_pow_d.params[2]:.1f}"
      f"  ΔAIC={generator.rip_pow_comparison['delta_aic']:+.1f}"
      f"  [AIC-selected]")
print(f"  SPW duration  : gamma     (waiting-time process)")
print(f"  Ripple dur    : gamma     (waiting-time process)")
print(f"  ISIs          : gamma     (inter-event intervals)")
print(f"  ──────────────────────────────────")
print(f"  Ref: Mizuseki & Buzsáki (2013); Buzsáki & Mizuseki (2014)")

print(f"    lib_path = DIRS['models'] / 'synthetic_libraries.pkl'")
print(f"    cfg_path = DIRS['models'] / 'generator_config.pkl'")
print("="*70)

# =============================================================================
# VALIDATION LIBRARY FOR FIGURE 2  +  STALE CACHE
# =============================================================================
# A dedicated null-model library, separate from the agent library (like the
# n=2000 calibration libraries Table 1 uses). The agent keeps its 200-trial grid.
VAL_TRIALS, VAL_SEED = 5000, 777     # seed independent of the agent library (42)
VAL_PKL = FILES['validation_null']

print(f"\n[Validation library] null model, ox = 1.0, {VAL_TRIALS} trials, seed {VAL_SEED} ...")
_val = generator.generate_library(OREXIN_MODELS['null'], [1.0],
                                  n_trials=VAL_TRIALS, duration_sec=DURATION_SEC,
                                  seed=VAL_SEED)
with open(VAL_PKL, 'wb') as f:
    pickle.dump({
        'libraries':       {'null': _val},
        'n_trials':        VAL_TRIALS,
        'seed':            VAL_SEED,
        'duration_sec':    DURATION_SEC,
        'source_sessions': [d['name'] for d in datasets],
        'min_ripple_cycles': MIN_RIPPLE_CYCLES,
        'use_empirical_freq_sampling': USE_EMPIRICAL_FREQ_SAMPLING,
        'created':         datetime.now().isoformat(),
        **stamp(),
    }, f)
_n_ev = sum(len(t['swr_durations']) for t in _val[f"Context_A_ox{ox_key(1.0)}"])
print(f"  \u2713 {VAL_PKL.name}: {_n_ev} synthetic SWR events in Context_A")

# Table 1 caches libraries drawn from the generator. They came from the
# previous generator state, so remove them; Table 1 regenerates on its next run.
_cal_cache = FILES['calibration_libraries']
if _cal_cache.exists():
    _cal_cache.unlink()
    print(f"  \u2713 removed stale {_cal_cache.name} (Table 1 will rebuild it)")

print("\n  Next: re-run the agent model, Table 1, and every figure downstream.")
print("="*70)
