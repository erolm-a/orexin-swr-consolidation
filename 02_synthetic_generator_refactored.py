"""
Stage 2 — synthetic SPW/ripple library generation (generator v5.8.3).

Reads:  data_processed/swr_detection.pkl
Writes: models/synthetic_libraries.pkl
        models/generator_config.pkl

Includes the validation cell that compares synthetic against empirical
statistics and tags each metric [CONSTRAINED] / [FIT-PRESERVED] /
[REPRODUCED].

Paths and artefact versions are resolved via orexin_paths.
"""

from common.paths import setup

DIRS, FILES, PROJECT_ROOT, VERSIONS, load_artifact = setup()

# =============================================================================
# CELL 7: SYNTHETIC SPW & RIPPLE GENERATOR (v5.8.3 - Dense Grid + Robust Keys)
# =============================================================================
"""
SYNTHETIC SPW-RIPPLE GENERATOR - v5.8.3

Changes from v5.8.2 → v5.8.3:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISTRIBUTION FAMILY UPDATE:
  SPW amplitude: lognormal (committed).
    Justified by both literature (Mizuseki & Buzsáki 2013; Buzsáki &
    Mizuseki 2014 — lognormal statistics for hippocampal magnitudes) and
    empirical AIC comparison on Girardeau et al. 2017 data.
  Ripple power: AIC-selected (gamma or lognormal, data-driven).
    No literature precedent specifically mandates lognormal for ripple
    power.  The empirical comparison is run and the better-fitting family
    is used automatically.  Both fits are stored for reporting.
  Durations and ISIs: gamma (unchanged — waiting-time processes).

  FittedDist wrapper class provides a uniform sampling interface.
  Gamma fits for all variables are preserved in pickle metadata for
  robustness reporting.

Inherited from v5.8.2:
  - Dense OX2R grid (15 levels matching agent sweep)
  - Canonical key formatter ox_key()
  - SEQUENCE_LENGTH = 16

Functional behaviour (models, coupling, calibration) unchanged.
"""

# =============================================================================
# SETUP
# =============================================================================
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SYNTHETIC SPW-RIPPLE GENERATOR (v5.8.3 - Dense Grid + Robust Keys)")
print("="*70)

# Paths come from orexin_paths (see header). DIRS / FILES are already bound.

# Shared utilities now live in common.* (canonical definitions).
from common.plotting import save_and_show
from common.utils import (
    _cat, _mean_or_nan, _safe_np, compute_isis_within_intervals,
    compute_min_duration_ms, coupled_event_arrays_from_trial, ox_key,
    pooled_coupled_event_arrays, pooled_p_couple, pooled_rate_from_trials,
    ramp_to_one, safe_concat, sample_trunc_norm, stable_seed,
)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
MIN_RIPPLE_CYCLES          = 3
UNCOUPLED_RATE_FRACTION    = 0.5
MAX_RIPPLE_DURATION_MS     = 250.0
USE_EMPIRICAL_FREQ_SAMPLING = True

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
# LOAD DATA FROM v5.4.5 DETECTOR
# =============================================================================
print("\n[1/7] Loading SWR detection v5.4.5 data...")

data_file = FILES['detector']
if not data_file.exists():
    raise FileNotFoundError(f"Data file not found: {data_file}\nRun Cell 6 (v5.4.5) first.")

with open(data_file, 'rb') as f:
    data = pickle.load(f)

datasets_raw = data['datasets']
version      = data.get('version', '5.4.5')
config       = data.get('config', {})
print(f"  ✓ Loaded {len(datasets_raw)} sessions (version {version})")

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
# MODULATION CLASSES  (unchanged)
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
# OREXIN MODELS  (identical to v5.8.1 — calibration unchanged)
# =============================================================================
print("\n[4/7] Defining orexin models (unchanged from v5.8.1)...")

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
print(f"  ✓ {len(OREXIN_MODELS)} models defined (calibration identical to v5.8.1)")

# =============================================================================
# GENERATOR CLASS  (generate() uses ox_key for internal keys — no change to API)
# =============================================================================
print("\n[5/7] Building generator...")

class SyntheticGenerator:
    """v5.8.3 — dense-grid, ox_key-based library keys."""
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
        'ox_key_source':       'f"{x:.4f}".rstrip("0").rstrip(".")',
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
        'detector_version': version,
        'generator_version': VERSIONS['generator'],
        'methodology': [
            'v5.8.3: Distribution family update',
            f'Grid: {OREXIN_LEVELS}',
            'Key format: context_ox{ox_key(level)} — canonical float formatter',
            'v5.8.1: Calibrated coupling for drug phenotypes',
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
        'ox_key_source':         'f"{x:.4f}".rstrip("0").rstrip(".")',
        'coupling_window_ms':    (generator.COUPLING_LAG_MIN_MS, generator.COUPLING_LAG_MAX_MS),
        'min_ripple_cycles':     MIN_RIPPLE_CYCLES,
        'max_ripple_duration_ms': MAX_RIPPLE_DURATION_MS,
        'spw_amp_distribution':  generator.spw_amp_dist.family,
        'rip_pow_distribution':  generator.rip_pow_dist.family,
        'spw_amp_dist_params':   generator.spw_amp_dist.params,
        'rip_pow_dist_params':   generator.rip_pow_dist.params,
        'created':               datetime.now().isoformat(),
        'detector_version':      version,
        'generator_version':     VERSIONS['generator'],
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
print("GENERATION COMPLETE (v5.8.3 - Dense Grid + Robust Keys)")
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

print(f"\n  NOTE: Agent model must update library path to v5.8.3")
print(f"    lib_path = DIRS['models'] / 'synthetic_libraries.pkl'")
print(f"    cfg_path = DIRS['models'] / 'generator_config.pkl'")
print("="*70)

# =============================================================================
# CELL 8: VALIDATION PLOTS (v5.8.3 - Per-Variable Distribution Selection)
#
# Reads from Cell 7 saved outputs:
#   - synthetic_libraries.pkl
#   - generator_config.pkl
#   - swr_detection.pkl  (empirical ground truth)
#
# NEW in this version:
#   - Section D: SPW amplitude & ripple power distribution comparison
#     (gamma vs lognormal fits on empirical data, confirming lognormal
#      commitment per Mizuseki & Buzsáki 2013; Buzsáki & Mizuseki 2014)
#
# Three-tier tag scheme:
#   [CONSTRAINED]     = scalar directly anchored to empirical value during
#                       fitting (e.g. baseline SPW rate, baseline coupling).
#   [FIT-PRESERVED]   = quantity the generator was fit to reproduce via
#                       distribution fitting or as a product of constrained
#                       inputs.
#   [REPRODUCED]      = metric NOT directly calibrated or fitted; emerges
#                       from the model. Agreement is a genuine validation.
# =============================================================================

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wasserstein_distance, ks_2samp, gamma as gamma_dist
from scipy.stats import lognorm as lognorm_dist
from scipy import stats

# ── Force inline rendering when running inside Jupyter ──
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
    from IPython.display import display
except Exception:
    def display(obj): pass   # no-op outside notebooks

print("=" * 70)
print("CELL 8: VALIDATION (v5.8.3 - Per-Variable Distribution Selection)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
# Paths come from orexin_paths (see header). DIRS / FILES are already bound.
FIG_DIR = DIRS["models"] / "figures_v5.8.3_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_PKL = FILES["detector"]
LIB_PKL      = FILES["libraries"]
CFG_PKL      = FILES["generator_config"]

CONTEXT  = "Context_A"
OX_NULL  = 1.0
OX_DRUG  = 0.3
N_BOOT   = 2000

plt.rcParams["figure.dpi"]        = 120
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# Tag colour palette
TAG_COLORS = {
    "[CONSTRAINED]":   "#fff3cd",
    "[FIT-PRESERVED]": "#cce5ff",
    "[REPRODUCED]":    "#d4edda",
}
TAG_DESCRIPTIONS = {
    "[CONSTRAINED]":   "Scalar directly anchored to empirical value during fitting",
    "[FIT-PRESERVED]": "Fit to empirical distribution; agreement expected by construction",
    "[REPRODUCED]":    "Not directly calibrated; genuine validation outcome",
}

# Section A metric configuration
metric_cfg = [
    ("spw_rate_per_min", "SPW rate (/min)", "[CONSTRAINED]"),
    ("swr_rate_per_min", "SWR rate (/min)", "[FIT-PRESERVED]"),
    ("p_couple",         "P(SPW→SWR)",      "[CONSTRAINED]"),
]

# Section B event-level metric configuration
event_metrics = [
    ("SWR frequency (Hz)", "swr_freqs",     "[FIT-PRESERVED]", (100, 260), 30),
    ("SWR duration (ms)",  "swr_durs",      "[REPRODUCED]",    (15,  150), 30),
    ("SWR cycles",         "swr_cycles",    "[FIT-PRESERVED]", (2,   18),  28),
    ("Coupling lag (ms)",  "coupling_lags", "[FIT-PRESERVED]", (-25,  60), 35),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers (generic ones live in common.utils: _safe_np, _mean_or_nan,
# coupled_event_arrays_from_trial, pooled_coupled_event_arrays,
# pooled_rate_from_trials, pooled_p_couple, _cat)
# ─────────────────────────────────────────────────────────────────────────────
# Remaining generator-specific pooling helpers:

def pooled_spw_arrays(trials):
    durs = (np.concatenate([t["spw_durations"] for t in trials
                             if len(t.get("spw_durations", []))])
            if trials else np.array([]))
    amps = (np.concatenate([t["spw_amplitudes"] for t in trials
                             if len(t.get("spw_amplitudes", []))])
            if trials else np.array([]))
    return durs, amps

def pooled_synthetic_isis(trials):
    isis = []
    for t in trials:
        st = _safe_np(t.get("spw_times", [])).astype(float)
        if len(st) > 1:
            isi = np.diff(np.sort(st))
            isi = isi[isi > 0]
            if len(isi): isis.append(isi)
    return np.concatenate(isis) if isis else np.array([])

def bootstrap_synthetic_sessions(null_trials, target_minutes, duration_sec=3.0,
                                  n_boot=2000, rng_seed=0):
    rng         = np.random.default_rng(rng_seed)
    n_needed    = max(1, int(np.ceil(target_minutes * 60.0 / duration_sec)))
    n_available = len(null_trials)
    rows = []
    for _ in range(n_boot):
        idx     = rng.integers(0, n_available, size=n_needed)
        sampled = [null_trials[i] for i in idx]
        total_spw = sum(int(t.get("n_spw",             0)) for t in sampled)
        total_swr = sum(int(t.get("n_coupled",          0)) for t in sampled)
        total_c   = sum(int(t.get("n_spw_with_ripple",  0)) for t in sampled)
        total_sec = n_needed * duration_sec
        p_c = total_c / total_spw if total_spw > 0 else np.nan
        d_arr, f_arr, _ = pooled_coupled_event_arrays(sampled)
        lags_arr = np.concatenate(
            [_safe_np(t.get("coupling_lags", [])) for t in sampled]
        ).astype(float)
        lags_arr = lags_arr[np.isfinite(lags_arr)]
        rows.append({
            "spw_rate_per_min": total_spw / total_sec * 60.0,
            "swr_rate_per_min": total_swr / total_sec * 60.0,
            "p_couple":         p_c,
            "mean_dur_ms":      _mean_or_nan(d_arr),
            "mean_freq_hz":     _mean_or_nan(f_arr),
            "mean_lag_ms":      _mean_or_nan(lags_arr),
        })
    return pd.DataFrame(rows)

def distributional_metrics(emp_arr, syn_arr, label):
    ec = emp_arr[np.isfinite(emp_arr)]
    sc = syn_arr[np.isfinite(syn_arr)]
    if len(ec) == 0 or len(sc) == 0:
        print(f"  {label:<28}  [insufficient data]")
        return {}
    ks_stat, ks_p = ks_2samp(ec, sc)
    emd           = wasserstein_distance(ec, sc)
    mean_diff     = np.mean(sc) - np.mean(ec)
    rel_err       = mean_diff / np.mean(ec) * 100 if np.mean(ec) != 0 else np.nan
    print(f"  {label:<28}  KS={ks_stat:.3f} (p={ks_p:.3f})  "
          f"EMD={emd:.2f}  Δmean={mean_diff:+.2f} ({rel_err:+.1f}%)")
    return {
        "label": label, "ks_stat": ks_stat, "ks_p": ks_p, "emd": emd,
        "mean_emp": np.mean(ec), "mean_syn": np.mean(sc),
        "median_emp": np.median(ec), "median_syn": np.median(sc),
        "rel_mean_err_pct": rel_err,
    }

def fit_and_compare(data, label):
    """Fit gamma and lognormal, return comparison dict for plotting."""
    x = data[data > 0].copy()
    n = len(x)
    k_p = 2
    g_params  = gamma_dist.fit(x, floc=0)
    g_ll      = np.sum(gamma_dist.logpdf(x, *g_params))
    g_aic     = 2*k_p - 2*g_ll
    g_ks, g_p = ks_2samp(x, gamma_dist.rvs(*g_params, size=min(n, 50000),
                          random_state=np.random.RandomState(42)))
    ln_params  = lognorm_dist.fit(x, floc=0)
    ln_ll      = np.sum(lognorm_dist.logpdf(x, *ln_params))
    ln_aic     = 2*k_p - 2*ln_ll
    ln_ks, ln_p = ks_2samp(x, lognorm_dist.rvs(*ln_params, size=min(n, 50000),
                            random_state=np.random.RandomState(42)))
    delta_aic = g_aic - ln_aic
    winner = "lognorm" if delta_aic > 0 else "gamma"
    print(f"  {label}:  n={n}")
    print(f"    Gamma    AIC={g_aic:.1f}  KS={g_ks:.4f}")
    print(f"    Lognorm  AIC={ln_aic:.1f}  KS={ln_ks:.4f}")
    print(f"    ΔAIC={delta_aic:+.1f}  →  AIC prefers: {winner}")
    return {
        "label": label, "data": x, "n": n,
        "g_params": g_params, "g_aic": g_aic, "g_ks": g_ks,
        "ln_params": ln_params, "ln_aic": ln_aic, "ln_ks": ln_ks,
        "delta_aic": delta_aic, "winner": winner,
    }


# ─────────────────────────────────────────────────────────────────────────────
# [1] Load empirical detector output
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading empirical detector output...")
if not DETECTOR_PKL.exists():
    raise FileNotFoundError(f"Missing: {DETECTOR_PKL}")
with open(DETECTOR_PKL, "rb") as f:
    det = pickle.load(f)

_spw_isis_all, _swr_durs_all  = [], []
_swr_freqs_all, _swr_cycs_all = [], []
_coup_lags_all                = []
_spw_amps_all, _swr_pows_all  = [], []   # NEW: for Section D
_sess_rows                    = []

for sess in det["datasets"]:
    ts    = float(sess["nrem_intervals"]["total_sec"])
    spw   = sess["spw_events"]
    swr   = sess["ripple_tiers"]["ripples_swr"]
    coup  = sess["coupling_swr"]

    n_spw = len(spw); n_swr = len(swr)
    n_c   = int(coup.get("n_coupled_spw", 0))

    st = np.array([e["peak_sec"] for e in spw], dtype=float)
    if len(st) > 1:
        isi = np.diff(np.sort(st)); isi = isi[isi > 0]
        if len(isi): _spw_isis_all.append(isi)

    # SPW amplitudes (for Section D)
    spw_amps = np.array([e["amplitude"] for e in spw], dtype=float)
    if len(spw_amps): _spw_amps_all.append(spw_amps)

    swr_durs  = np.array([e["duration_ms"]           for e in swr], dtype=float)
    swr_freqs = np.array([e.get("peak_freq", np.nan) for e in swr], dtype=float)
    swr_pows  = np.array([e.get("power", np.nan)     for e in swr], dtype=float)

    cyc = []
    for e in swr:
        if e.get("n_cycles") is not None:
            cyc.append(float(e["n_cycles"]))
        else:
            pf = e.get("peak_freq", np.nan)
            cyc.append((e["duration_ms"] / 1000.0) * pf if np.isfinite(pf) else np.nan)
    swr_cycs = np.array(cyc, dtype=float)
    lags     = np.array(coup.get("coupling_lags_ms", []), dtype=float)

    if len(swr_durs):   _swr_durs_all.append(swr_durs)
    fv = swr_freqs[np.isfinite(swr_freqs)]
    if len(fv):         _swr_freqs_all.append(fv)
    cv = swr_cycs[np.isfinite(swr_cycs)]
    if len(cv):         _swr_cycs_all.append(cv)
    lv = lags[np.isfinite(lags)]
    if len(lv):         _coup_lags_all.append(lv)
    pv = swr_pows[np.isfinite(swr_pows)]
    if len(pv):         _swr_pows_all.append(pv)

    _sess_rows.append({
        "session":          sess.get("name", f"sess_{len(_sess_rows)}"),
        "duration_min":     ts / 60.0,
        "spw_rate_per_min": n_spw / ts * 60.0 if ts > 0 else np.nan,
        "swr_rate_per_min": n_swr / ts * 60.0 if ts > 0 else np.nan,
        "p_couple":         n_c / n_spw        if n_spw > 0 else np.nan,
        "n_spw": n_spw, "n_swr": n_swr,
    })

# (_cat is imported from common.utils)

sess_df = pd.DataFrame(_sess_rows)
emp = {
    "spw_rate_per_min": sess_df["spw_rate_per_min"].mean(),
    "swr_rate_per_min": sess_df["swr_rate_per_min"].mean(),
    "p_couple":         sess_df["p_couple"].mean(),
    "spw_isis":         _cat(_spw_isis_all),
    "swr_durs":         _cat(_swr_durs_all),
    "swr_freqs":        _cat(_swr_freqs_all),
    "swr_cycles":       _cat(_swr_cycs_all),
    "coupling_lags":    _cat(_coup_lags_all),
    "spw_amps":         _cat(_spw_amps_all),
    "swr_pows":         _cat(_swr_pows_all),
}

print(f"\n  Empirical: {len(sess_df)} sessions")
print(f"  Pooled: SPW {emp['spw_rate_per_min']:.2f}/min  "
      f"SWR {emp['swr_rate_per_min']:.2f}/min  "
      f"P(couple) {emp['p_couple']:.4f}")
print(f"  SPW amplitudes: n={len(emp['spw_amps'])}")
print(f"  SWR powers: n={len(emp['swr_pows'])}")


# ─────────────────────────────────────────────────────────────────────────────
# [2] Load synthetic libraries
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Loading Cell 7 synthetic libraries (v5.8.3)...")
for path in [LIB_PKL, CFG_PKL]:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}\nRun Cell 7 first.")

with open(LIB_PKL, "rb") as f:
    lib = pickle.load(f)
with open(CFG_PKL, "rb") as f:
    cfg = pickle.load(f)

libraries    = lib["libraries"]
duration_sec = float(lib.get("duration_sec", 3.0))

print(f"  Libraries loaded: {list(libraries.keys())}")
print(f"  Duration per trial: {duration_sec}s")

# Distribution info from generator
dist_comp = lib.get("distribution_comparison", {})
if dist_comp:
    amp_info = dist_comp.get("spw_amplitude", {})
    pow_info = dist_comp.get("ripple_power", {})
    if amp_info:
        print(f"  SPW amplitude: {amp_info.get('active', '?')}"
              f"  ({amp_info.get('selection', '')})"
              f"  ΔAIC={amp_info.get('delta_aic', 0):+.1f}")
    if pow_info:
        print(f"  Ripple power:  {pow_info.get('active', '?')}"
              f"  ({pow_info.get('selection', '')})"
              f"  ΔAIC={pow_info.get('delta_aic', 0):+.1f}")

null_trials = libraries["null"][f"{CONTEXT}_ox{ox_key(OX_NULL)}"]
print(f"\n  Null trials: {len(null_trials)}")

null_swr_d, null_swr_f, null_swr_c = pooled_coupled_event_arrays(null_trials)
null_lags = np.concatenate(
    [_safe_np(t.get("coupling_lags", [])) for t in null_trials]
).astype(float)
null_lags  = null_lags[np.isfinite(null_lags)]
null_isis  = pooled_synthetic_isis(null_trials)

# Synthetic amplitude and power arrays
null_spw_d, null_spw_a = pooled_spw_arrays(null_trials)

syn_event_arrays = {
    "swr_freqs":     null_swr_f,
    "swr_durs":      null_swr_d,
    "swr_cycles":    null_swr_c,
    "coupling_lags": null_lags,
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — Session-level agreement
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION A: Session-level agreement — empirical vs synthetic pseudo-sessions")
print("=" * 70)

boot_dfs = {}
for _, row in sess_df.iterrows():
    name = row["session"]
    print(f"  {name}  ({row['duration_min']:.1f} min) …", end=" ", flush=True)
    boot_dfs[name] = bootstrap_synthetic_sessions(
        null_trials, row["duration_min"],
        duration_sec=duration_sec, n_boot=N_BOOT, rng_seed=42,
    )
    print("done")

col_to_pct_key = {
    "spw_rate_per_min": "spw",
    "swr_rate_per_min": "swr",
    "p_couple":         "p_couple",
}

print("\n  Per-session percentile of empirical value in matched bootstrap:")
print(f"  {'Session':<20} {'Duration':>10}  "
      f"{'SPW rate pctile':>17}  {'SWR rate pctile':>17}  {'P(couple) pctile':>17}")
print("  " + "-" * 90)

session_percentiles = {}
for _, row in sess_df.iterrows():
    name  = row["session"]
    bdf   = boot_dfs[name]
    p_spw = float(np.mean(bdf["spw_rate_per_min"] <= row["spw_rate_per_min"])) * 100
    p_swr = float(np.mean(bdf["swr_rate_per_min"] <= row["swr_rate_per_min"])) * 100
    p_c   = float(np.mean(bdf["p_couple"]          <= row["p_couple"]))         * 100
    session_percentiles[name] = {"spw": p_spw, "swr": p_swr, "p_couple": p_c}
    print(f"  {name:<20} {row['duration_min']:>10.1f}  "
          f"{p_spw:>17.1f}  {p_swr:>17.1f}  {p_c:>17.1f}")

# ── Figure A1 ──
session_names = list(boot_dfs.keys())
n_sessions    = len(session_names)

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
for ax, (col, ylabel, tag) in zip(axes, metric_cfg):
    for s_idx, sname in enumerate(session_names):
        bdf     = boot_dfs[sname]
        bvals   = bdf[col].dropna().values
        emp_val = float(sess_df.loc[sess_df["session"] == sname, col].iloc[0])
        lo, hi  = np.percentile(bvals, [2.5, 97.5])
        pct     = session_percentiles[sname][col_to_pct_key[col]]
        pos = s_idx
        vp  = ax.violinplot(bvals, positions=[pos], widths=0.55,
                            showmedians=True, showextrema=False)
        for pc in vp["bodies"]:
            pc.set_facecolor("#5b9bd5"); pc.set_alpha(0.45)
        vp["cmedians"].set_color("#1a5fa8"); vp["cmedians"].set_linewidth(2)
        ax.plot([pos, pos], [lo, hi], color="#1a5fa8", lw=1.2, ls="--", alpha=0.7)
        ax.scatter([pos], [emp_val], color="#e05a4b", zorder=6, s=90, marker="D")
        ax.annotate(f"{pct:.0f}th", xy=(pos, emp_val),
                    xytext=(pos + 0.25, emp_val), fontsize=8, color="#e05a4b",
                    va="center", arrowprops=dict(arrowstyle="->", color="#e05a4b", lw=0.8))
    ax.set_facecolor(TAG_COLORS[tag] + "55")
    ax.set_xticks(range(n_sessions))
    ax.set_xticklabels(
        [f"{sn}\n({sess_df.loc[sess_df['session']==sn,'duration_min'].iloc[0]:.0f} min)"
         for sn in session_names], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}\n{tag}", fontsize=9)

fig.suptitle("A: Session-level agreement  |  Null model @ ox=1.0",
             fontsize=11, fontweight="bold")
plt.tight_layout()
p = FIG_DIR / "A1_session_level_agreement.png"
save_and_show(fig, p)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — Event-level distribution agreement
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION B: Event-level distribution agreement")
print("=" * 70)

print("\n  Distributional metrics (event-pooled):")
dist_results = []
for label, emp_key, tag, xlim, nbins in event_metrics:
    r = distributional_metrics(emp[emp_key], syn_event_arrays[emp_key], label)
    if r:
        r["tag"] = tag; r["emp_key"] = emp_key
        r["xlim"] = xlim; r["nbins"] = nbins
        dist_results.append(r)

# ── Figure B1: 2×2 event-level distributions ──
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
axes = axes.flatten()
for ax, (label, emp_key, tag, xlim, nbins), r in zip(axes, event_metrics, dist_results):
    ec = emp[emp_key]; ec = ec[np.isfinite(ec)]
    sc = syn_event_arrays[emp_key]; sc = sc[np.isfinite(sc)]
    bins = np.linspace(xlim[0], xlim[1], nbins)
    ax.set_facecolor(TAG_COLORS[tag] + "44")
    ax.hist(ec, bins=bins, density=True, alpha=0.70, color="#e05a4b",
            label=f"Empirical (n={len(ec)})")
    ax.hist(sc, bins=bins, density=True, alpha=0.55, color="#5b9bd5",
            label=f"Synthetic null@1.0 (n={len(sc)})")
    if len(ec):
        ax.axvline(np.mean(ec), color="#e05a4b", lw=1.5, ls="--",
                   label=f"Emp mean: {np.mean(ec):.1f}")
    if len(sc):
        ax.axvline(np.mean(sc), color="#5b9bd5", lw=1.5, ls="--",
                   label=f"Syn mean: {np.mean(sc):.1f}")
    if "cycle" in label.lower():
        ax.axvline(3, color="red", lw=1.2, ls=":", label="Min 3 cycles")
    ks_box = (f"KS={r['ks_stat']:.3f} (p={r['ks_p']:.3f})\n"
              f"EMD={r['emd']:.2f}   Δmean={r['mean_syn']-r['mean_emp']:+.2f}")
    ax.text(0.97, 0.97, ks_box, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
    ax.set_title(f"{label}   {tag}", fontsize=9)
    ax.set_xlabel(label); ax.set_ylabel("Density")
    ax.set_xlim(xlim); ax.legend(fontsize=7, loc="upper left")

fig.suptitle("B: Event-level distribution agreement  |  Null model @ ox=1.0",
             fontsize=10, fontweight="bold")
plt.tight_layout()
p = FIG_DIR / "B1_event_distributions.png"
save_and_show(fig, p)

# ── Figure B2: SPW ISI ──
emp_isis_5 = emp["spw_isis"][emp["spw_isis"] <= 5.0] if len(emp["spw_isis"]) else np.array([])
syn_isis_5 = null_isis[null_isis <= 5.0]             if len(null_isis)       else np.array([])
lam_exp    = emp["spw_rate_per_min"] / 60.0

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.set_facecolor(TAG_COLORS["[REPRODUCED]"] + "44")
bins = np.linspace(0, 3.0, 35)
ax.hist(emp_isis_5, bins=bins, density=True, alpha=0.70, color="#e05a4b",
        label=f"Empirical ISIs (n={len(emp_isis_5)})")
ax.hist(syn_isis_5, bins=bins, density=True, alpha=0.55, color="#5b9bd5",
        label=f"Synthetic null@1.0 (n={len(syn_isis_5)})")
xg = np.linspace(0.01, 3.0, 300)
ax.plot(xg, lam_exp * np.exp(-lam_exp * xg), "k--", lw=1.4,
        label=f"Exponential(λ={lam_exp:.2f}/s)")
r_isi = distributional_metrics(emp_isis_5, syn_isis_5, "SPW ISI (s)")
if r_isi:
    ks_box = (f"KS={r_isi['ks_stat']:.3f} (p={r_isi['ks_p']:.3f})\n"
              f"EMD={r_isi['emd']:.3f}")
    ax.text(0.97, 0.97, ks_box, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
ax.set_title("SPW ISI distribution   [REPRODUCED]", fontsize=9)
ax.set_xlabel("SPW ISI (s)"); ax.set_ylabel("Density")
ax.legend(fontsize=8)
plt.tight_layout()
p = FIG_DIR / "B2_spw_isi.png"
save_and_show(fig, p)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — Pharmacology alignment (Kostansek et al. 2023)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION C: Pharmacology alignment check")
print("=" * 70)

base_trials   = libraries["null_paper_band"][f"{CONTEXT}_ox{ox_key(OX_NULL)}"]
base_spw_rate = pooled_rate_from_trials(base_trials, "n_spw",    duration_sec)
base_swr_rate = pooled_rate_from_trials(base_trials, "n_coupled",duration_sec)
base_pcoup    = pooled_p_couple(base_trials)
base_spw_d, base_spw_a    = pooled_spw_arrays(base_trials)
base_swr_d, base_swr_f, _ = pooled_coupled_event_arrays(base_trials)
base_spw_amp  = _mean_or_nan(base_spw_a)
base_spw_dur  = _mean_or_nan(base_spw_d)
base_swr_dur  = _mean_or_nan(base_swr_d)

paper_targets = {
    "sb_ox1r":   [None, None, None, 0.55, None, None],
    "empa_ox2r": [None, None, None, 0.38, None, None],
    "dora":      [None, None, None, 0.51, None, None],
}

drug_models  = ["sb_ox1r", "empa_ox2r", "dora"]
metric_names = ["SPW rate", "SPW amp", "SPW dur", "SWR rate", "SWR dur", "P(coup)"]
colors_d     = {"sb_ox1r": "#4472c4", "empa_ox2r": "#ed7d31", "dora": "#70ad47"}

mult = {m: [] for m in drug_models}
for m in drug_models:
    trials   = libraries[m][f"{CONTEXT}_ox{ox_key(OX_DRUG)}"]
    spw_rate = pooled_rate_from_trials(trials, "n_spw",    duration_sec)
    swr_rate = pooled_rate_from_trials(trials, "n_coupled",duration_sec)
    pc       = pooled_p_couple(trials)
    spw_d, spw_a = pooled_spw_arrays(trials)
    swr_d, _, _  = pooled_coupled_event_arrays(trials)
    mult[m] = [
        spw_rate / base_spw_rate if base_spw_rate else np.nan,
        _mean_or_nan(spw_a) / base_spw_amp if base_spw_amp else np.nan,
        _mean_or_nan(spw_d) / base_spw_dur if base_spw_dur else np.nan,
        swr_rate / base_swr_rate if base_swr_rate else np.nan,
        _mean_or_nan(swr_d) / base_swr_dur if base_swr_dur else np.nan,
        pc / base_pcoup if base_pcoup else np.nan,
    ]

fig, ax = plt.subplots(figsize=(10, 5.0))
x = np.arange(len(metric_names))
w = 0.24
for i, m in enumerate(drug_models):
    vals = np.array(mult[m], dtype=float)
    ax.bar(x + (i - 1) * w, vals, width=w, label=m.upper(),
           color=colors_d[m], alpha=0.82)
    for j, v in enumerate(vals):
        if not np.isfinite(v): continue
        glyph = "↑" if v > 1.05 else ("↓" if v < 0.95 else "~")
        ax.text(x[j] + (i - 1) * w, v + 0.02, glyph,
                ha="center", va="bottom", fontsize=11)
        target = paper_targets[m][j]
        if target is not None:
            ax.plot([x[j] + (i-1)*w - w*0.4, x[j] + (i-1)*w + w*0.4],
                    [target, target], color="black", lw=2.0, zorder=5)
ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.0)
ax.set_xticks(x); ax.set_xticklabels(metric_names)
ax.set_ylabel("Multiplier vs Null-PaperBand @ ox=1.0")
target_line = plt.Line2D([0],[0], color="black", lw=2.5,
                          label="Paper target (Kostansek et al. 2023)")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles + [target_line], loc="upper right", fontsize=8.5)
ax.set_title("C: Pharmacology alignment @ ox=0.3  |  Kostansek et al. 2023",
             fontsize=9, fontweight="bold")
plt.tight_layout()
p = FIG_DIR / "C1_pharmacology_alignment.png"
save_and_show(fig, p)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — Distribution family comparison: Gamma vs Lognormal
#             Per-variable selection:
#               SPW amplitude: lognormal (literature-committed)
#               Ripple power:  AIC-selected (data-driven)
#             (Mizuseki & Buzsáki 2013; Buzsáki & Mizuseki 2014)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION D: Distribution family comparison — Gamma vs Lognormal")
print("  SPW amplitude: lognormal (committed — literature + empirical)")
print("  Ripple power:  AIC-selected (data-driven, no override)")
print("=" * 70)

# Fit both families to empirical data
amp_comp = fit_and_compare(emp["spw_amps"], "SPW amplitude (µV)")
pow_comp = fit_and_compare(emp["swr_pows"], "Ripple power") if len(emp["swr_pows"]) > 10 else None

# ── Figure D1: SPW amplitude — empirical + gamma fit + lognormal fit ──
fig, axes = plt.subplots(1, 2 if pow_comp else 1,
                          figsize=(12 if pow_comp else 6.5, 5))
if pow_comp is None:
    axes = [axes]

for ax, comp, title_label in zip(
    axes if pow_comp else [axes[0]],
    [amp_comp] + ([pow_comp] if pow_comp else []),
    ["SPW amplitude"] + (["Ripple power"] if pow_comp else []),
):
    x_data = comp["data"]
    xmin, xmax = np.percentile(x_data, [0.5, 99.5])
    xgrid = np.linspace(xmin, xmax, 300)
    bins  = np.linspace(xmin, xmax, 50)

    ax.hist(x_data, bins=bins, density=True, alpha=0.55, color="#888888",
            label=f"Empirical (n={comp['n']})", edgecolor="white", linewidth=0.3)

    # Gamma PDF
    g_pdf = gamma_dist.pdf(xgrid, *comp["g_params"])
    ax.plot(xgrid, g_pdf, color="#e05a4b", lw=2.0, ls="--",
            label=f"Gamma  AIC={comp['g_aic']:.0f}")

    # Lognormal PDF
    ln_pdf = lognorm_dist.pdf(xgrid, *comp["ln_params"])
    ax.plot(xgrid, ln_pdf, color="#2c7fb8", lw=2.5,
            label=f"Lognorm  AIC={comp['ln_aic']:.0f}")

    # Winner annotation
    delta = comp["delta_aic"]
    aic_winner = "Lognormal" if delta > 0 else "Gamma"
    # SPW amplitude: committed to lognormal; Ripple power: AIC-selected
    if "amplitude" in title_label.lower():
        gen_uses = "Lognormal (committed)"
        ref_line = "(Mizuseki & Buzsáki 2013)"
        box_color = "#d4edda"
    else:
        gen_uses = f"{aic_winner} (AIC-selected)"
        ref_line = "(data-driven, no override)"
        box_color = "#cce5ff"
    ax.text(0.97, 0.85,
            f"ΔAIC = {delta:+.1f}\n"
            f"AIC prefers: {aic_winner}\n"
            f"Generator uses: {gen_uses}\n"
            f"{ref_line}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.85))

    ax.set_xlabel(title_label)
    ax.set_ylabel("Density")
    ax.set_title(f"{title_label}: Gamma vs Lognormal fit to empirical data\n"
                 f"[FIT-PRESERVED]",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle(
    "D: Distribution family comparison\n"
    "SPW amplitude: lognormal (literature-committed)  |  "
    "Ripple power: AIC-selected\n"
    "Ref: Mizuseki & Buzsáki (2013) Cell Rep; "
    "Buzsáki & Mizuseki (2014) Nat Rev Neurosci",
    fontsize=10, fontweight="bold",
)
plt.tight_layout()
p = FIG_DIR / "D1_gamma_vs_lognormal_comparison.png"
save_and_show(fig, p)

# ── Figure D2: Synthetic vs empirical amplitude (lognormal sampling check) ──
if len(null_spw_a) > 0 and len(emp["spw_amps"]) > 0:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xmin, xmax = np.percentile(
        np.concatenate([emp["spw_amps"], null_spw_a]), [0.5, 99.5])
    bins = np.linspace(xmin, xmax, 50)
    ax.hist(emp["spw_amps"], bins=bins, density=True, alpha=0.70,
            color="#e05a4b", label=f"Empirical (n={len(emp['spw_amps'])})")
    ax.hist(null_spw_a, bins=bins, density=True, alpha=0.55,
            color="#5b9bd5", label=f"Synthetic null@1.0 (n={len(null_spw_a)})")

    r_amp = distributional_metrics(emp["spw_amps"], null_spw_a, "SPW amplitude")
    if r_amp:
        ks_box = (f"KS={r_amp['ks_stat']:.3f} (p={r_amp['ks_p']:.3f})\n"
                  f"EMD={r_amp['emd']:.1f}  Δmean={r_amp['mean_syn']-r_amp['mean_emp']:+.1f}")
        ax.text(0.97, 0.97, ks_box, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))

    ax.set_title("SPW amplitude: Empirical vs Synthetic (lognormal-committed)\n"
                 "[FIT-PRESERVED]", fontsize=9)
    ax.set_xlabel("SPW amplitude"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = FIG_DIR / "D2_spw_amplitude_empirical_vs_synthetic.png"
    save_and_show(fig, p)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — Parameter stability: leave-one-session-out
#             Shows lognormal fit is not dominated by any single session
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION E: Parameter stability — leave-one-session-out")
print("=" * 70)

n_sess = len(_spw_amps_all)
sess_names_raw = [r["session"] for r in _sess_rows]

# Full fit (all sessions)
all_amps_pooled = np.concatenate(_spw_amps_all)
all_amps_pos    = all_amps_pooled[all_amps_pooled > 0]
full_ln_params  = lognorm_dist.fit(all_amps_pos, floc=0)
full_s, _, full_scale = full_ln_params

print(f"\n  Full fit (all {n_sess} sessions, n={len(all_amps_pos)}):")
print(f"    lognorm s={full_s:.4f}  scale={full_scale:.1f}")

print(f"\n  Leave-one-out fits:")
print(f"  {'Held-out':<20} {'n_train':<10} {'s':<10} {'Δs (%)':<10} "
      f"{'scale':<10} {'Δscale (%)':<12}")
print("  " + "-" * 75)

loo_results = []
for i in range(n_sess):
    held_name = sess_names_raw[i]
    train_amps = np.concatenate([_spw_amps_all[j] for j in range(n_sess) if j != i])
    train_pos  = train_amps[train_amps > 0]
    loo_params = lognorm_dist.fit(train_pos, floc=0)
    loo_s, _, loo_scale = loo_params

    ds_pct     = (loo_s - full_s) / full_s * 100
    dscale_pct = (loo_scale - full_scale) / full_scale * 100

    loo_results.append({
        "held_out": held_name, "n_train": len(train_pos),
        "s": loo_s, "scale": loo_scale,
        "ds_pct": ds_pct, "dscale_pct": dscale_pct,
    })

    print(f"  {held_name:<20} {len(train_pos):<10} {loo_s:<10.4f} "
          f"{ds_pct:<+10.2f} {loo_scale:<10.1f} {dscale_pct:<+12.2f}")

max_ds   = max(abs(r["ds_pct"]) for r in loo_results)
max_dscl = max(abs(r["dscale_pct"]) for r in loo_results)

print(f"\n  Max |Δs|:     {max_ds:.2f}%")
print(f"  Max |Δscale|: {max_dscl:.2f}%")
stable = max_ds < 10 and max_dscl < 10
print(f"  Parameter stability (<10% deviation): {'PASS' if stable else 'CHECK'}")

# Same for ripple power if available
if len(_swr_pows_all) == n_sess and all(len(p) > 0 for p in _swr_pows_all):
    all_pows_pos = np.concatenate(_swr_pows_all)
    all_pows_pos = all_pows_pos[all_pows_pos > 0]
    full_pow_params = lognorm_dist.fit(all_pows_pos, floc=0)
    full_pow_s, _, full_pow_scale = full_pow_params

    print(f"\n  Ripple power — full fit: s={full_pow_s:.4f}  scale={full_pow_scale:.1f}")
    print(f"  {'Held-out':<20} {'s':<10} {'Δs (%)':<10} {'scale':<10} {'Δscale (%)':<12}")
    print("  " + "-" * 55)
    for i in range(n_sess):
        train_pows = np.concatenate([_swr_pows_all[j] for j in range(n_sess) if j != i])
        train_pos  = train_pows[train_pows > 0]
        lp = lognorm_dist.fit(train_pos, floc=0)
        ds = (lp[0] - full_pow_s) / full_pow_s * 100
        dsc = (lp[2] - full_pow_scale) / full_pow_scale * 100
        print(f"  {sess_names_raw[i]:<20} {lp[0]:<10.4f} {ds:<+10.2f} "
              f"{lp[2]:<10.1f} {dsc:<+12.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — Modulated-metrics validation
#             After orexin-dependent modulation, do SWR features remain
#             within physiologically plausible bounds?
#             This is the core validation claim: modulation ≠ artifact.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION F: Modulated-metrics validation")
print("  After OX2R modulation, do SWR features stay physiologically plausible?")
print("=" * 70)

# Physiological bounds (from Girardeau et al. 2017 + Kostansek et al. 2023)
PHYS_BOUNDS = {
    "swr_dur_ms":  (10, 200),    # ripple duration
    "swr_freq_hz": (80, 300),    # ripple frequency (broad, covers paper + in-vivo bands)
    "swr_cycles":  (3, 25),      # cycle count
    "coup_lag_ms": (-30, 80),    # coupling lag
}

# Test across the EMPA-OX2R model (the one the agent uses)
ox_levels = lib.get("orexin_levels", [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8])

print(f"\n  Model: empa_ox2r  |  OX2R levels: {len(ox_levels)}")
print(f"\n  {'OX2R':<7} {'n_SWR':<8} {'dur(ms)':<12} {'freq(Hz)':<12} "
      f"{'cycles':<10} {'lag(ms)':<12} {'violations'}")
print("  " + "-" * 80)

total_violations = 0
modulated_rows = []
for ox in ox_levels:
    key = f"{CONTEXT}_ox{ox_key(ox)}"
    if key not in libraries["empa_ox2r"]:
        continue
    trials = libraries["empa_ox2r"][key]
    sd, sf, sc = pooled_coupled_event_arrays(trials)
    sl = np.concatenate([_safe_np(t.get("coupling_lags", [])) for t in trials]).astype(float)
    sl = sl[np.isfinite(sl)]

    n_swr = len(sd)
    m_dur  = _mean_or_nan(sd)
    m_freq = _mean_or_nan(sf)
    m_cyc  = _mean_or_nan(sc)
    m_lag  = _mean_or_nan(sl)

    # Count out-of-bound events
    v = 0
    if len(sd): v += int(np.sum((sd < PHYS_BOUNDS["swr_dur_ms"][0]) |
                                 (sd > PHYS_BOUNDS["swr_dur_ms"][1])))
    if len(sf): v += int(np.sum((sf < PHYS_BOUNDS["swr_freq_hz"][0]) |
                                 (sf > PHYS_BOUNDS["swr_freq_hz"][1])))
    if len(sc): v += int(np.sum(sc < PHYS_BOUNDS["swr_cycles"][0]))

    total_violations += v

    print(f"  {ox:<7.2f} {n_swr:<8} {m_dur:<12.1f} {m_freq:<12.1f} "
          f"{m_cyc:<10.1f} {m_lag:<12.1f} {v}")

    modulated_rows.append({
        "ox": ox, "n_swr": n_swr,
        "dur": m_dur, "freq": m_freq, "cyc": m_cyc, "lag": m_lag,
        "violations": v,
    })

print(f"\n  Total out-of-bounds events across all levels: {total_violations}")
all_n = sum(r["n_swr"] for r in modulated_rows)
pct_clean = (1 - total_violations / max(all_n, 1)) * 100
print(f"  Events within physiological bounds: {pct_clean:.1f}%")
print(f"  Modulation preserves physiological plausibility: "
      f"{'PASS' if pct_clean > 95 else 'CHECK'}")

# ── Figure F1: modulated metrics across OX2R ──
if modulated_rows:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    ox_vals = [r["ox"] for r in modulated_rows]
    plot_cfg = [
        ("dur",  "SWR duration (ms)",  PHYS_BOUNDS["swr_dur_ms"]),
        ("freq", "SWR frequency (Hz)", PHYS_BOUNDS["swr_freq_hz"]),
        ("cyc",  "SWR cycles",         PHYS_BOUNDS["swr_cycles"]),
        ("lag",  "Coupling lag (ms)",   PHYS_BOUNDS["coup_lag_ms"]),
    ]
    for ax, (key, ylabel, (lo, hi)) in zip(axes, plot_cfg):
        vals = [r[key] for r in modulated_rows]
        ax.plot(ox_vals, vals, "o-", color="#2c7fb8", lw=2, markersize=6)
        ax.axhspan(lo, hi, alpha=0.10, color="#4caf50",
                   label="Physiological range")
        ax.axhline(lo, color="#4caf50", ls=":", lw=0.8)
        ax.axhline(hi, color="#4caf50", ls=":", lw=0.8)
        ax.axvline(1.0, color="gray", ls="--", lw=0.8, label="Baseline OX2R")
        ax.set_xlabel("OX2R level")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "F: Modulated metrics across OX2R  |  EMPA-OX2R model\n"
        "Green band = physiological bounds.  "
        "Validation: modulation does not produce artifact events.",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    p = FIG_DIR / "F1_modulated_metrics_across_ox2r.png"
    save_and_show(fig, p)


# ─────────────────────────────────────────────────────────────────────────────
# Final console summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY (v5.8.3)")
print("=" * 70)

null_spw_rate = pooled_rate_from_trials(null_trials, "n_spw",    duration_sec)
null_swr_rate = pooled_rate_from_trials(null_trials, "n_coupled",duration_sec)
null_pcoup    = pooled_p_couple(null_trials)

print("\n  ── (i) Model-construction constraints ──")
print(f"  {'Metric':<35} {'Empirical':<16} {'Synthetic':<16} {'Tag'}")
print("  " + "-" * 80)
print(f"  {'SPW rate (/min)':<35} {emp['spw_rate_per_min']:<16.2f} {null_spw_rate:<16.2f} [CONSTRAINED]")
print(f"  {'P(couple)':<35} {emp['p_couple']:<16.4f} {null_pcoup:<16.4f} [CONSTRAINED]")

print("\n  ── (ii) Fit-preserved quantities ──")
print(f"  {'Metric':<35} {'Empirical':<16} {'Synthetic':<16} {'Tag'}")
print("  " + "-" * 80)
print(f"  {'SWR rate (/min)':<35} {emp['swr_rate_per_min']:<16.2f} {null_swr_rate:<16.2f} [FIT-PRESERVED]")
print(f"  {'SWR frequency (Hz)':<35} {_mean_or_nan(emp['swr_freqs']):<16.2f} {_mean_or_nan(null_swr_f):<16.2f} [FIT-PRESERVED]")
print(f"  {'SWR cycles':<35} {_mean_or_nan(emp['swr_cycles']):<16.2f} {_mean_or_nan(null_swr_c):<16.2f} [FIT-PRESERVED]")
print(f"  {'Coupling lag (ms)':<35} {_mean_or_nan(emp['coupling_lags']):<16.2f} {_mean_or_nan(null_lags):<16.2f} [FIT-PRESERVED]")
print(f"  {'SPW amplitude (lognorm)':<35} {_mean_or_nan(emp['spw_amps']):<16.1f} {_mean_or_nan(null_spw_a):<16.1f} [FIT-PRESERVED]")

print("\n  ── (iii) Genuine validation outcomes ──")
print(f"  {'Metric':<35} {'Empirical':<16} {'Synthetic':<16} {'Tag'}")
print("  " + "-" * 80)
print(f"  {'SWR duration (ms)':<35} {_mean_or_nan(emp['swr_durs']):<16.2f} {_mean_or_nan(null_swr_d):<16.2f} [REPRODUCED]")
print(f"  {'SPW ISI shape':<35} {'(see Fig B2)':<16} {'(see Fig B2)':<16} [REPRODUCED]")

print("\n  ── (iv) Distribution family (Section D) ──")
print(f"  {'SPW amplitude':<35} {'lognormal':<16} ΔAIC={amp_comp['delta_aic']:+.1f}  [committed — literature]")
if pow_comp:
    pow_active = pow_comp['winner']
    print(f"  {'Ripple power':<35} {pow_active:<16} ΔAIC={pow_comp['delta_aic']:+.1f}  [AIC-selected]")
print(f"  {'SPW/ripple duration':<35} {'gamma':<16} (waiting-time process)")
print(f"  {'ISIs':<35} {'gamma':<16} (inter-event intervals)")

min_cyc = float(np.min(null_swr_c)) if len(null_swr_c) else np.nan
viol    = int(np.sum(null_swr_c < 3)) if len(null_swr_c) else 0
print(f"\n  ≥3 cycles constraint: Min={min_cyc:.2f}, Violations={viol}/{len(null_swr_c)}")

print(f"\n  ── (v) Parameter stability — LOO (Section E) ──")
print(f"  Max |Δs|:     {max_ds:.2f}%")
print(f"  Max |Δscale|: {max_dscl:.2f}%")
print(f"  Stable (<10%): {'PASS' if stable else 'CHECK'}")

print(f"\n  ── (vi) Modulated-metrics validation (Section F) ──")
print(f"  Events within physiological bounds: {pct_clean:.1f}%")
print(f"  Modulation preserves plausibility: {'PASS' if pct_clean > 95 else 'CHECK'}")

print(f"\n  Figures saved to: {FIG_DIR}")
print("=" * 70)
print("Ready for Cell 9")
print("  NOTE: Agent model must update library path to v5.8.3:")
print("    lib_path = DIRS['models'] / 'synthetic_libraries.pkl'")
print("    cfg_path = DIRS['models'] / 'generator_config.pkl'")
print("=" * 70)
