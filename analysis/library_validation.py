"""
Library validation - synthetic against empirical statistics.

Reads:  data_processed/swr_detection.pkl
        models/synthetic_libraries.pkl
        models/generator_config.pkl

Each metric is tagged by provenance: [CONSTRAINED] was calibrated to,
[FIT-PRESERVED] follows from a fitted distribution, [REPRODUCED] was
neither and is therefore a genuine check.

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
from orexin_keys import ox_key  # noqa: E402,F401

ensure_dirs()

# =============================================================================
# LIBRARY VALIDATION PLOTS (training-session reference)
#
# Reads:
#   - swr_detection.pkl        empirical detections (all five sessions)
#   - synthetic_libraries.pkl  agent libraries (200 trials per condition)
#   - generator_config.pkl
#   - validation_null.pkl       null model, ox = 1.0, 5000 trials, seed 777
#
# Revision (Sep 2026)
#   1. The empirical reference is the three TRAINING sessions only. The detector
#      pickle also holds the held-out session (Buddy) and the excluded session
#      (Gatsby); reading all five contaminated every comparison in Sections A-E.
#   2. Sections A, B, D2 and the ISI test compare against the 5000-trial
#      validation library (the one Figure 2 uses) instead of the 200-trial agent
#      library, which held only 69 synthetic SWR events. Sections C and F still
#      read the agent libraries, because they check those libraries themselves.
#   3. Provenance check: both libraries must list the training set as their
#      source_sessions, or the cell stops.
#   4. Empirical rates are time-weighted (total events / total time), which is
#      how the generator anchors its baseline. Session means are printed too.
#   5. SPW ISIs. A synthetic trial is a 3 s window: it cannot contain an interval
#      longer than 3 s and it favours short intervals (an interval of length x
#      fits inside the window with weight proportional to 3 - x). Empirical ISIs
#      are now observed through the same non-overlapping 3 s windows inside the
#      NREM/Intermediate intervals, and the reference curve is a Poisson process
#      seen through a 3 s window.
#   6. The distribution-family KS uses the one-sample test, as in the generator, so
#      Section D reproduces the generator's numbers.
#   7. The Section A bootstrap is vectorised (same statistic, much faster).
#   8. New Section G: the held-out session (Sample 4) against the synthetic
#      reference. Sample 5 is excluded on detection yield and is not tested.
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
import matplotlib.pyplot as plt
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
print("LIBRARY VALIDATION (training-session reference)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
# Paths come from orexin_paths (see header). DIRS / FILES are already bound.
FIG_DIR = DIRS["figures"] / "library_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_PKL = FILES["detector"]
LIB_PKL      = FILES["libraries"]
CFG_PKL      = FILES["generator_config"]
VAL_PKL      = FILES["validation_null"]

CONTEXT  = "Context_A"
OX_NULL  = 1.0
OX_DRUG  = 0.3
N_BOOT   = 2000

# ─────────────────────────────────────────────────────────────────────────────
# Session roles (must match GENERATOR_SESSIONS and the Figure 1 key)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SESSIONS    = ['Achilles_10252013', 'Cicero_09102014', 'Cicero_09172014']
HELDOUT_SESSIONS  = ['Buddy_06272013']
EXCLUDED_SESSIONS = ['Gatsby_08282013']   # excluded on detection yield
SAMPLE_LABEL = {
    'Achilles_10252013': 'Sample 1',
    'Cicero_09102014':   'Sample 2',
    'Cicero_09172014':   'Sample 3',
    'Buddy_06272013':    'Sample 4',
    'Gatsby_08282013':   'Sample 5',
}

def lbl(name):
    return SAMPLE_LABEL.get(name, name)

# ox_key comes from orexin_keys (see header).

plt.rcParams["figure.dpi"]        = 120
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

def save_and_show(fig, filepath):
    """Save figure to disk and display if running inside Jupyter."""
    fig.savefig(filepath, bbox_inches="tight")
    display(fig)
    plt.show()
    plt.close(fig)
    print(f"  Saved: {filepath}")

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
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe_np(a):
    return np.array(a) if a is not None else np.array([])

def _mean_or_nan(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else np.nan

def _cat(lst):
    lst = [np.asarray(a) for a in lst if len(a)]
    return np.concatenate(lst) if lst else np.array([])

def coupled_event_arrays_from_trial(trial):
    key_sets = [
        ("swr_durations",            "swr_frequencies",            "swr_cycles"),
        ("swr_event_durations",      "swr_event_frequencies",      "swr_event_cycles"),
        ("coupled_ripple_durations", "coupled_ripple_frequencies", "coupled_ripple_cycles"),
    ]
    for kd, kf, kc in key_sets:
        if kd in trial and kf in trial and kc in trial:
            d = _safe_np(trial[kd]).astype(float)
            f = _safe_np(trial[kf]).astype(float)
            c = _safe_np(trial[kc]).astype(float)
            return d[np.isfinite(d)], f[np.isfinite(f)], c[np.isfinite(c)]

    spw_times = _safe_np(trial.get("spw_times",      [])).astype(float)
    spw_has   = _safe_np(trial.get("spw_has_ripple", [])).astype(bool)
    rps       = _safe_np(trial.get("ripples_per_spw",[])).astype(int)
    lags      = _safe_np(trial.get("coupling_lags",  [])).astype(float)
    rt = _safe_np(trial.get("ripple_times",      [])).astype(float)
    rd = _safe_np(trial.get("ripple_durations",  [])).astype(float)
    rf = _safe_np(trial.get("ripple_frequencies",[])).astype(float)
    rc = _safe_np(trial.get("ripple_cycles",     [])).astype(float)

    if len(spw_times) == 0 or len(lags) == 0 or len(rt) == 0:
        return np.array([]), np.array([]), np.array([])

    coupled_times, idx = [], 0
    for i in range(len(spw_times)):
        if i < len(spw_has) and spw_has[i]:
            n_here = int(rps[i]) if i < len(rps) and rps[i] > 0 else 1
            for _ in range(n_here):
                if idx >= len(lags): break
                coupled_times.append(spw_times[i] + lags[idx] / 1000.0)
                idx += 1
    coupled_times = np.array(coupled_times, dtype=float)
    if len(coupled_times) == 0:
        return np.array([]), np.array([]), np.array([])

    d_out, f_out, c_out = [], [], []
    for t0 in coupled_times:
        j    = int(np.searchsorted(rt, t0))
        cand = [jj for jj in [j, j - 1] if 0 <= jj < len(rt)]
        if not cand: continue
        k = min(cand, key=lambda ii: abs(rt[ii] - t0))
        if abs(rt[k] - t0) > 1e-3: continue
        d_out.append(rd[k]); f_out.append(rf[k]); c_out.append(rc[k])

    d_out = np.array(d_out, dtype=float)
    f_out = np.array(f_out, dtype=float)
    c_out = np.array(c_out, dtype=float)
    return d_out[np.isfinite(d_out)], f_out[np.isfinite(f_out)], c_out[np.isfinite(c_out)]


def pooled_coupled_event_arrays(trials):
    d_all, f_all, c_all = [], [], []
    for t in trials:
        d, f, c = coupled_event_arrays_from_trial(t)
        if len(d): d_all.append(d)
        if len(f): f_all.append(f)
        if len(c): c_all.append(c)
    return (
        np.concatenate(d_all) if d_all else np.array([]),
        np.concatenate(f_all) if f_all else np.array([]),
        np.concatenate(c_all) if c_all else np.array([]),
    )

def pooled_rate_from_trials(trials, count_key, duration_sec):
    total      = sum(int(t.get(count_key, 0)) for t in trials)
    total_time = len(trials) * float(duration_sec)
    return (total / total_time) * 60.0 if total_time > 0 else 0.0

def pooled_p_couple(trials):
    tot_spw = sum(int(t.get("n_spw",             0)) for t in trials)
    tot_c   = sum(int(t.get("n_spw_with_ripple", 0)) for t in trials)
    return (tot_c / tot_spw) if tot_spw > 0 else np.nan

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
                                  n_boot=2000, rng_seed=0, chunk=250):
    """
    Synthetic pseudo-sessions of the empirical session's length, built by
    resampling 3 s null-model trials with replacement. Returns one row per
    pseudo-session with its SPW rate, SWR rate and coupling probability.
    Vectorised over trials; the statistic is unchanged.
    """
    rng       = np.random.default_rng(rng_seed)
    n_needed  = max(1, int(np.ceil(target_minutes * 60.0 / duration_sec)))
    total_sec = n_needed * duration_sec
    spw = np.array([int(t.get("n_spw",             0)) for t in null_trials], dtype=np.int64)
    swr = np.array([int(t.get("n_coupled",         0)) for t in null_trials], dtype=np.int64)
    cpl = np.array([int(t.get("n_spw_with_ripple", 0)) for t in null_trials], dtype=np.int64)
    out = []
    for start in range(0, n_boot, chunk):
        m   = min(chunk, n_boot - start)
        idx = rng.integers(0, len(null_trials), size=(m, n_needed))
        s_spw = spw[idx].sum(axis=1)
        s_swr = swr[idx].sum(axis=1)
        s_c   = cpl[idx].sum(axis=1)
        p_c   = np.where(s_spw > 0, s_c / np.maximum(s_spw, 1), np.nan)
        out.append(pd.DataFrame({
            "spw_rate_per_min": s_spw / total_sec * 60.0,
            "swr_rate_per_min": s_swr / total_sec * 60.0,
            "p_couple":         p_c,
        }))
    return pd.concat(out, ignore_index=True)

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
          f"EMD={emd:.2f}  Δmean={mean_diff:+.2f} ({rel_err:+.1f}%)  "
          f"n_emp={len(ec)}  n_syn={len(sc)}")
    return {
        "label": label, "ks_stat": ks_stat, "ks_p": ks_p, "emd": emd,
        "mean_emp": np.mean(ec), "mean_syn": np.mean(sc),
        "median_emp": np.median(ec), "median_syn": np.median(sc),
        "rel_mean_err_pct": rel_err, "n_emp": len(ec), "n_syn": len(sc),
    }

def fit_and_compare(data, label):
    """
    Fit gamma and lognormal (loc fixed at 0) and compare by AIC.
    KS is the one-sample test against each fitted CDF, as in the generator, so these
    numbers should reproduce the generator's printout exactly.
    """
    x = data[data > 0].copy()
    n = len(x)
    k_p = 2
    g_params  = gamma_dist.fit(x, floc=0)
    g_ll      = np.sum(gamma_dist.logpdf(x, *g_params))
    g_aic     = 2*k_p - 2*g_ll
    g_ks, g_p = stats.kstest(x, 'gamma', args=g_params)
    ln_params  = lognorm_dist.fit(x, floc=0)
    ln_ll      = np.sum(lognorm_dist.logpdf(x, *ln_params))
    ln_aic     = 2*k_p - 2*ln_ll
    ln_ks, ln_p = stats.kstest(x, 'lognorm', args=ln_params)
    delta_aic = g_aic - ln_aic
    winner = "lognorm" if delta_aic > 0 else "gamma"
    print(f"  {label}:  n={n}")
    print(f"    Gamma    AIC={g_aic:.1f}  KS={g_ks:.4f} (p={g_p:.4f})")
    print(f"    Lognorm  AIC={ln_aic:.1f}  KS={ln_ks:.4f} (p={ln_p:.4f})")
    print(f"    ΔAIC={delta_aic:+.1f}  →  AIC prefers: {winner}")
    return {
        "label": label, "data": x, "n": n,
        "g_params": g_params, "g_aic": g_aic, "g_ks": g_ks, "g_p": g_p,
        "ln_params": ln_params, "ln_aic": ln_aic, "ln_ks": ln_ks, "ln_p": ln_p,
        "delta_aic": delta_aic, "winner": winner,
    }

def get_intervals(sess):
    """NREM/Intermediate intervals (s) for a detector session; same key logic
    as the generator."""
    ni = sess["nrem_intervals"]
    if "intervals_sec" in ni:
        return [tuple(iv) for iv in ni["intervals_sec"]]
    if "intervals" in ni:
        return [tuple(iv) for iv in ni["intervals"]]
    if "start_times" in ni and "end_times" in ni:
        return list(zip(ni["start_times"], ni["end_times"]))
    return [(0.0, float(ni["total_sec"]))]

def windowed_isis(event_times, intervals, win_sec):
    """
    ISIs observed through non-overlapping windows of win_sec laid inside each
    interval. This is the observation process of a synthetic trial, so the two
    ISI distributions are compared on equal terms.
    """
    t = np.sort(np.asarray(event_times, dtype=float))
    out = []
    for s, e in intervals:
        w0 = float(s)
        while w0 + win_sec <= e:
            i0, i1 = np.searchsorted(t, [w0, w0 + win_sec], side="left")
            if i1 - i0 > 1:
                d = np.diff(t[i0:i1])
                d = d[d > 0]
                if len(d): out.append(d)
            w0 += win_sec
    return np.concatenate(out) if out else np.array([])

def poisson_window_isi_pdf(x, lam, T):
    """ISI density of a Poisson process (rate lam, 1/s) seen through a window of length T."""
    x = np.asarray(x, dtype=float)
    norm = T / lam - (1.0 - np.exp(-lam * T)) / lam**2
    return np.where((x >= 0) & (x < T), (T - x) * np.exp(-lam * x) / norm, 0.0)

def poisson_window_isi_cdf(x, lam, T):
    x = np.clip(np.asarray(x, dtype=float), 0.0, T)
    norm = T / lam - (1.0 - np.exp(-lam * T)) / lam**2
    num  = T * (1.0 - np.exp(-lam * x)) / lam - (1.0 - np.exp(-lam * x) * (1.0 + lam * x)) / lam**2
    return num / norm


# ─────────────────────────────────────────────────────────────────────────────
# [1] Load empirical detector output
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading empirical detector output...")
if not DETECTOR_PKL.exists():
    raise FileNotFoundError(f"Missing: {DETECTOR_PKL}")
with open(DETECTOR_PKL, "rb") as f:
    det = pickle.load(f)

by_name  = {s["name"]: s for s in det["datasets"]}
_missing = [n for n in TRAIN_SESSIONS + HELDOUT_SESSIONS if n not in by_name]
if _missing:
    raise KeyError(f"Sessions missing from the detector pickle: {_missing}")
print(f"  Detector pickle holds: {sorted(by_name)}")
print(f"  Empirical reference (training): {[lbl(n) for n in TRAIN_SESSIONS]}")
print(f"  Held out (Section G only):      {[lbl(n) for n in HELDOUT_SESSIONS]}")
print(f"  Excluded (not tested):          {[lbl(n) for n in EXCLUDED_SESSIONS]}")

def session_features(sess):
    ts   = float(sess["nrem_intervals"]["total_sec"])
    spw  = sess["spw_events"]
    swr  = sess["ripple_tiers"]["ripples_swr"]
    coup = sess["coupling_swr"]

    def fin(a):
        a = np.asarray(a, dtype=float)
        return a[np.isfinite(a)]

    cyc = []
    for e in swr:
        if e.get("n_cycles") is not None:
            cyc.append(float(e["n_cycles"]))
        else:
            pf = e.get("peak_freq", np.nan)
            cyc.append((e["duration_ms"] / 1000.0) * pf if np.isfinite(pf) else np.nan)

    n_spw = len(spw); n_swr = len(swr)
    n_c   = int(coup.get("n_coupled_spw", 0))
    return {
        "session":          sess["name"],
        "label":            lbl(sess["name"]),
        "total_sec":        ts,
        "duration_min":     ts / 60.0,
        "n_spw": n_spw, "n_swr": n_swr, "n_coupled_spw": n_c,
        "spw_rate_per_min": n_spw / ts * 60.0 if ts > 0 else np.nan,
        "swr_rate_per_min": n_swr / ts * 60.0 if ts > 0 else np.nan,
        "p_couple":         n_c / n_spw        if n_spw > 0 else np.nan,
        "spw_times":        np.sort(np.array([e["peak_sec"] for e in spw], dtype=float)),
        "intervals":        get_intervals(sess),
        "spw_amps":         fin([e["amplitude"] for e in spw]),
        "swr_durs":         fin([e["duration_ms"] for e in swr]),
        "swr_freqs":        fin([e.get("peak_freq", np.nan) for e in swr]),
        "swr_cycles":       fin(cyc),
        "swr_pows":         fin([e.get("power", np.nan) for e in swr]),
        "coupling_lags":    fin(coup.get("coupling_lags_ms", [])),
    }

train_feats   = [session_features(by_name[n]) for n in TRAIN_SESSIONS]
heldout_feats = [session_features(by_name[n]) for n in HELDOUT_SESSIONS]

sess_df = pd.DataFrame([{k: f[k] for k in
                         ("session", "label", "duration_min", "spw_rate_per_min",
                          "swr_rate_per_min", "p_couple", "n_spw", "n_swr")}
                        for f in train_feats])

# Time-weighted pooled rates: total events / total time (the generator's anchor)
_tot_sec = sum(f["total_sec"]     for f in train_feats)
_tot_spw = sum(f["n_spw"]         for f in train_feats)
_tot_swr = sum(f["n_swr"]         for f in train_feats)
_tot_c   = sum(f["n_coupled_spw"] for f in train_feats)

emp = {
    "spw_rate_per_min": _tot_spw / _tot_sec * 60.0,
    "swr_rate_per_min": _tot_swr / _tot_sec * 60.0,
    "p_couple":         _tot_c / _tot_spw,
    "swr_durs":         _cat([f["swr_durs"]      for f in train_feats]),
    "swr_freqs":        _cat([f["swr_freqs"]     for f in train_feats]),
    "swr_cycles":       _cat([f["swr_cycles"]    for f in train_feats]),
    "coupling_lags":    _cat([f["coupling_lags"] for f in train_feats]),
    "spw_amps":         _cat([f["spw_amps"]      for f in train_feats]),
    "swr_pows":         _cat([f["swr_pows"]      for f in train_feats]),
}

print(f"\n  Empirical reference: {len(sess_df)} training sessions, "
      f"{_tot_sec/60:.1f} min POST NREM+Int.")
print(f"  Time-weighted: SPW {emp['spw_rate_per_min']:.2f}/min  "
      f"SWR {emp['swr_rate_per_min']:.2f}/min  P(couple) {emp['p_couple']:.4f}")
print(f"  Session means: SPW {sess_df['spw_rate_per_min'].mean():.2f}/min  "
      f"SWR {sess_df['swr_rate_per_min'].mean():.2f}/min  "
      f"P(couple) {sess_df['p_couple'].mean():.4f}")
print(f"  SPW amplitudes: n={len(emp['spw_amps'])}")
print(f"  SWR powers: n={len(emp['swr_pows'])}")


# ─────────────────────────────────────────────────────────────────────────────
# [2] Load synthetic libraries
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Loading synthetic libraries...")
for path in [LIB_PKL, CFG_PKL]:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}\nRun the generator first.")

with open(LIB_PKL, "rb") as f:
    lib = pickle.load(f)
with open(CFG_PKL, "rb") as f:
    cfg = pickle.load(f)

_src = sorted(lib.get("source_sessions", []))
if _src != sorted(TRAIN_SESSIONS):
    raise RuntimeError(f"Agent library source_sessions {_src} != training set "
                       f"{sorted(TRAIN_SESSIONS)}. Re-run the generator.")

libraries    = lib["libraries"]
duration_sec = float(lib.get("duration_sec", 3.0))

print(f"  Libraries loaded: {list(libraries.keys())}")
print(f"  Duration per trial: {duration_sec}s")
print(f"  Source sessions: {[lbl(n) for n in _src]}  ✓ training set")

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

# Validation library: the synthetic reference for Sections A, B, D2, ISI and G
if not VAL_PKL.exists():
    raise FileNotFoundError(
        f"Missing: {VAL_PKL}\nRun the generator; it writes the validation library.")
with open(VAL_PKL, "rb") as f:
    val = pickle.load(f)
_vsrc = sorted(val.get("source_sessions", []))
if _vsrc != sorted(TRAIN_SESSIONS):
    raise RuntimeError(f"Validation library source_sessions {_vsrc} != training set. "
                       f"Re-run the generator.")
if abs(float(val.get("duration_sec", duration_sec)) - duration_sec) > 1e-9:
    raise RuntimeError("Validation and agent libraries use different trial lengths.")

null_trials = val["libraries"]["null"][f"{CONTEXT}_ox{ox_key(OX_NULL)}"]
print(f"\n  Validation library: {VAL_PKL.name}  "
      f"({len(null_trials)} trials, seed {val.get('seed', '?')})")

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
print(f"  Synthetic SWR events: {len(null_swr_d)}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — Session-level agreement
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION A: Session-level agreement — training sessions vs synthetic pseudo-sessions")
print("=" * 70)

boot_dfs = {}
for _, row in sess_df.iterrows():
    name = row["session"]
    print(f"  {lbl(name)} ({name}, {row['duration_min']:.1f} min) …", end=" ", flush=True)
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
print(f"  {'Session':<12} {'Duration':>10}  "
      f"{'SPW rate pctile':>17}  {'SWR rate pctile':>17}  {'P(couple) pctile':>17}")
print("  " + "-" * 82)

session_percentiles = {}
for _, row in sess_df.iterrows():
    name  = row["session"]
    bdf   = boot_dfs[name]
    p_spw = float(np.mean(bdf["spw_rate_per_min"] <= row["spw_rate_per_min"])) * 100
    p_swr = float(np.mean(bdf["swr_rate_per_min"] <= row["swr_rate_per_min"])) * 100
    p_c   = float(np.mean(bdf["p_couple"]          <= row["p_couple"]))         * 100
    session_percentiles[name] = {"spw": p_spw, "swr": p_swr, "p_couple": p_c}
    print(f"  {lbl(name):<12} {row['duration_min']:>10.1f}  "
          f"{p_spw:>17.1f}  {p_swr:>17.1f}  {p_c:>17.1f}")

print("\n  Empirical value vs synthetic 95% interval:")
for _, row in sess_df.iterrows():
    name = row["session"]; bdf = boot_dfs[name]
    parts = []
    for col, short_name in [("spw_rate_per_min", "SPW"), ("swr_rate_per_min", "SWR"),
                            ("p_couple", "P(c)")]:
        lo, hi = np.nanpercentile(bdf[col], [2.5, 97.5])
        fmt = ".3f" if col == "p_couple" else ".2f"
        parts.append(f"{short_name} {row[col]:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]")
    print(f"  {lbl(name):<12} " + "   ".join(parts))

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
        [f"{lbl(sn)}\n({sess_df.loc[sess_df['session']==sn,'duration_min'].iloc[0]:.0f} min)"
         for sn in session_names], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}\n{tag}", fontsize=9)

fig.suptitle("A: Session-level agreement  |  Null model @ ox=1.0 (validation library)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
p = FIG_DIR / "A1_session_level_agreement.png"
save_and_show(fig, p)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — Event-level distribution agreement
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION B: Event-level distribution agreement (training vs synthetic)")
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
            label=f"Training (n={len(ec)})")
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

fig.suptitle("B: Event-level distribution agreement  |  Null model @ ox=1.0 (validation library)",
             fontsize=10, fontweight="bold")
plt.tight_layout()
p = FIG_DIR / "B1_event_distributions.png"
save_and_show(fig, p)

# ── Figure B2: SPW ISI, both observed through the same 3 s windows ──
print(f"\n  SPW ISI, observed through non-overlapping {duration_sec:g} s windows:")
emp_isis_w = _cat([windowed_isis(f["spw_times"], f["intervals"], duration_sec)
                   for f in train_feats])
syn_isis_w = null_isis[null_isis < duration_sec] if len(null_isis) else np.array([])

lam_emp = emp["spw_rate_per_min"] / 60.0                                   # 1/s
lam_syn = pooled_rate_from_trials(null_trials, "n_spw", duration_sec) / 60.0

r_isi = distributional_metrics(emp_isis_w, syn_isis_w, "SPW ISI (s), windowed")

# Sanity check: synthetic SPW times are a Poisson process, so their windowed
# ISIs should follow the analytic windowed-Poisson curve.
_ks_syn_ref = stats.kstest(syn_isis_w, lambda x: poisson_window_isi_cdf(x, lam_syn, duration_sec))
_ks_emp_ref = stats.kstest(emp_isis_w, lambda x: poisson_window_isi_cdf(x, lam_emp, duration_sec))
print(f"  Synthetic vs windowed Poisson (λ={lam_syn:.3f}/s): "
      f"KS={_ks_syn_ref.statistic:.3f} (p={_ks_syn_ref.pvalue:.3f})  [sanity check, expect p > 0.05]")
print(f"  Training  vs windowed Poisson (λ={lam_emp:.3f}/s): "
      f"KS={_ks_emp_ref.statistic:.3f} (p={_ks_emp_ref.pvalue:.3g})")
for thr in (0.05, 0.10, 0.25):
    print(f"    fraction of ISIs < {thr*1000:.0f} ms: training {np.mean(emp_isis_w < thr):.3f}, "
          f"synthetic {np.mean(syn_isis_w < thr):.3f}, "
          f"Poisson {poisson_window_isi_cdf(thr, lam_emp, duration_sec):.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.set_facecolor(TAG_COLORS["[REPRODUCED]"] + "44")
bins = np.linspace(0, duration_sec, 31)
ax.hist(emp_isis_w, bins=bins, density=True, alpha=0.70, color="#e05a4b",
        label=f"Training ISIs (n={len(emp_isis_w)})")
ax.hist(syn_isis_w, bins=bins, density=True, alpha=0.55, color="#5b9bd5",
        label=f"Synthetic null@1.0 (n={len(syn_isis_w)})")
xg = np.linspace(0.0, duration_sec, 300, endpoint=False)
ax.plot(xg, poisson_window_isi_pdf(xg, lam_emp, duration_sec), "k--", lw=1.4,
        label=f"Poisson (λ={lam_emp:.2f}/s) in a {duration_sec:g} s window")
if r_isi:
    ks_box = (f"KS={r_isi['ks_stat']:.3f} (p={r_isi['ks_p']:.3f})\n"
              f"EMD={r_isi['emd']:.3f}")
    ax.text(0.97, 0.97, ks_box, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
ax.set_title(f"SPW ISI within {duration_sec:g} s windows   [REPRODUCED]", fontsize=9)
ax.set_xlabel("SPW ISI (s)"); ax.set_ylabel("Density")
ax.legend(fontsize=8)
plt.tight_layout()
p = FIG_DIR / "B2_spw_isi.png"
save_and_show(fig, p)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — Pharmacology alignment (Kostansek et al. 2023)
#             Reads the agent libraries (drug models), not the validation library.
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

print(f"\n  {'Model':<12}" + "".join(f"{n:>10}" for n in metric_names))
for m in drug_models:
    print(f"  {m:<12}" + "".join(f"{v:>10.2f}" for v in mult[m]))
print("  (200-trial agent libraries; Table 1 gives the high-n calibration values)")

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
#             Fitted to the training sessions, so it must reproduce the
#             generator's own fit.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION D: Distribution family comparison — Gamma vs Lognormal")
print("  SPW amplitude: lognormal (committed — literature + empirical)")
print("  Ripple power:  AIC-selected (data-driven, no override)")
print("=" * 70)

amp_comp = fit_and_compare(emp["spw_amps"], "SPW amplitude (detector units)")
pow_comp = fit_and_compare(emp["swr_pows"], "Ripple power") if len(emp["swr_pows"]) > 10 else None

# Consistency with the generator's own fit (same data, same fit)
if dist_comp:
    _g_amp = dist_comp.get("spw_amplitude", {}).get("delta_aic", np.nan)
    _g_pow = dist_comp.get("ripple_power",  {}).get("delta_aic", np.nan)
    print(f"\n  generator ΔAIC: amplitude {_g_amp:+.1f}, power {_g_pow:+.1f}")
    print(f"  recomputed ΔAIC: amplitude {amp_comp['delta_aic']:+.1f}, "
          f"power {pow_comp['delta_aic'] if pow_comp else np.nan:+.1f}")
    _ok = (np.isfinite(_g_amp) and abs(_g_amp - amp_comp['delta_aic']) < 0.5 and
           (pow_comp is None or abs(_g_pow - pow_comp['delta_aic']) < 0.5))
    print(f"  Reproduces the generator fit: {'YES' if _ok else 'NO — check session filter'}")

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
            label=f"Training (n={comp['n']})", edgecolor="white", linewidth=0.3)

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
    ax.set_title(f"{title_label}: Gamma vs Lognormal fit to training data\n"
                 f"[FIT-PRESERVED]",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle(
    "D: Distribution family comparison (training sessions)\n"
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
r_amp = {}
if len(null_spw_a) > 0 and len(emp["spw_amps"]) > 0:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xmin, xmax = np.percentile(
        np.concatenate([emp["spw_amps"], null_spw_a]), [0.5, 99.5])
    bins = np.linspace(xmin, xmax, 50)
    ax.hist(emp["spw_amps"], bins=bins, density=True, alpha=0.70,
            color="#e05a4b", label=f"Training (n={len(emp['spw_amps'])})")
    ax.hist(null_spw_a, bins=bins, density=True, alpha=0.55,
            color="#5b9bd5", label=f"Synthetic null@1.0 (n={len(null_spw_a)})")

    r_amp = distributional_metrics(emp["spw_amps"], null_spw_a, "SPW amplitude")
    if r_amp:
        ks_box = (f"KS={r_amp['ks_stat']:.3f} (p={r_amp['ks_p']:.3f})\n"
                  f"EMD={r_amp['emd']:.1f}  Δmean={r_amp['mean_syn']-r_amp['mean_emp']:+.1f}")
        ax.text(0.97, 0.97, ks_box, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))

    ax.set_title("SPW amplitude: Training vs Synthetic (lognormal-committed)\n"
                 "[FIT-PRESERVED]", fontsize=9)
    ax.set_xlabel("SPW amplitude (detector units)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = FIG_DIR / "D2_spw_amplitude_empirical_vs_synthetic.png"
    save_and_show(fig, p)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — Parameter stability: leave-one-session-out (training sessions)
#             Shows the lognormal fit is not dominated by any single session
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION E: Parameter stability — leave-one-session-out (training sessions)")
print("=" * 70)

_spw_amps_all  = [f["spw_amps"] for f in train_feats]
_swr_pows_all  = [f["swr_pows"] for f in train_feats]
n_sess         = len(_spw_amps_all)
sess_names_raw = [f["label"] for f in train_feats]

# Full fit (all training sessions)
all_amps_pooled = np.concatenate(_spw_amps_all)
all_amps_pos    = all_amps_pooled[all_amps_pooled > 0]
full_ln_params  = lognorm_dist.fit(all_amps_pos, floc=0)
full_s, _, full_scale = full_ln_params

print(f"\n  Full fit (all {n_sess} training sessions, n={len(all_amps_pos)}):")
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
#             Reads the EMPA agent library (the one the agent uses).
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION F: Modulated-metrics validation")
print("  After OX2R modulation, do SWR features stay physiologically plausible?")
print("=" * 70)

# Physiological bounds (hc-11, Grosmark & Buzsáki 2016; Kostansek et al. 2023)
PHYS_BOUNDS = {
    "swr_dur_ms":  (10, 200),    # ripple duration
    "swr_freq_hz": (80, 300),    # ripple frequency (broad, covers paper + in-vivo bands)
    "swr_cycles":  (3, 25),      # cycle count
    "coup_lag_ms": (-30, 80),    # coupling lag
}

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
print(f"  Events within physiological bounds: {all_n - total_violations}/{all_n} ({pct_clean:.1f}%)")
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — Held-out session vs synthetic reference
#             The held-out session contributed nothing to the library. Sample 5
#             is excluded on detection yield and is not tested here.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION G: Held-out session vs synthetic reference")
print("=" * 70)

heldout_summary = []
for f in heldout_feats:
    print(f"\n  {f['label']} ({f['session']}, {f['duration_min']:.1f} min): "
          f"{f['n_spw']} SPW, {f['n_swr']} SWR-tier")
    bdf = bootstrap_synthetic_sessions(null_trials, f["duration_min"],
                                       duration_sec=duration_sec, n_boot=N_BOOT, rng_seed=42)
    rates = {}
    print(f"  {'Rate':<14}{'held out':>10}{'training':>10}{'synthetic 95% interval':>26}{'pctile':>9}")
    for col, name in [("spw_rate_per_min", "SPW /min"), ("swr_rate_per_min", "SWR /min"),
                      ("p_couple", "P(SPW→SWR)")]:
        lo, hi = np.nanpercentile(bdf[col], [2.5, 97.5])
        pct    = float(np.mean(bdf[col] <= f[col])) * 100
        fmt    = ".3f" if col == "p_couple" else ".2f"
        rates[col] = {"value": f[col], "lo": lo, "hi": hi, "pct": pct}
        print(f"  {name:<14}{f[col]:>10{fmt}}{emp[col]:>10{fmt}}"
              f"{'[' + format(lo, fmt) + ', ' + format(hi, fmt) + ']':>26}{pct:>9.1f}")

    print("\n  Event level, held out vs synthetic:")
    ev = {}
    for label, key, _, _, _ in event_metrics:
        r = distributional_metrics(f[key], syn_event_arrays[key], label)
        if r:
            r["mean_train"] = _mean_or_nan(emp[key])
            ev[key] = r
    print("\n  Means (held out | training | synthetic):")
    for label, key, _, _, _ in event_metrics:
        if key in ev:
            print(f"    {label:<22} {ev[key]['mean_emp']:>8.2f} | "
                  f"{ev[key]['mean_train']:>8.2f} | {ev[key]['mean_syn']:>8.2f}")
    heldout_summary.append({"session": f["session"], "label": f["label"],
                            "rates": rates, "events": ev})


# ─────────────────────────────────────────────────────────────────────────────
# Final console summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY (training-session reference)")
print("=" * 70)
print(f"  Empirical reference: {[lbl(n) for n in TRAIN_SESSIONS]}  (time-weighted rates)")
print(f"  Synthetic reference: {VAL_PKL.name}, {len(null_trials)} trials "
      f"({len(null_trials) * duration_sec / 60:.0f} min)")

null_spw_rate = pooled_rate_from_trials(null_trials, "n_spw",    duration_sec)
null_swr_rate = pooled_rate_from_trials(null_trials, "n_coupled",duration_sec)
null_pcoup    = pooled_p_couple(null_trials)

def _dev(a, b):
    return f"{(b - a) / a * 100:+.1f}%" if a else "n/a"

print("\n  ── (i) Model-construction constraints ──")
print(f"  {'Metric':<35} {'Training':<16} {'Synthetic':<16} {'Dev.':<9} {'Tag'}")
print("  " + "-" * 88)
print(f"  {'SPW rate (/min)':<35} {emp['spw_rate_per_min']:<16.2f} {null_spw_rate:<16.2f} "
      f"{_dev(emp['spw_rate_per_min'], null_spw_rate):<9} [CONSTRAINED]")
print(f"  {'P(couple)':<35} {emp['p_couple']:<16.4f} {null_pcoup:<16.4f} "
      f"{_dev(emp['p_couple'], null_pcoup):<9} [CONSTRAINED]")

print("\n  ── (ii) Fit-preserved quantities ──")
print(f"  {'Metric':<35} {'Training':<16} {'Synthetic':<16} {'Tag'}")
print("  " + "-" * 80)
print(f"  {'SWR rate (/min)':<35} {emp['swr_rate_per_min']:<16.2f} {null_swr_rate:<16.2f} "
      f"[FIT-PRESERVED]  ({_dev(emp['swr_rate_per_min'], null_swr_rate)})")
print(f"  {'SWR frequency (Hz)':<35} {_mean_or_nan(emp['swr_freqs']):<16.2f} {_mean_or_nan(null_swr_f):<16.2f} [FIT-PRESERVED]")
print(f"  {'SWR cycles':<35} {_mean_or_nan(emp['swr_cycles']):<16.2f} {_mean_or_nan(null_swr_c):<16.2f} [FIT-PRESERVED]")
print(f"  {'Coupling lag (ms)':<35} {_mean_or_nan(emp['coupling_lags']):<16.2f} {_mean_or_nan(null_lags):<16.2f} [FIT-PRESERVED]")
print(f"  {'SPW amplitude (lognorm)':<35} {_mean_or_nan(emp['spw_amps']):<16.1f} {_mean_or_nan(null_spw_a):<16.1f} [FIT-PRESERVED]")
if r_amp:
    print(f"  {'  amplitude KS / EMD':<35} KS={r_amp['ks_stat']:.3f} (p={r_amp['ks_p']:.3g})  "
          f"EMD={r_amp['emd']:.1f}  Δmean={r_amp['mean_syn'] - r_amp['mean_emp']:+.1f}")

print("\n  ── (iii) Genuine validation outcomes ──")
print(f"  {'Metric':<35} {'Training':<16} {'Synthetic':<16} {'Tag'}")
print("  " + "-" * 80)
print(f"  {'SWR duration (ms)':<35} {_mean_or_nan(emp['swr_durs']):<16.2f} {_mean_or_nan(null_swr_d):<16.2f} [REPRODUCED]")
if r_isi:
    print(f"  {'SPW ISI, ' + format(duration_sec, 'g') + ' s windows':<35} "
          f"KS={r_isi['ks_stat']:.3f} (p={r_isi['ks_p']:.3g})  EMD={r_isi['emd']:.3f}  "
          f"n={r_isi['n_emp']}/{r_isi['n_syn']}   [REPRODUCED]")

print("\n  ── (iv) Distribution family (Section D) ──")
print(f"  {'SPW amplitude':<35} {'lognormal':<16} ΔAIC={amp_comp['delta_aic']:+.1f}  [committed — literature]")
print(f"  {'':<35} Gamma KS={amp_comp['g_ks']:.4f}, lognormal KS={amp_comp['ln_ks']:.4f}")
if pow_comp:
    pow_active = pow_comp['winner']
    print(f"  {'Ripple power':<35} {pow_active:<16} ΔAIC={pow_comp['delta_aic']:+.1f}  [AIC-selected]")
    print(f"  {'':<35} Gamma KS={pow_comp['g_ks']:.4f}, lognormal KS={pow_comp['ln_ks']:.4f}")
print(f"  {'SPW/ripple duration':<35} {'gamma':<16} (waiting-time process)")
print(f"  {'ISIs':<35} {'gamma':<16} (inter-event intervals)")

min_cyc = float(np.min(null_swr_c)) if len(null_swr_c) else np.nan
viol    = int(np.sum(null_swr_c < 3)) if len(null_swr_c) else 0
print(f"\n  ≥3 cycles constraint: Min={min_cyc:.2f}, Violations={viol}/{len(null_swr_c)}")

print(f"\n  ── (v) Parameter stability — LOO over training sessions (Section E) ──")
print(f"  Max |Δs|:     {max_ds:.2f}%")
print(f"  Max |Δscale|: {max_dscl:.2f}%")
print(f"  Stable (<10%): {'PASS' if stable else 'CHECK'}")

print(f"\n  ── (vi) Modulated-metrics validation (Section F, EMPA agent library) ──")
print(f"  Events within physiological bounds: {all_n - total_violations}/{all_n} ({pct_clean:.1f}%)")
print(f"  Modulation preserves plausibility: {'PASS' if pct_clean > 95 else 'CHECK'}")

print(f"\n  ── (vii) Held-out session (Section G) ──")
for h in heldout_summary:
    rr = h["rates"]
    print(f"  {h['label']}: SPW rate pctile {rr['spw_rate_per_min']['pct']:.1f}, "
          f"SWR rate pctile {rr['swr_rate_per_min']['pct']:.1f}, "
          f"P(couple) pctile {rr['p_couple']['pct']:.1f}")
    for key, r in h["events"].items():
        print(f"    {r['label']:<22} KS={r['ks_stat']:.3f} (p={r['ks_p']:.3g})  "
              f"mean {r['mean_emp']:.2f} vs synthetic {r['mean_syn']:.2f}")

print(f"\n  Figures saved to: {FIG_DIR}")
print("=" * 70)
print("Ready for the agent model")
print("=" * 70)
