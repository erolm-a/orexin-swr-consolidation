"""
common.utils — small, shared numerical / array helpers used across the pipeline.

Single canonical definitions for utility functions that were historically
copied between the generator, the agent, the figures and the table builder:
``ox_key``, ``bandpass``, ``safe_concat``, ``_safe_np``/``_mean_or_nan`` &
friends, plus the trial-pooling helpers shared by the synthetic generator and
the figure script.
"""

import hashlib
from typing import List, Tuple

import numpy as np
from scipy import signal

from orexin_keys import ox_key  # canonical key formatter


def bandpass(data, fs, lo, hi, order=4):
    """Band-pass filter `data` (axis=0) between lo and hi Hz with a Butterworth SOS."""
    hi = min(hi, fs / 2 - 1)
    if hi <= lo:
        return data
    sos = signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=0)


def stable_seed(base_seed: int, key: str) -> int:
    """Deterministic per-key seed offset: seed + stable hash of ``key``."""
    h = hashlib.md5(key.encode()).hexdigest()
    return base_seed + (int(h[:8], 16) % (2**31))


def safe_concat(arrays, allow_empty=False):
    """Concatenate the non-empty arrays; return a scalar/empty fallback otherwise."""
    valid = [np.atleast_1d(a) for a in arrays if len(a) > 0]
    if valid:
        return np.concatenate(valid)
    elif allow_empty:
        return np.array([])
    else:
        return np.array([1.0])


def sample_trunc_norm(rng, mean, sd, lo, hi, max_tries=1000):
    """Draw a normal sample retrying until it falls inside [lo, hi]."""
    for _ in range(max_tries):
        x = rng.normal(mean, sd)
        if lo <= x <= hi:
            return x
    return np.clip(mean, lo, hi)


def compute_min_duration_ms(freq_hz: float, min_cycles: int = 3) -> float:
    """Minimum event duration in ms consistent with `min_cycles` at `freq_hz`."""
    return (min_cycles / freq_hz) * 1000.0


def compute_isis_within_intervals(event_times: np.ndarray,
                                   intervals: List[Tuple[float, float]]) -> np.ndarray:
    """Within-interval inter-spike intervals (positive differences only)."""
    isis_list = []
    event_times = np.asarray(event_times, dtype=float)
    for (start, end) in intervals:
        mask = (event_times >= start) & (event_times <= end)
        t = event_times[mask]
        if len(t) > 1:
            isis = np.diff(np.sort(t))
            isis = isis[isis > 0]
            if len(isis) > 0:
                isis_list.append(isis)
    return np.concatenate(isis_list) if isis_list else np.array([])


def ramp_to_one(p_low: float, ox: float, ox_low: float = 0.3, ox_high: float = 1.0) -> float:
    """Linear ramp from ``p_low`` at ``ox_low`` to 1.0 at ``ox_high``."""
    t = max(0.0, min(1.0, (ox - ox_low) / (ox_high - ox_low)))
    return p_low + t * (1.0 - p_low)


# --------------------------------------------------------------------------- #
# Array / scalar summarisers (NaN-safe)
# --------------------------------------------------------------------------- #

def _safe_np(a):
    return np.array(a) if a is not None else np.array([])


def _mean_or_nan(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else np.nan


def _median_or_nan(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


def _std_or_nan(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a)) if len(a) > 1 else np.nan


def _cat(lst):
    return np.concatenate(lst) if lst else np.array([])


# --------------------------------------------------------------------------- #
# Trial pooling (synthetic libraries)
# --------------------------------------------------------------------------- #

def coupled_event_arrays_from_trial(trial):
    """Return (durations, frequencies, cycles) of SPW-coupled ripples in one trial."""
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
        j = int(np.searchsorted(rt, t0))
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
    """Concatenate durations/frequencies/cycles across a list of trials."""
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


def pooled_rate_from_trials(trials, count_key, duration_sec, empty=np.nan):
    """Pooled per-minute rate of ``count_key`` over a list of trials."""
    total      = sum(int(t.get(count_key, 0)) for t in trials)
    total_time = len(trials) * float(duration_sec)
    return (total / total_time) * 60.0 if total_time > 0 else empty


def pooled_rate_per_min(trials, count_key, duration_sec):
    """Alias of :func:`pooled_rate_from_trials` returning NaN for no data."""
    return pooled_rate_from_trials(trials, count_key, duration_sec)


def pooled_p_couple(trials):
    """Proportion of SPWs carrying at least one ripple, pooled over trials."""
    tot_spw = sum(int(t.get("n_spw",             0)) for t in trials)
    tot_c   = sum(int(t.get("n_spw_with_ripple", 0)) for t in trials)
    return (tot_c / tot_spw) if tot_spw > 0 else np.nan


__all__ = [
    "ox_key",
    "bandpass",
    "stable_seed",
    "safe_concat",
    "sample_trunc_norm",
    "compute_min_duration_ms",
    "compute_isis_within_intervals",
    "ramp_to_one",
    "_safe_np",
    "_mean_or_nan",
    "_median_or_nan",
    "_std_or_nan",
    "_cat",
    "coupled_event_arrays_from_trial",
    "pooled_coupled_event_arrays",
    "pooled_rate_from_trials",
    "pooled_rate_per_min",
    "pooled_p_couple",
]