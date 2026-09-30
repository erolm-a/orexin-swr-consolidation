"""
Stage 1 - SWR detection from LFP recordings.

Reads:  data_raw/<session>/*.lfp|.eeg, the accompanying XML metadata
        and sleep-state annotations
Writes: data_processed/swr_detection.pkl

Channel selection, spectral purity, SPW and ripple detection, coupling
classification and the gold-standard subsets.

Channel selection is band-power only: SPW channels are the strongest
5-40 Hz channels, ripple channels the strongest 150-250 Hz channels among
those left, noise references the strongest 300-500 Hz channels among those
left again. No probe layout is parsed, so no channel is excluded for being
off-shank; the XML is read only for the LFP channel count and rate.

Paths are resolved by orexin_paths; nothing here is machine-specific.
"""

import sys
from pathlib import Path


def _repo_root() -> Path:
    """Find the repository root whether this runs as a file or pasted in a cell."""
    start = (
        Path(globals()["__file__"]).resolve().parent
        if "__file__" in globals()
        else Path.cwd().resolve()
    )
    for d in (start, *start.parents):
        if (d / "orexin_paths.py").exists():
            return d
    raise ImportError(
        "orexin_paths.py not found above " + str(start) + ".\n"
        "Running from a notebook? cd into the repository first, e.g.\n"
        "    %cd /path/to/orexin-swr-consolidation"
    )


sys.path.insert(0, str(_repo_root()))

from orexin_paths import DIRS, FILES, ensure_dirs, stamp  # noqa: E402
from common.utils import bandpass  # noqa: E402

ensure_dirs()

# =============================================================================
# SWR DETECTION
# =============================================================================


import numpy as np
from scipy import signal, stats
from pathlib import Path
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    import mat73
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "mat73", "-q"])
    import mat73

# Paths come from orexin_paths (see header). DIRS / FILES are already bound.

print("=" * 70)
print("SWR DETECTION")
print("=" * 70)


CONFIG = {
    "hc11_lfp_fs": 1250,
    "chunk_sec": 120,
    "chunk_overlap_sec": 4,
    "n_spw_channels": 8,
    "n_ripple_channels": 8,
    "n_noise_channels": 4,
    "max_kurtosis": 10,
    "min_channel_std": 5,
    "max_channel_std": 5000,
    "spw_band": (5, 40),
    "spw_threshold_mad": 4.0,
    "spw_min_duration_ms": 30,
    "spw_max_duration_ms": 200,
    "spw_min_channels": 2,
    "spw_merge_within_ms": 30,
    "spw_refractory_ms": 50,
    "spw_gold_duration_range_ms": (50, 120),
    "spw_gold_min_amplitude_pct": 50,
    "ripple_band": (150, 250),
    "ripple_threshold_mad": 5.0,
    "ripple_min_duration_ms": 20,
    "ripple_max_duration_ms": 100,
    "ripple_min_cycles": 4,
    "ripple_min_channels": 3,
    "ripple_merge_within_ms": 15,
    "ripple_refractory_ms": 30,
    "ripple_envelope_percentile": 75,
    "ripple_gold_duration_range_ms": (20, 80),
    "ripple_gold_min_power_pct": 50,
    "ripple_swr_purity_min": 2.0,
    "ripple_swr_peak_freq_range": (150, 200),
    "coupling_pre_ms": 20,
    "coupling_post_ms": 50,
    "ripple_all_purity_min": 1.5,
    "ripple_all_peak_freq_range": (140, 220),
    "noise_band": (300, 500),
    "noise_veto_threshold_mad": 4.5,
    "purity_nperseg": 128,
    "purity_noverlap": 64,
    "purity_window_ms": 160,
    "gamma_band": (60, 120),
    "purity_gamma_floor_percentile": 5,
    "target_states": ["NREM", "Intermediate"],
    "stability_bin_sec": 300,
    # Coverage threshold for QA
    "min_ripple_coverage": 0.6,  # Flag if <60% of top-k power
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def parse_xml_full(xml_file):
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(xml_file).getroot()
    except:
        return None
    params = {"n_channels": None, "sampling_rate": None, "lfp_sampling_rate": None}
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        txt = (elem.text or "").strip()
        if tag == "nChannels" and params["n_channels"] is None:
            try:
                params["n_channels"] = int(float(txt))
            except:
                pass
        if tag == "samplingRate" and params["sampling_rate"] is None:
            try:
                params["sampling_rate"] = int(float(txt))
            except:
                pass
        if tag == "lfpSamplingRate" and params["lfp_sampling_rate"] is None:
            try:
                params["lfp_sampling_rate"] = int(float(txt))
            except:
                pass
    return params


def get_lfp_params(lfp_file, xml_files):
    lfp_size = lfp_file.stat().st_size
    n_int16 = lfp_size // 2
    for xml_file in xml_files:
        params = parse_xml_full(xml_file)
        if params is None:
            print("  ⚠  Unable to parse the XML file")
            continue
        n_ch = params["n_channels"]
        if n_ch is None or n_int16 % n_ch != 0:
            print("  ⚠  Ch size is not multiple of n_ch!")
            continue
        if params["lfp_sampling_rate"] is not None:
            n_samples = n_int16 // n_ch
            duration_sec = n_samples / params["lfp_sampling_rate"]
            if duration_sec > 3600:
                return {
                    "xml_file": xml_file,
                    "n_channels": n_ch,
                    "fs": params["lfp_sampling_rate"],
                    "duration_sec": duration_sec,
                    "source": "xml_lfp_rate",
                }
        if n_ch is not None:
            fs = CONFIG["hc11_lfp_fs"]
            n_samples = n_int16 // n_ch
            duration_sec = n_samples / fs
            if duration_sec > 1800:
                return {
                    "xml_file": xml_file,
                    "n_channels": n_ch,
                    "fs": fs,
                    "duration_sec": duration_sec,
                    "source": "hc11_default",
                }
    return None


def mad(x, axis=None):
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis)


def robust_threshold(x, k=4.0):
    return np.median(x) + k * 1.4826 * mad(x)


def normalize_epoch_array(arr):
    if arr is None:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.atleast_1d(np.asarray(arr, dtype=np.float64))
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 1:
        if len(arr) == 2:
            return arr.reshape(1, 2)
        elif len(arr) % 2 == 0:
            return arr.reshape(-1, 2)
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr
        elif arr.shape[0] == 2:
            return arr.T
    flat = arr.flatten()
    if len(flat) >= 2 and len(flat) % 2 == 0:
        return flat.reshape(-1, 2)
    return np.empty((0, 2), dtype=np.float64)


def load_sessinfo(sessinfo_path):
    print(f"  Loading sessInfo: {sessinfo_path.name}")
    try:
        data = mat73.loadmat(str(sessinfo_path))
        if "sessInfo" not in data:
            return None
        si = data["sessInfo"]
        sessinfo = {"epochs": {}}
        if "Epochs" in si:
            ep = si["Epochs"]
            for field in [
                "PREEpoch",
                "MazeEpoch",
                "POSTEpoch",
                "sessDuration",
                "Wake",
                "Drowsy",
                "NREM",
                "Intermediate",
                "REM",
            ]:
                if field in ep:
                    val = ep[field]
                    if field == "sessDuration":
                        sessinfo["total_duration"] = (
                            float(val)
                            if np.isscalar(val)
                            else float(np.asarray(val).flat[0])
                        )
                    else:
                        sessinfo["epochs"][field] = normalize_epoch_array(val)
        return sessinfo
    except Exception as e:
        print(f"    mat73 failed: {e}")
        return None


def get_nrem_intervals_in_post(sessinfo):
    if sessinfo is None or "epochs" not in sessinfo:
        return None
    epochs = sessinfo["epochs"]
    if "POSTEpoch" not in epochs or len(epochs["POSTEpoch"]) == 0:
        return None
    post = epochs["POSTEpoch"]
    post_start, post_end = post[0, 0], post[0, 1]
    print(
        f"    POST: {post_start:.1f} - {post_end:.1f} sec ({(post_end - post_start) / 60:.1f} min)"
    )
    target_intervals = []
    for state in CONFIG["target_states"]:
        if state in epochs and len(epochs[state]) > 0:
            for i in range(len(epochs[state])):
                s, e = epochs[state][i, 0], epochs[state][i, 1]
                if e > post_start and s < post_end:
                    cs, ce = max(s, post_start), min(e, post_end)
                    if ce > cs + 1:
                        target_intervals.append((cs, ce))
    if not target_intervals:
        return None
    target_intervals.sort()
    merged = [list(target_intervals[0])]
    for s, e in target_intervals[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    merged = [(s, e) for s, e in merged]
    total_sec = sum(e - s for s, e in merged)
    print(
        f"    {len(merged)} NREM intervals, {total_sec / 60:.1f} min ({total_sec / (post_end - post_start) * 100:.1f}%)"
    )
    return {
        "intervals": merged,
        "total_sec": total_sec,
        "post_start": post_start,
        "post_end": post_end,
    }


# CHANNEL SELECTION — RAW BAND POWER WITH MUTUAL EXCLUSION
#
# SPW, ripple and noise channels are picked from band power alone: SPW by
# 5-40 Hz power, ripple by 150-250 Hz power among the channels SPW did not
# take, noise by 300-500 Hz power among the remainder. No probe layout is
# consulted, so auxiliary channels are not excluded a priori; a channel that
# is neither quiet nor flat is dropped by the veto, not by the wiring.

import numpy as np
from scipy import signal, stats

CHANNEL_RULE = (
    "raw band power; SPW = top SW power; "
    "ripple = top ripple power excluding SPW; "
    "noise = top noise/ripple ratio excluding both"
)


def select_channels_with_noise(lfp_file, n_ch, fs, intervals, verbose=True):
    """Select SPW, ripple and noise-reference channels from band power."""
    lfp_file = Path(lfp_file)

    # ── channel metrics ─────────────────────────────────────────────────────
    mmap = np.memmap(lfp_file, dtype=np.int16, mode="r").reshape(-1, n_ch)
    n_samples = mmap.shape[0]
    print(f"    LFP: {n_samples / fs:.1f} sec, {n_ch} channels")

    if intervals and len(intervals["intervals"]) > 0:
        s, e = intervals["intervals"][0]
        mid = (s + e) / 2
        sample_start = int(max(s, mid - 150) * fs)
        sample_end = int(min(e, mid + 150) * fs)
    else:
        mid = (n_samples / fs) / 2
        sample_start = int(max(0, mid - 150) * fs)
        sample_end = int(min(n_samples / fs, mid + 150) * fs)

    sample_start = max(0, min(sample_start, n_samples - 1))
    sample_end = max(sample_start + int(fs), min(sample_end, n_samples))

    channel_metrics = []
    for ch in range(n_ch):
        data = mmap[sample_start:sample_end, ch].astype(np.float32)
        std = np.std(data)
        kurt = stats.kurtosis(data)

        vetoed = (
            std < CONFIG["min_channel_std"]
            or std > CONFIG["max_channel_std"]
            or abs(kurt) > CONFIG["max_kurtosis"]
        )
        if vetoed:
            channel_metrics.append(
                {
                    "ch": ch,
                    "vetoed": True,
                    "spw_power": 0,
                    "ripple_power": 0,
                    "noise_power": 0,
                }
            )
            continue

        try:
            spw_power = np.var(bandpass(data, fs, *CONFIG["spw_band"]))
        except Exception:
            spw_power = 0
        try:
            ripple_power = np.var(bandpass(data, fs, *CONFIG["ripple_band"]))
        except Exception:
            ripple_power = 0
        try:
            noise_power = np.var(bandpass(data, fs, *CONFIG["noise_band"]))
        except Exception:
            noise_power = 0

        channel_metrics.append(
            {
                "ch": ch,
                "vetoed": False,
                "spw_power": spw_power,
                "ripple_power": ripple_power,
                "noise_power": noise_power,
            }
        )
    del mmap

    valid = [m for m in channel_metrics if not m["vetoed"]]
    print(f"    Valid: {len(valid)}/{n_ch} (vetoed {n_ch - len(valid)})")
    if len(valid) < 4:
        return None

    # ── 1. SPW channels: top-n by raw SW-band power ─────────────────────────
    n_spw = min(CONFIG["n_spw_channels"], len(valid))
    spw_sorted = sorted(valid, key=lambda m: m["spw_power"], reverse=True)
    spw_channels = sorted(m["ch"] for m in spw_sorted[:n_spw])
    spw_set = set(spw_channels)

    # ── 2. ripple channels: top-n by raw ripple power, SPW already taken ────
    rip_sorted = sorted(
        (m for m in valid if m["ch"] not in spw_set),
        key=lambda m: m["ripple_power"],
        reverse=True,
    )
    n_rip = min(CONFIG["n_ripple_channels"], len(rip_sorted))
    ripple_channels = sorted(m["ch"] for m in rip_sorted[:n_rip])
    rip_set = set(ripple_channels)

    # Coverage: how much of the top-k ripple power the selection actually took
    all_rip = sorted((m["ripple_power"] for m in valid), reverse=True)
    selected_power = sum(m["ripple_power"] for m in rip_sorted[:n_rip])
    topk_power = sum(all_rip[:n_rip])
    coverage = selected_power / (topk_power + 1e-10)

    if verbose:
        print(f"    --- Channel Selection ({CHANNEL_RULE.split(';')[0]}) ---")
        print(f"    SPW:    {spw_channels}")
        print(
            f"            SW-band power "
            f"{min(m['spw_power'] for m in spw_sorted[:n_spw]):.3g}–"
            f"{max(m['spw_power'] for m in spw_sorted[:n_spw]):.3g}"
        )
        print(f"    Ripple: {ripple_channels}")
        print(
            f"            top {n_rip} by raw ripple power were "
            f"{sorted(m['ch'] for m in rip_sorted[:n_rip])}; coverage "
            f"{coverage * 100:.1f}%"
        )
        if coverage < CONFIG["min_ripple_coverage"]:
            print(
                f"            ⚠ coverage {coverage * 100:.1f}% < "
                f"{CONFIG['min_ripple_coverage'] * 100:.0f}% threshold"
            )

    if len(spw_channels) < CONFIG["spw_min_channels"]:
        print(f"    ⚠ Only {len(spw_channels)} SPW candidates; session skipped.")
        return None

    # ── 3. noise-reference channels ─────────────────────────────────────────
    exclude_set = spw_set | rip_set
    noise_candidates = [m for m in valid if m["ch"] not in exclude_set]
    eps = 1e-10
    for m in noise_candidates:
        m["noise_score"] = m["noise_power"] / (m["ripple_power"] + eps)
    noise_candidates.sort(key=lambda m: m["noise_score"], reverse=True)
    n_noise = min(CONFIG["n_noise_channels"], len(noise_candidates))
    noise_channels = [m["ch"] for m in noise_candidates[:n_noise]]

    return {
        "spw_channels": spw_channels,
        "ripple_channels": ripple_channels,
        "noise_channels": sorted(noise_channels),
        "n_overlap": len(spw_set & rip_set),
        "n_valid": len(valid),
        "fs": fs,
        "n_channels_total": n_ch,
        "ripple_coverage": coverage,
        "rule": CHANNEL_RULE,
        "all_channel_metrics": valid,
    }


# =============================================================================
# SPECTRAL PURITY
# =============================================================================


def compute_spectral_purity_welch(
    data, fs, ripple_band, gamma_band, nperseg=128, noverlap=64, gamma_floor=None
):
    data = np.asarray(data)
    if data.shape[0] < 32:
        return np.nan, np.nan, False

    nperseg = min(nperseg, data.shape[0])
    noverlap = min(noverlap, nperseg // 2)

    if data.ndim == 2:
        freqs, psd = signal.welch(
            data, fs=fs, axis=0, nperseg=nperseg, noverlap=noverlap
        )
        psd = np.mean(psd, axis=1)
    else:
        freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

    ripple_mask = (freqs >= ripple_band[0]) & (freqs <= ripple_band[1])
    gamma_mask = (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])

    ripple_power = (
        np.trapezoid(psd[ripple_mask], freqs[ripple_mask])
        if np.any(ripple_mask)
        else 0.0
    )
    gamma_power = (
        np.trapezoid(psd[gamma_mask], freqs[gamma_mask])
        if np.any(gamma_mask)
        else np.nan
    )

    if gamma_floor is not None and np.isfinite(gamma_power):
        gamma_power = max(gamma_power, gamma_floor)

    purity = (
        ripple_power / (gamma_power + 1e-10)
        if np.isfinite(gamma_power) and gamma_power > 0
        else np.nan
    )

    if np.any(ripple_mask):
        idx = np.argmax(psd[ripple_mask])
        peak_freq = freqs[ripple_mask][idx]
        return float(purity), float(peak_freq), True

    return float(purity), np.nan, False


# =============================================================================
# EVENT MERGING
# =============================================================================


def merge_events(events, merge_within_ms, refractory_ms, amplitude_key="amplitude"):
    if len(events) == 0:
        return events
    merge_sec = merge_within_ms / 1000
    refractory_sec = refractory_ms / 1000
    events = sorted(events, key=lambda x: x["start_sec"])
    result = []
    current = events[0].copy()
    for evt in events[1:]:
        gap = evt["start_sec"] - current["end_sec"]
        if gap < merge_sec:
            current["end_sec"] = max(current["end_sec"], evt["end_sec"])
            current["duration_ms"] = (current["end_sec"] - current["start_sec"]) * 1000
            if evt.get(amplitude_key, 0) > current.get(amplitude_key, 0):
                current["peak_sec"] = evt["peak_sec"]
                current[amplitude_key] = evt[amplitude_key]
                for key in [
                    "frequency",
                    "n_cycles",
                    "spectral_purity",
                    "peak_freq",
                    "n_channels",
                ]:
                    if key in evt:
                        current[key] = evt[key]
        elif gap < refractory_sec:
            if evt.get(amplitude_key, 0) > current.get(amplitude_key, 0):
                current = evt.copy()
        else:
            result.append(current)
            current = evt.copy()
    result.append(current)
    return result


# =============================================================================
# SPW DETECTION
# =============================================================================


def detect_spw(lfp_file, n_ch, fs, channels, intervals):
    mmap = np.memmap(lfp_file, dtype=np.int16, mode="r").reshape(-1, n_ch)
    chunk_samp = int(CONFIG["chunk_sec"] * fs)
    overlap_samp = int(CONFIG["chunk_overlap_sec"] * fs)
    min_samp = int(CONFIG["spw_min_duration_ms"] * fs / 1000)
    max_samp = int(CONFIG["spw_max_duration_ms"] * fs / 1000)
    all_events = []

    for int_start, int_end in intervals["intervals"]:
        int_start_samp = max(0, int(int_start * fs))
        int_end_samp = min(mmap.shape[0], int(int_end * fs))
        if int_end_samp <= int_start_samp:
            continue
        chunk_starts = list(
            range(int_start_samp, int_end_samp, chunk_samp - overlap_samp)
        )
        for chunk_idx, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + chunk_samp, int_end_samp)
            is_first = chunk_idx == 0
            is_last = chunk_end >= int_end_samp
            core_start = 0 if is_first else overlap_samp
            core_end = (
                (chunk_end - chunk_start)
                if is_last
                else (chunk_end - chunk_start - overlap_samp)
            )
            data = mmap[chunk_start:chunk_end, channels].astype(np.float32)
            if len(data) < min_samp * 3:
                continue
            filtered = bandpass(data, fs, CONFIG["spw_band"][0], CONFIG["spw_band"][1])
            envelopes = np.abs(signal.hilbert(filtered, axis=0))
            mean_env = np.mean(envelopes, axis=1)
            threshold = robust_threshold(mean_env, k=CONFIG["spw_threshold_mad"])
            ch_thresholds = np.array(
                [
                    robust_threshold(envelopes[:, i], k=CONFIG["spw_threshold_mad"])
                    for i in range(len(channels))
                ]
            )
            above = mean_env > threshold
            diff = np.diff(above.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            if len(starts) == 0 or len(ends) == 0:
                continue
            if above[0]:
                starts = np.concatenate([[0], starts])
            if above[-1]:
                ends = np.concatenate([ends, [len(mean_env)]])
            for i in range(min(len(starts), len(ends))):
                s, e = starts[i], ends[i]
                dur = e - s
                if not (min_samp <= dur <= max_samp):
                    continue
                if s < core_start or e > core_end:
                    continue
                event_window = envelopes[s:e, :]
                n_ch_above = np.sum(np.any(event_window > ch_thresholds, axis=0))
                if n_ch_above < CONFIG["spw_min_channels"]:
                    continue
                peak_idx = s + np.argmax(mean_env[s:e])
                all_events.append(
                    {
                        "start_sec": (chunk_start + s) / fs,
                        "end_sec": (chunk_start + e) / fs,
                        "peak_sec": (chunk_start + peak_idx) / fs,
                        "duration_ms": dur / fs * 1000,
                        "amplitude": mean_env[peak_idx],
                        "n_channels": n_ch_above,
                    }
                )
    del mmap
    return merge_events(
        all_events,
        CONFIG["spw_merge_within_ms"],
        CONFIG["spw_refractory_ms"],
        "amplitude",
    )


# =============================================================================
# RIPPLE DETECTION
# =============================================================================


def detect_ripples_with_noise_veto(
    lfp_file, n_ch, fs, ripple_channels, noise_channels, intervals
):
    mmap = np.memmap(lfp_file, dtype=np.int16, mode="r").reshape(-1, n_ch)
    chunk_samp = int(CONFIG["chunk_sec"] * fs)
    overlap_samp = int(CONFIG["chunk_overlap_sec"] * fs)
    min_samp = int(CONFIG["ripple_min_duration_ms"] * fs / 1000)
    max_samp = int(CONFIG["ripple_max_duration_ms"] * fs / 1000)
    psd_half_win = int((CONFIG["purity_window_ms"] / 2) * fs / 1000)
    envelope_pct = CONFIG["ripple_envelope_percentile"]

    all_events = []
    n_noise_rejected = 0
    gamma_powers_all = []

    # First pass: estimate gamma floor
    for int_start, int_end in intervals["intervals"]:
        int_start_samp = max(0, int(int_start * fs))
        int_end_samp = min(mmap.shape[0], int(int_end * fs))
        if int_end_samp - int_start_samp < chunk_samp:
            continue
        sample_starts = [int_start_samp, (int_start_samp + int_end_samp) // 2]
        for chunk_start in sample_starts[:2]:
            chunk_end = min(chunk_start + chunk_samp, int_end_samp)
            data = mmap[chunk_start:chunk_end, ripple_channels].astype(np.float32)
            if len(data) < 1000:
                continue
            for win_start in range(0, len(data) - 256, 256):
                win_data = data[win_start : win_start + 256]
                freqs, psd = signal.welch(
                    win_data, fs=fs, axis=0, nperseg=128, noverlap=64
                )
                psd = np.mean(psd, axis=1)
                gamma_mask = (freqs >= CONFIG["gamma_band"][0]) & (
                    freqs <= CONFIG["gamma_band"][1]
                )
                if np.any(gamma_mask):
                    gamma_pwr = np.trapezoid(psd[gamma_mask], freqs[gamma_mask])
                    if np.isfinite(gamma_pwr) and gamma_pwr > 0:
                        gamma_powers_all.append(gamma_pwr)

    gamma_floor = (
        np.percentile(gamma_powers_all, CONFIG["purity_gamma_floor_percentile"])
        if gamma_powers_all
        else None
    )

    # Main detection
    for int_start, int_end in intervals["intervals"]:
        int_start_samp = max(0, int(int_start * fs))
        int_end_samp = min(mmap.shape[0], int(int_end * fs))
        if int_end_samp <= int_start_samp:
            continue
        chunk_starts = list(
            range(int_start_samp, int_end_samp, chunk_samp - overlap_samp)
        )

        for chunk_idx, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + chunk_samp, int_end_samp)
            is_first = chunk_idx == 0
            is_last = chunk_end >= int_end_samp
            core_start = 0 if is_first else overlap_samp
            core_end = (
                (chunk_end - chunk_start)
                if is_last
                else (chunk_end - chunk_start - overlap_samp)
            )

            data_rip = mmap[chunk_start:chunk_end, ripple_channels].astype(np.float32)

            if noise_channels:
                data_noise = mmap[chunk_start:chunk_end, noise_channels].astype(
                    np.float32
                )
            else:
                data_noise = None

            if len(data_rip) < min_samp * 3:
                continue

            filt_rip = bandpass(
                data_rip, fs, CONFIG["ripple_band"][0], CONFIG["ripple_band"][1]
            )
            env_rip = np.abs(signal.hilbert(filt_rip, axis=0))
            agg_env = np.percentile(env_rip, envelope_pct, axis=1)

            if data_noise is not None:
                filt_noise = bandpass(
                    data_noise, fs, CONFIG["noise_band"][0], CONFIG["noise_band"][1]
                )
                env_noise = np.abs(signal.hilbert(filt_noise, axis=0))
                mean_env_noise = np.mean(env_noise, axis=1)
                noise_threshold = robust_threshold(
                    mean_env_noise, k=CONFIG["noise_veto_threshold_mad"]
                )
            else:
                mean_env_noise = None
                noise_threshold = None

            threshold = robust_threshold(agg_env, k=CONFIG["ripple_threshold_mad"])
            ch_thresholds = np.array(
                [
                    robust_threshold(env_rip[:, i], k=CONFIG["ripple_threshold_mad"])
                    for i in range(len(ripple_channels))
                ]
            )

            above = agg_env > threshold
            diff = np.diff(above.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1

            if len(starts) == 0 or len(ends) == 0:
                continue
            if above[0]:
                starts = np.concatenate([[0], starts])
            if above[-1]:
                ends = np.concatenate([ends, [len(agg_env)]])

            for i in range(min(len(starts), len(ends))):
                s, e = starts[i], ends[i]
                dur = e - s
                if not (min_samp <= dur <= max_samp):
                    continue
                if s < core_start or e > core_end:
                    continue

                event_window = env_rip[s:e, :]
                n_channels_above = np.sum(np.any(event_window > ch_thresholds, axis=0))
                if n_channels_above < CONFIG["ripple_min_channels"]:
                    continue

                segment = np.mean(filt_rip[s:e], axis=1)
                zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)
                n_cycles = zero_crossings / 2
                if n_cycles < CONFIG["ripple_min_cycles"]:
                    continue

                peak_idx = s + np.argmax(agg_env[s:e])

                if mean_env_noise is not None and noise_threshold is not None:
                    noise_at_event = np.max(mean_env_noise[s:e])
                    if noise_at_event > noise_threshold:
                        n_noise_rejected += 1
                        continue

                psd_start = max(0, peak_idx - psd_half_win)
                psd_end = min(data_rip.shape[0], peak_idx + psd_half_win)
                purity, peak_freq, _ = compute_spectral_purity_welch(
                    data_rip[psd_start:psd_end],
                    fs,
                    CONFIG["ripple_band"],
                    CONFIG["gamma_band"],
                    CONFIG["purity_nperseg"],
                    CONFIG["purity_noverlap"],
                    gamma_floor=gamma_floor,
                )

                all_events.append(
                    {
                        "start_sec": (chunk_start + s) / fs,
                        "end_sec": (chunk_start + e) / fs,
                        "peak_sec": (chunk_start + peak_idx) / fs,
                        "duration_ms": dur / fs * 1000,
                        "n_channels": n_channels_above,
                        "power": np.mean(agg_env[s:e]),
                        "n_cycles": n_cycles,
                        "frequency": n_cycles / (dur / fs) if dur > 0 else 0,
                        "spectral_purity": purity,
                        "peak_freq": peak_freq,
                    }
                )

    del mmap
    print(f"    Noise veto rejected: {n_noise_rejected}")
    if gamma_floor is not None:
        print(f"    Gamma floor: {gamma_floor:.2e}")

    return merge_events(
        all_events,
        CONFIG["ripple_merge_within_ms"],
        CONFIG["ripple_refractory_ms"],
        "power",
    )


# =============================================================================
# COUPLING & CLASSIFICATION
# =============================================================================


def compute_coupling_spw_to_ripple(spw_events, ripple_events):
    if len(spw_events) == 0 or len(ripple_events) == 0:
        return {
            "coupling_rate": 0,
            "n_coupled_spw": 0,
            "n_spw": len(spw_events),
            "coupled_spw_mask": np.zeros(len(spw_events), dtype=bool),
            "coupling_lags_ms": np.array([]),
            "coupled_ripple_indices": [],
        }

    pre_sec = CONFIG["coupling_pre_ms"] / 1000
    post_sec = CONFIG["coupling_post_ms"] / 1000

    spw_peaks = np.array([e["peak_sec"] for e in spw_events])
    ripple_peaks = np.array([e["peak_sec"] for e in ripple_events])

    coupled_spw_mask = np.zeros(len(spw_events), dtype=bool)
    coupling_lags = []
    coupled_ripple_indices = []

    for i, spw_peak in enumerate(spw_peaks):
        ripple_lags = ripple_peaks - spw_peak
        valid = (ripple_lags >= -pre_sec) & (ripple_lags <= post_sec)

        if np.any(valid):
            coupled_spw_mask[i] = True
            valid_lags = ripple_lags[valid]
            closest_idx = np.where(valid)[0][np.argmin(np.abs(valid_lags))]
            coupling_lags.append(ripple_lags[closest_idx] * 1000)
            coupled_ripple_indices.append(closest_idx)

    return {
        "coupling_rate": np.mean(coupled_spw_mask),
        "n_coupled_spw": np.sum(coupled_spw_mask),
        "n_spw": len(spw_events),
        "coupled_spw_mask": coupled_spw_mask,
        "coupling_lags_ms": np.array(coupling_lags),
        "coupled_ripple_indices": coupled_ripple_indices,
    }


def classify_ripples(ripple_events, coupling_info):
    if len(ripple_events) == 0:
        return {"ripples_all": [], "ripples_swr": [], "n_all": 0, "n_swr": 0}

    coupled_set = set(coupling_info.get("coupled_ripple_indices", []))
    ripples_all = []
    ripples_swr = []

    for i, evt in enumerate(ripple_events):
        purity = evt.get("spectral_purity", np.nan)
        peak_freq = evt.get("peak_freq", np.nan)

        passes_all = (
            np.isfinite(purity)
            and purity >= CONFIG["ripple_all_purity_min"]
            and np.isfinite(peak_freq)
            and CONFIG["ripple_all_peak_freq_range"][0]
            <= peak_freq
            <= CONFIG["ripple_all_peak_freq_range"][1]
        )

        if passes_all:
            ripples_all.append(evt)

        passes_swr = (
            (i in coupled_set)
            and np.isfinite(purity)
            and purity >= CONFIG["ripple_swr_purity_min"]
            and np.isfinite(peak_freq)
            and CONFIG["ripple_swr_peak_freq_range"][0]
            <= peak_freq
            <= CONFIG["ripple_swr_peak_freq_range"][1]
        )

        if passes_swr:
            ripples_swr.append(evt)

    return {
        "ripples_all": ripples_all,
        "ripples_swr": ripples_swr,
        "n_all": len(ripples_all),
        "n_swr": len(ripples_swr),
    }


# =============================================================================
# GOLD SETS & QUALITY
# =============================================================================


def extract_gold_sets(spw_events, ripple_tiers):
    spw_gold = []
    if spw_events:
        amps = np.array([e["amplitude"] for e in spw_events])
        amp_thresh = np.percentile(amps, 100 - CONFIG["spw_gold_min_amplitude_pct"])
        for evt in spw_events:
            if (
                evt["duration_ms"] >= CONFIG["spw_gold_duration_range_ms"][0]
                and evt["duration_ms"] <= CONFIG["spw_gold_duration_range_ms"][1]
                and evt["amplitude"] >= amp_thresh
                and evt["n_channels"] >= 2
            ):
                spw_gold.append(evt)

    ripple_gold = []
    swr = ripple_tiers["ripples_swr"]
    if swr:
        powers = np.array([e["power"] for e in swr])
        pwr_thresh = (
            np.percentile(powers, 100 - CONFIG["ripple_gold_min_power_pct"])
            if len(powers) > 0
            else 0
        )
        for evt in swr:
            pf = evt.get("peak_freq", np.nan)
            pur = evt.get("spectral_purity", np.nan)
            if (
                evt["duration_ms"] >= CONFIG["ripple_gold_duration_range_ms"][0]
                and evt["duration_ms"] <= CONFIG["ripple_gold_duration_range_ms"][1]
                and evt["power"] >= pwr_thresh
                and np.isfinite(pf)
                and CONFIG["ripple_swr_peak_freq_range"][0]
                <= pf
                <= CONFIG["ripple_swr_peak_freq_range"][1]
                and np.isfinite(pur)
                and pur >= CONFIG["ripple_swr_purity_min"]
                and evt["n_channels"] >= 2
            ):
                ripple_gold.append(evt)

    return {
        "spw_gold": spw_gold,
        "ripple_gold": ripple_gold,
        "n_spw_gold": len(spw_gold),
        "n_ripple_gold": len(ripple_gold),
    }


def compute_quality(
    spw_events, ripple_tiers, gold_sets, coupling_raw, coupling_swr, intervals, ch_info
):
    nrem_min = intervals["total_sec"] / 60
    metrics = {"nrem_duration_min": nrem_min}
    score = 0

    spw_rate = len(spw_events) / nrem_min if nrem_min > 0 else 0
    metrics["spw_rate"] = spw_rate

    swr_rate = ripple_tiers["n_swr"] / nrem_min if nrem_min > 0 else 0
    metrics["swr_rate"] = swr_rate

    metrics["ripple_all_rate"] = ripple_tiers["n_all"] / nrem_min if nrem_min > 0 else 0

    metrics["coupling_raw"] = coupling_raw["coupling_rate"]
    metrics["coupling_swr"] = coupling_swr["coupling_rate"]

    # Coverage metric
    metrics["ripple_coverage"] = ch_info.get("ripple_coverage", 0)

    if coupling_swr["coupling_rate"] >= 0.15:
        score += 1
    elif coupling_swr["coupling_rate"] >= 0.05:
        score += 0.5

    swr = ripple_tiers["ripples_swr"]
    if swr:
        purities = [
            e["spectral_purity"]
            for e in swr
            if np.isfinite(e.get("spectral_purity", np.nan))
        ]
        if purities:
            metrics["median_purity_swr"] = np.median(purities)
            if metrics["median_purity_swr"] >= 2.5:
                score += 1
            elif metrics["median_purity_swr"] >= 2.0:
                score += 0.5
        else:
            metrics["median_purity_swr"] = np.nan
        peak_freqs = [
            e["peak_freq"] for e in swr if np.isfinite(e.get("peak_freq", np.nan))
        ]
        metrics["median_peak_freq_swr"] = (
            np.median(peak_freqs) if peak_freqs else np.nan
        )
    else:
        metrics["median_purity_swr"] = np.nan
        metrics["median_peak_freq_swr"] = np.nan

    spw_gf = gold_sets["n_spw_gold"] / max(1, len(spw_events))
    swr_gf = gold_sets["n_ripple_gold"] / max(1, ripple_tiers["n_swr"])
    metrics["spw_gold_frac"] = spw_gf
    metrics["swr_gold_frac"] = swr_gf
    if spw_gf >= 0.1 and swr_gf >= 0.1:
        score += 1
    elif spw_gf >= 0.05 or swr_gf >= 0.05:
        score += 0.5

    metrics["quality_score"] = score
    metrics["max_score"] = 3.5
    return metrics


# =============================================================================
# MAIN PROCESSOR
# =============================================================================


def process_session(session_name, data_dir):
    print(f"\n{'=' * 60}")
    print(f"Processing: {session_name}")
    print(f"{'=' * 60}")

    session_dir = data_dir / session_name
    lfp_files = list(session_dir.glob("*.eeg")) + list(session_dir.glob("*.lfp"))
    xml_files = list(session_dir.glob("*.[xX][mM][lL]"))
    sessinfo_files = list(session_dir.glob("*_sessInfo.mat")) + list(
        session_dir.glob("*sessInfo.mat")
    )

    if not lfp_files:
        print(f"  ⚠ No LFP file")
        return None

    lfp_file = lfp_files[0]
    lfp_params = get_lfp_params(lfp_file, xml_files)
    if lfp_params is None:
        print(f"  ⚠ Could not determine LFP parameters")
        return None

    n_ch = lfp_params["n_channels"]
    fs = lfp_params["fs"]
    print(
        f"  LFP: n_ch={n_ch}, fs={fs} Hz, dur={lfp_params['duration_sec'] / 60:.1f} min"
    )

    sessinfo = load_sessinfo(sessinfo_files[0]) if sessinfo_files else None
    if sessinfo is None:
        print(f"  ⚠ Could not load sessInfo")
        return None

    intervals = get_nrem_intervals_in_post(sessinfo)
    if intervals is None:
        return None

    print(f"  Selecting channels...")
    ch_info = select_channels_with_noise(lfp_file, n_ch, fs, intervals, verbose=True)
    if ch_info is None:
        return None
    print(f"    SPW: {ch_info['spw_channels']}")
    print(f"    Ripple: {ch_info['ripple_channels']}")
    print(f"    Noise: {ch_info['noise_channels']}")
    print(f"    Overlap: {ch_info['n_overlap']}")

    print(f"  Detecting SPWs...")
    spw_events = detect_spw(lfp_file, n_ch, fs, ch_info["spw_channels"], intervals)
    print(
        f"    {len(spw_events)} SPWs ({len(spw_events) / (intervals['total_sec'] / 60):.1f}/min)"
    )

    print(f"  Detecting ripples...")
    ripple_events = detect_ripples_with_noise_veto(
        lfp_file,
        n_ch,
        fs,
        ch_info["ripple_channels"],
        ch_info["noise_channels"],
        intervals,
    )
    print(f"    {len(ripple_events)} ripples detected")

    coupling_raw = compute_coupling_spw_to_ripple(spw_events, ripple_events)

    print(f"  Classifying ripples...")
    ripple_tiers = classify_ripples(ripple_events, coupling_raw)
    print(
        f"    Ripples (all): {ripple_tiers['n_all']} ({ripple_tiers['n_all'] / (intervals['total_sec'] / 60):.1f}/min)"
    )
    print(
        f"    Ripples (SWR): {ripple_tiers['n_swr']} ({ripple_tiers['n_swr'] / (intervals['total_sec'] / 60):.1f}/min)"
    )

    coupling_swr = compute_coupling_spw_to_ripple(
        spw_events, ripple_tiers["ripples_swr"]
    )

    print(f"  Coupling:")
    print(f"    SPW→raw ripple: {coupling_raw['coupling_rate'] * 100:.1f}%")
    print(f"    SPW→SWR-tier:   {coupling_swr['coupling_rate'] * 100:.1f}%")

    if len(coupling_swr["coupling_lags_ms"]) > 0:
        print(
            f"    SWR lag: {np.mean(coupling_swr['coupling_lags_ms']):.1f} ± {np.std(coupling_swr['coupling_lags_ms']):.1f} ms"
        )

    gold_sets = extract_gold_sets(spw_events, ripple_tiers)
    print(f"  Gold: {gold_sets['n_spw_gold']} SPW, {gold_sets['n_ripple_gold']} SWR")

    quality = compute_quality(
        spw_events,
        ripple_tiers,
        gold_sets,
        coupling_raw,
        coupling_swr,
        intervals,
        ch_info,
    )
    print(f"  Quality: {quality['quality_score']:.1f}/{quality['max_score']}")

    return {
        "name": session_name,
        **stamp(),
        "lfp_params": lfp_params,
        "nrem_intervals": intervals,
        "channel_info": ch_info,
        "spw_events": spw_events,
        "ripple_events_raw": ripple_events,
        "ripple_tiers": ripple_tiers,
        "coupling_raw": coupling_raw,
        "coupling_swr": coupling_swr,
        "gold": gold_sets,
        "quality": quality,
    }


# =============================================================================
# BATCH RUN
# =============================================================================


def process_all(data_dir):
    sessions = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(sessions)} sessions")
    results = []
    for session in sessions:
        try:
            r = process_session(session, data_dir)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            import traceback

            traceback.print_exc()
    return results


# RUN
results = process_all(DIRS["data_raw"])

if results:
    results_sorted = sorted(
        results, key=lambda x: x["quality"]["quality_score"], reverse=True
    )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(
        f"\n{'Session':<18} {'NREM':<7} {'SPW/m':<7} {'SWR/m':<7} {'All/m':<7} {'Coup_SWR':<9} {'Purity':<7} {'Cover%':<7}"
    )
    print("-" * 90)

    for r in results_sorted:
        q = r["quality"]
        purity_str = (
            f"{q['median_purity_swr']:.2f}"
            if np.isfinite(q.get("median_purity_swr", np.nan))
            else "N/A"
        )
        cov_str = f"{q['ripple_coverage'] * 100:.0f}%"
        print(
            f"{r['name']:<18} {q['nrem_duration_min']:<7.1f} {q['spw_rate']:<7.1f} "
            f"{q['swr_rate']:<7.1f} {q['ripple_all_rate']:<7.1f} "
            f"{q['coupling_swr'] * 100:<9.1f} {purity_str:<7} {cov_str:<7}"
        )

    # Save
    output_path = FILES["detector"]
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                **stamp(),
                "config": CONFIG,
                "datasets": results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
        )
    print(f"\n✓ Saved to {output_path}")

    # Pooled stats
    print("\n" + "=" * 90)
    print("POOLED STATISTICS")
    print("=" * 90)
    total_spw = sum(len(r["spw_events"]) for r in results)
    total_swr = sum(r["ripple_tiers"]["n_swr"] for r in results)
    total_gold_spw = sum(r["gold"]["n_spw_gold"] for r in results)
    total_gold_swr = sum(r["gold"]["n_ripple_gold"] for r in results)

    print(f"  Total SPW events: {total_spw}")
    print(f"  Total SWR events: {total_swr}")
    print(
        f"  Gold SPW: {total_gold_spw} ({total_gold_spw / max(1, total_spw) * 100:.1f}%)"
    )
    print(
        f"  Gold SWR: {total_gold_swr} ({total_gold_swr / max(1, total_swr) * 100:.1f}% of SWR)"
    )

else:
    print("\n⚠ No sessions processed")

# ===== POST-BLOCK-6 VISUAL DIAGNOSTICS (no changes to Block 6) =====

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# --------- CONFIG YOU CAN TWEAK ----------
SESSION_NAME = (
    "Cicero_09172014"  # e.g., "Achilles_10252013" ; None = use first available
)
WIN_SEC = 60  # analysis window length (seconds). 60 is fast + stable
RIPPLE_BAND = (150, 250)
NOISE_BAND = (300, 500)
FILTER_ORDER = 4
TOP_N = 24  # show top-N channels in bar plot
# ----------------------------------------


def _band_sos(fs, band, order=4):
    lo, hi = band
    hi = min(hi, fs / 2 - 1)
    if hi <= lo:
        return None
    return signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def load_results_fallback(DIRS):
    # Use in-memory `results` if present, else load the detector artefact
    if (
        "results" in globals()
        and isinstance(globals()["results"], list)
        and len(globals()["results"]) > 0
    ):
        return globals()["results"]
    if not FILES["detector"].exists():
        raise RuntimeError(
            f"No in-memory `results` and no {FILES['detector'].name} found."
        )
    with open(FILES["detector"], "rb") as f:
        obj = pickle.load(f)
    return obj.get("datasets", [])


def pick_session(results, session_name=None):
    if not results:
        raise RuntimeError("No sessions found in results.")
    if session_name is None:
        return results[0]
    for r in results:
        if r.get("name") == session_name:
            return r
    print(
        f"⚠ SESSION_NAME='{session_name}' not found. Using first session: {results[0].get('name')}"
    )
    return results[0]


def compute_channel_powers(lfp_path, n_ch, fs, start_samp, end_samp):
    mmap = np.memmap(lfp_path, dtype=np.int16, mode="r").reshape(-1, n_ch)

    rip_sos = _band_sos(fs, RIPPLE_BAND, FILTER_ORDER)
    noi_sos = _band_sos(fs, NOISE_BAND, FILTER_ORDER)

    ripple_power = np.zeros(n_ch, dtype=float)
    noise_power = np.zeros(n_ch, dtype=float)

    for ch in range(n_ch):
        x = mmap[start_samp:end_samp, ch].astype(np.float32)

        # ripple band
        if rip_sos is not None:
            xr = signal.sosfiltfilt(rip_sos, x)
            ripple_power[ch] = float(np.var(xr))
        else:
            ripple_power[ch] = np.nan

        # noise band
        if noi_sos is not None:
            xn = signal.sosfiltfilt(noi_sos, x)
            noise_power[ch] = float(np.var(xn))
        else:
            noise_power[ch] = np.nan

    del mmap
    return ripple_power, noise_power


def plot_selection(ripple_power, noise_power, selected_channels, top_n=24):
    ch = np.arange(len(ripple_power))
    selected = np.isin(ch, np.array(selected_channels))

    # 1) Ranked ripple power
    order = np.argsort(ripple_power)[::-1]
    show = order[: min(top_n, len(order))]
    ch_s = ch[show]
    rp_s = ripple_power[show]
    sel_s = selected[show]

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(rp_s)), rp_s)
    plt.xticks(np.arange(len(rp_s)), ch_s, rotation=90)
    plt.title("Top channels by ripple-band power (◯ = selected)")
    for i, is_sel in enumerate(sel_s):
        if is_sel:
            plt.plot(i, rp_s[i], marker="o", linestyle="None")
    plt.tight_layout()
    plt.show()

    # 2) Ripple vs Noise scatter
    plt.figure(figsize=(5.5, 5))
    plt.scatter(ripple_power, noise_power, s=18)
    plt.scatter(ripple_power[selected], noise_power[selected], s=60, marker="x")
    plt.xlabel("Ripple power (150–250 Hz)")
    plt.ylabel("Noise power (300–500 Hz)")
    plt.title("Ripple vs Noise space (x = selected)")
    plt.tight_layout()
    plt.show()

    # 3) Coverage curve + selected/top8 coverage
    rp_sorted = np.sort(ripple_power)[::-1]
    eps = 1e-10
    cum = np.cumsum(rp_sorted) / (np.sum(rp_sorted) + eps)

    k = len(selected_channels)
    selected_power = float(np.sum(ripple_power[selected]))
    topk_power = float(np.sum(rp_sorted[:k]))
    coverage = selected_power / (topk_power + eps)

    plt.figure(figsize=(7, 4))
    plt.plot(cum)
    plt.axvline(k - 1, linestyle="--")
    plt.ylim(0, 1.01)
    plt.xlabel("Channel rank (by ripple power)")
    plt.ylabel("Cumulative fraction of ripple power")
    plt.title(f"Coverage curve | selected/top-{k} ripple-power = {coverage * 100:.1f}%")
    plt.tight_layout()
    plt.show()

    topk_channels = list(np.argsort(ripple_power)[::-1][:k])
    print("Selected ripple channels:", selected_channels)
    print("Top-k ripple channels:     ", topk_channels)
    print(f"Selected/top-{k} ripple-power coverage: {coverage * 100:.1f}%")


# ---------- RUN ----------
# expects DIRS from your notebook; otherwise define it here
# DIRS = {'data_raw': Path(...), 'data_processed': Path(...)}  # only needed if results not in memory

results_local = load_results_fallback(DIRS)
sess = pick_session(results_local, SESSION_NAME)

name = sess["name"]
session_dir = DIRS["data_raw"] / name
lfp_files = list(session_dir.glob("*.eeg")) + list(session_dir.glob("*.lfp"))
if not lfp_files:
    raise RuntimeError(f"No LFP file found in {session_dir}")
lfp_path = lfp_files[0]

n_ch = int(sess["lfp_params"]["n_channels"])
fs = float(sess["lfp_params"]["fs"])

# Pick a mid-point window inside the first NREM interval (fast + representative)
intervals = sess["nrem_intervals"]["intervals"]
s0, e0 = intervals[0]
mid = 0.5 * (s0 + e0)
half = WIN_SEC / 2
win_start_sec = max(s0, mid - half)
win_end_sec = min(e0, mid + half)

start_samp = int(win_start_sec * fs)
end_samp = int(win_end_sec * fs)
end_samp = max(end_samp, start_samp + int(fs))  # at least 1s

selected_ripple = sess["channel_info"]["ripple_channels"]

print(f"Session: {name}")
print(
    f"Window:  {win_start_sec:.1f}–{win_end_sec:.1f} s ({(win_end_sec - win_start_sec):.1f} s), fs={fs}, n_ch={n_ch}"
)

rp, npow = compute_channel_powers(lfp_path, n_ch, fs, start_samp, end_samp)
plot_selection(rp, npow, selected_ripple, top_n=TOP_N)
