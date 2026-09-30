"""
Detector validation - example events and distribution checks.

Reads:  data_processed/swr_detection.pkl
        data_raw/<session>/*.lfp|.eeg (for the example traces)

Plots randomly sampled SWR events with their band-filtered traces and
spectrograms, then checks the detected distributions and coupling
against the published ranges. Run before building the library.

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
# VISUAL VALIDATION — EXAMPLE SWR EVENTS
# =============================================================================
"""
Visual validation of detected SWR events.
For each event, show:
1. Raw LFP (wideband) - SPW channel
2. SPW band envelope (5-40 Hz) - radiatum
3. Ripple-band filtered (150-250 Hz) - pyramidal
4. Spectrogram around peak
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import pickle

# ============ CONFIG ============
SESSION_NAME = "Cicero_09102014"  # None = use best session, or specify e.g. "Cicero_09172014"
N_EVENTS = 30        # Number of random SWR events to plot
WINDOW_MS = 200      # Window around peak (±100ms)
COLS = 5             # Columns in grid
SPECTROGRAM = True   # Include spectrogram panel
SEED = 42            # For reproducibility
# ================================

# Paths come from orexin_paths (see header). DIRS / FILES are already bound.

# Load results
pkl_path = FILES['detector']
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
results = data['datasets']
config = data['config']

# Pick session
if SESSION_NAME:
    sess = next((r for r in results if r['name'] == SESSION_NAME), results[0])
else:
    sess = max(results, key=lambda r: r['ripple_tiers']['n_swr'])

print(f"Session: {sess['name']}")
print(f"SWR events: {sess['ripple_tiers']['n_swr']}")
print(f"SPW channels: {sess['channel_info']['spw_channels']}")
print(f"Ripple channels: {sess['channel_info']['ripple_channels']}")

# Get paths and params
session_dir = DIRS['data_raw'] / sess['name']
lfp_files = list(session_dir.glob("*.eeg")) + list(session_dir.glob("*.lfp"))
lfp_path = lfp_files[0]

n_ch = sess['lfp_params']['n_channels']
fs = sess['lfp_params']['fs']

# Get SWR events
swr_events = sess['ripple_tiers']['ripples_swr']
spw_channels = sess['channel_info']['spw_channels']
ripple_channels = sess['channel_info']['ripple_channels']

# Sample random events
np.random.seed(SEED)
n_to_plot = min(N_EVENTS, len(swr_events))
indices = np.random.choice(len(swr_events), n_to_plot, replace=False)
indices = sorted(indices)

# Load LFP
mmap = np.memmap(lfp_path, dtype=np.int16, mode='r').reshape(-1, n_ch)

# Filter design
def bandpass(data, fs, lo, hi, order=4):
    hi = min(hi, fs/2 - 1)
    sos = signal.butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data, axis=0)

# Window samples
half_win = int((WINDOW_MS / 2) * fs / 1000)

# Figure setup
rows_per_event = 4 if SPECTROGRAM else 3
n_rows = int(np.ceil(n_to_plot / COLS))

fig, axes = plt.subplots(n_rows * rows_per_event, COLS,
                          figsize=(COLS * 3, n_rows * rows_per_event * 1.2))
if n_rows == 1:
    axes = axes.reshape(rows_per_event, -1)

session_name = sess['name']
fig.suptitle(f'{session_name} - {n_to_plot} Random SWR Events', fontsize=14, y=1.02)

for plot_idx, evt_idx in enumerate(indices):
    evt = swr_events[evt_idx]
    peak_samp = int(evt['peak_sec'] * fs)

    # Extract window
    start = max(0, peak_samp - half_win)
    end = min(mmap.shape[0], peak_samp + half_win)

    # Get data from representative channels
    spw_ch = spw_channels[0]  # First SPW channel (radiatum)
    rip_ch = ripple_channels[0]  # First ripple channel (pyramidal)

    raw_spw = mmap[start:end, spw_ch].astype(np.float32)
    raw_rip = mmap[start:end, rip_ch].astype(np.float32)

    # Filter
    spw_filt = bandpass(raw_spw, fs, 5, 40)
    spw_env = np.abs(signal.hilbert(spw_filt))
    rip_filt = bandpass(raw_rip, fs, 150, 250)

    # Time axis
    t_ms = np.arange(len(raw_spw)) / fs * 1000 - (WINDOW_MS / 2)

    # Grid position
    col = plot_idx % COLS
    row_base = (plot_idx // COLS) * rows_per_event

    # Panel 1: Raw LFP (SPW channel)
    ax1 = axes[row_base, col] if n_rows > 1 or rows_per_event > 1 else axes[col]
    ax1.plot(t_ms, raw_spw, 'k', lw=0.5)
    ax1.axvline(0, color='r', lw=0.5, alpha=0.5)
    ax1.set_ylabel('Raw', fontsize=7)
    purity_val = evt.get('spectral_purity', 0)
    ax1.set_title(f'#{evt_idx} | {evt["duration_ms"]:.0f}ms | pur={purity_val:.1f}', fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.set_xticklabels([])

    # Panel 2: SPW envelope
    ax2 = axes[row_base + 1, col]
    ax2.plot(t_ms, spw_filt, 'b', lw=0.5, alpha=0.5)
    ax2.plot(t_ms, spw_env, 'b', lw=1)
    ax2.axvline(0, color='r', lw=0.5, alpha=0.5)
    ax2.set_ylabel('SPW', fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.set_xticklabels([])

    # Panel 3: Ripple filtered
    ax3 = axes[row_base + 2, col]
    ax3.plot(t_ms, rip_filt, 'purple', lw=0.5)
    ax3.axvline(0, color='r', lw=0.5, alpha=0.5)
    ax3.set_ylabel('Ripple', fontsize=7)
    ax3.tick_params(labelsize=6)

    if SPECTROGRAM:
        ax3.set_xticklabels([])
        # Panel 4: Spectrogram
        ax4 = axes[row_base + 3, col]

        # Compute spectrogram
        nperseg = min(64, len(raw_rip) // 4)
        if nperseg >= 16:
            f, t_spec, Sxx = signal.spectrogram(raw_rip, fs=fs, nperseg=nperseg,
                                                 noverlap=nperseg//2, nfft=128)
            # Focus on ripple band
            freq_mask = (f >= 50) & (f <= 300)
            t_spec_ms = (t_spec - t_spec[len(t_spec)//2]) * 1000

            ax4.pcolormesh(t_spec_ms, f[freq_mask],
                          10*np.log10(Sxx[freq_mask] + 1e-10),
                          shading='gouraud', cmap='hot')
            ax4.axhline(150, color='cyan', lw=0.5, ls='--', alpha=0.7)
            ax4.axhline(250, color='cyan', lw=0.5, ls='--', alpha=0.7)
            ax4.axvline(0, color='white', lw=0.5, alpha=0.5)
        ax4.set_ylabel('Hz', fontsize=7)
        ax4.set_xlabel('ms', fontsize=7)
        ax4.tick_params(labelsize=6)
    else:
        ax3.set_xlabel('ms', fontsize=7)

# Hide empty subplots
for idx in range(n_to_plot, COLS * n_rows):
    col = idx % COLS
    row_base = (idx // COLS) * rows_per_event
    for r in range(rows_per_event):
        if row_base + r < axes.shape[0]:
            axes[row_base + r, col].axis('off')

plt.tight_layout()
save_path = DIRS['data_processed'] / f'swr_validation_{session_name}.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

del mmap

print(f"\nSaved to {save_path}")


# =============================================================================
# BONUS: Summary statistics of plotted events
# =============================================================================
print("\n" + "="*60)
print("PLOTTED EVENTS SUMMARY")
print("="*60)

plotted_events = [swr_events[i] for i in indices]
durations = [e['duration_ms'] for e in plotted_events]
purities = [e['spectral_purity'] for e in plotted_events if np.isfinite(e.get('spectral_purity', np.nan))]
peak_freqs = [e['peak_freq'] for e in plotted_events if np.isfinite(e.get('peak_freq', np.nan))]
n_cycles_list = [e['n_cycles'] for e in plotted_events]

print(f"Duration: {np.mean(durations):.1f} +/- {np.std(durations):.1f} ms (range: {np.min(durations):.0f}-{np.max(durations):.0f})")
print(f"Purity:   {np.mean(purities):.2f} +/- {np.std(purities):.2f} (range: {np.min(purities):.1f}-{np.max(purities):.1f})")
print(f"Peak Hz:  {np.mean(peak_freqs):.1f} +/- {np.std(peak_freqs):.1f} (range: {np.min(peak_freqs):.0f}-{np.max(peak_freqs):.0f})")
print(f"Cycles:   {np.mean(n_cycles_list):.1f} +/- {np.std(n_cycles_list):.1f} (range: {np.min(n_cycles_list):.0f}-{np.max(n_cycles_list):.0f})")

# =============================================================================
# DISTRIBUTION VALIDATION + COUPLING METRICS
# =============================================================================
"""
Histogram validation of SWR-tier events across all sessions.
Shows: peak_freq, duration_ms, spectral_purity, coupling_lags_ms
Plus coupling timing KPIs and frequency resolution diagnostic.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Paths come from orexin_paths (see header). DIRS / FILES are already bound.

# Load results
pkl_path = FILES['detector']
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
results = data['datasets']

# Colors for sessions
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Collect data per session (compute fresh to avoid bugs)
session_data = {}
for i, sess in enumerate(results):
    name = sess['name']
    swr_events = sess['ripple_tiers']['ripples_swr']
    coupling_info = sess['coupling_swr']

    # Extract arrays fresh
    peak_freqs = [e['peak_freq'] for e in swr_events if np.isfinite(e.get('peak_freq', np.nan))]
    durations = [e['duration_ms'] for e in swr_events]
    purities = [e['spectral_purity'] for e in swr_events if np.isfinite(e.get('spectral_purity', np.nan))]
    lags = coupling_info['coupling_lags_ms'].tolist() if len(coupling_info['coupling_lags_ms']) > 0 else []

    session_data[name] = {
        'color': colors[i % len(colors)],
        'peak_freq': peak_freqs,
        'duration_ms': durations,
        'purity': purities,
        'coupling_lags': lags,
        'n_swr': len(swr_events),
        'n_unique_freq': len(set(peak_freqs))
    }

# Compute pooled arrays
all_freq = [f for d in session_data.values() for f in d['peak_freq']]
all_dur = [dur for d in session_data.values() for dur in d['duration_ms']]
all_purity = [p for d in session_data.values() for p in d['purity']]
all_lags = [l for d in session_data.values() for l in d['coupling_lags']]

# Create figure with extra row for KPIs
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.3)

fig.suptitle('SWR-Tier Distribution Validation (All Sessions)', fontsize=14, fontweight='bold')

# Define bins
bins_freq = np.linspace(140, 220, 25)
bins_dur = np.linspace(15, 100, 25)
bins_purity = np.linspace(0, 15, 30)
bins_lag = np.linspace(-30, 60, 30)

# 1. Peak Frequency
ax1 = fig.add_subplot(gs[0, 0])
for name, d in session_data.items():
    if d['peak_freq']:
        ax1.hist(d['peak_freq'], bins=bins_freq, alpha=0.5, label=f"{name} (n={len(d['peak_freq'])})",
                 color=d['color'], edgecolor='black', linewidth=0.5)
ax1.axvline(150, color='red', linestyle='--', lw=1.5, label='Ripple band')
ax1.axvline(200, color='red', linestyle='--', lw=1.5)
ax1.axvline(175, color='green', linestyle='-', lw=2, alpha=0.7, label='Expected peak (~175 Hz)')
ax1.set_xlabel('Peak Frequency (Hz)', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Peak Frequency Distribution', fontsize=12)
ax1.legend(fontsize=8)
ax1.set_xlim(140, 220)

# Per-session stats in text box
freq_stats = "Per-session (mean ± std):\n"
for name, d in session_data.items():
    if d['peak_freq']:
        freq_stats += f"  {name[:8]}: {np.mean(d['peak_freq']):.1f}±{np.std(d['peak_freq']):.1f}\n"
freq_stats += f"Pooled: {np.mean(all_freq):.1f}±{np.std(all_freq):.1f} Hz"
ax1.text(0.97, 0.97, freq_stats, transform=ax1.transAxes, fontsize=8, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')

# 2. Duration
ax2 = fig.add_subplot(gs[0, 1])
for name, d in session_data.items():
    if d['duration_ms']:
        ax2.hist(d['duration_ms'], bins=bins_dur, alpha=0.5, label=f"{name} (n={len(d['duration_ms'])})",
                 color=d['color'], edgecolor='black', linewidth=0.5)
ax2.axvline(30, color='orange', linestyle='--', lw=1.5, label='Typical range')
ax2.axvline(80, color='orange', linestyle='--', lw=1.5)
ax2.set_xlabel('Duration (ms)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Duration Distribution', fontsize=12)
ax2.legend(fontsize=8)
ax2.set_xlim(15, 100)

dur_stats = "Per-session (mean ± std):\n"
for name, d in session_data.items():
    if d['duration_ms']:
        dur_stats += f"  {name[:8]}: {np.mean(d['duration_ms']):.1f}±{np.std(d['duration_ms']):.1f}\n"
dur_stats += f"Pooled: {np.mean(all_dur):.1f}±{np.std(all_dur):.1f} ms"
ax2.text(0.97, 0.97, dur_stats, transform=ax2.transAxes, fontsize=8, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')

# 3. Spectral Purity
ax3 = fig.add_subplot(gs[1, 0])
for name, d in session_data.items():
    if d['purity']:
        ax3.hist(d['purity'], bins=bins_purity, alpha=0.5, label=f"{name} (n={len(d['purity'])})",
                 color=d['color'], edgecolor='black', linewidth=0.5)
ax3.axvline(2.0, color='red', linestyle='--', lw=1.5, label='SWR threshold (≥2.0)')
ax3.set_xlabel('Spectral Purity (ripple/gamma power)', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Spectral Purity Distribution', fontsize=12)
ax3.legend(fontsize=8)
ax3.set_xlim(0, 15)

pur_stats = "Per-session (mean ± std):\n"
for name, d in session_data.items():
    if d['purity']:
        pur_stats += f"  {name[:8]}: {np.mean(d['purity']):.1f}±{np.std(d['purity']):.1f}\n"
pur_stats += f"Pooled: {np.mean(all_purity):.1f}±{np.std(all_purity):.1f}"
ax3.text(0.97, 0.97, pur_stats, transform=ax3.transAxes, fontsize=8, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')

# 4. Coupling Lags
ax4 = fig.add_subplot(gs[1, 1])
for name, d in session_data.items():
    if d['coupling_lags']:
        ax4.hist(d['coupling_lags'], bins=bins_lag, alpha=0.5, label=f"{name} (n={len(d['coupling_lags'])})",
                 color=d['color'], edgecolor='black', linewidth=0.5)
ax4.axvline(0, color='black', linestyle='-', lw=2, label='SPW peak (t=0)')
ax4.axvline(-20, color='gray', linestyle=':', lw=1, alpha=0.7)
ax4.axvline(50, color='gray', linestyle=':', lw=1, alpha=0.7, label='Coupling window')
ax4.set_xlabel('Coupling Lag (ms, ripple peak - SPW peak)', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('SPW→SWR Coupling Lag Distribution', fontsize=12)
ax4.legend(fontsize=8)
ax4.set_xlim(-30, 60)

lag_stats = "Per-session (mean ± std):\n"
for name, d in session_data.items():
    if d['coupling_lags']:
        lag_stats += f"  {name[:8]}: {np.mean(d['coupling_lags']):.1f}±{np.std(d['coupling_lags']):.1f}\n"
if all_lags:
    lag_stats += f"Pooled: {np.mean(all_lags):.1f}±{np.std(all_lags):.1f} ms"
ax4.text(0.97, 0.97, lag_stats, transform=ax4.transAxes, fontsize=8, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')

# 5. KPI Table (bottom row spanning both columns)
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

# Compute KPIs per session
kpi_data = []
for name, d in session_data.items():
    lags = np.array(d['coupling_lags'])
    if len(lags) > 0:
        pct_lead = np.mean(lags < 0) * 100
        median_lag = np.median(lags)
        q25, q75 = np.percentile(lags, [25, 75])
        iqr = q75 - q25
        kpi_data.append({
            'session': name,
            'n': len(lags),
            'pct_lead': pct_lead,
            'median': median_lag,
            'q25': q25,
            'q75': q75,
            'iqr': iqr
        })

# Pooled KPIs
all_lags_arr = np.array(all_lags)
pooled_pct_lead = np.mean(all_lags_arr < 0) * 100
pooled_median = np.median(all_lags_arr)
pooled_q25, pooled_q75 = np.percentile(all_lags_arr, [25, 75])
pooled_iqr = pooled_q75 - pooled_q25

# Create table text
table_text = "COUPLING TIMING KPIs\n"
table_text += "=" * 95 + "\n"
table_text += f"{'Session':<22} {'N':<8} {'% Ripple Leads':<16} {'Median Lag':<14} {'IQR':<14} {'[Q25, Q75]':<18}\n"
table_text += "-" * 95 + "\n"

for kpi in kpi_data:
    table_text += f"{kpi['session']:<22} {kpi['n']:<8} {kpi['pct_lead']:<16.1f} {kpi['median']:<14.1f} {kpi['iqr']:<14.1f} [{kpi['q25']:.1f}, {kpi['q75']:.1f}]\n"

table_text += "-" * 95 + "\n"
table_text += f"{'POOLED':<22} {len(all_lags_arr):<8} {pooled_pct_lead:<16.1f} {pooled_median:<14.1f} {pooled_iqr:<14.1f} [{pooled_q25:.1f}, {pooled_q75:.1f}]\n"
table_text += "=" * 95 + "\n\n"

# Interpretation
table_text += "INTERPRETATION:\n"
if 40 < pooled_pct_lead < 60:
    table_text += f"  • Ripple leads {pooled_pct_lead:.0f}% → SPW and ripple are near-synchronous (expected)\n"
elif pooled_pct_lead <= 40:
    table_text += f"  • Ripple leads {pooled_pct_lead:.0f}% → Ripple typically follows SPW peak\n"
else:
    table_text += f"  • Ripple leads {pooled_pct_lead:.0f}% → Ripple often precedes SPW peak (normal for CA1)\n"

if abs(pooled_median) < 5:
    table_text += f"  • Median lag = {pooled_median:.1f} ms → Near-perfect alignment ✓\n"
elif pooled_median < -5:
    table_text += f"  • Median lag = {pooled_median:.1f} ms → Ripple precedes SPW deflection\n"
else:
    table_text += f"  • Median lag = {pooled_median:.1f} ms → Ripple follows SPW deflection\n"

if pooled_iqr < 20:
    table_text += f"  • IQR = {pooled_iqr:.1f} ms → Tight coupling (low jitter) ✓\n"
elif pooled_iqr < 30:
    table_text += f"  • IQR = {pooled_iqr:.1f} ms → Moderate coupling jitter\n"
else:
    table_text += f"  • IQR = {pooled_iqr:.1f} ms → High jitter - review detection\n"

# Add frequency resolution note
table_text += "\n" + "-" * 95 + "\n"
table_text += "NOTE: Peak frequency has limited resolution (fs/nperseg = 1250/128 ≈ 9.8 Hz bins)\n"
for name, d in session_data.items():
    table_text += f"  {name}: {d['n_unique_freq']} unique freq values\n"

ax5.text(0.5, 0.98, table_text, transform=ax5.transAxes, fontsize=9,
         family='monospace', va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.tight_layout()
save_path = DIRS['data_processed'] / 'swr_distributions_with_kpis.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved to {save_path}")

# =============================================================================
# Console Output: Full Stats Table
# =============================================================================
print("\n" + "="*100)
print("FULL SWR-TIER DISTRIBUTION SUMMARY (Per-Session)")
print("="*100)
print(f"\n{'Session':<22} {'N_SWR':<8} {'Freq Mean±Std':<18} {'Dur Mean±Std':<18} {'Purity Mean±Std':<18} {'Unique Freq':<12}")
print("-"*100)

for name, d in session_data.items():
    freq_mean = np.mean(d['peak_freq']) if d['peak_freq'] else np.nan
    freq_std = np.std(d['peak_freq']) if d['peak_freq'] else np.nan
    dur_mean = np.mean(d['duration_ms']) if d['duration_ms'] else np.nan
    dur_std = np.std(d['duration_ms']) if d['duration_ms'] else np.nan
    pur_mean = np.mean(d['purity']) if d['purity'] else np.nan
    pur_std = np.std(d['purity']) if d['purity'] else np.nan

    print(f"{name:<22} {d['n_swr']:<8} {freq_mean:.1f} ± {freq_std:.1f} Hz    {dur_mean:.1f} ± {dur_std:.1f} ms    {pur_mean:.2f} ± {pur_std:.2f}       {d['n_unique_freq']}")

print("-"*100)
print(f"{'POOLED':<22} {sum(d['n_swr'] for d in session_data.values()):<8} "
      f"{np.mean(all_freq):.1f} ± {np.std(all_freq):.1f} Hz    "
      f"{np.mean(all_dur):.1f} ± {np.std(all_dur):.1f} ms    "
      f"{np.mean(all_purity):.2f} ± {np.std(all_purity):.2f}       "
      f"{len(set(all_freq))}")
print("="*100)

# =============================================================================
# Coupling KPIs Console
# =============================================================================
print("\n" + "="*100)
print("COUPLING TIMING KPIs")
print("="*100)
print(f"{'Session':<22} {'N':<8} {'% Ripple Leads':<16} {'Median Lag':<14} {'IQR':<14} {'[Q25, Q75]':<18}")
print("-"*100)

for kpi in kpi_data:
    print(f"{kpi['session']:<22} {kpi['n']:<8} {kpi['pct_lead']:<16.1f} {kpi['median']:<14.1f} {kpi['iqr']:<14.1f} [{kpi['q25']:.1f}, {kpi['q75']:.1f}]")

print("-"*100)
print(f"{'POOLED':<22} {len(all_lags_arr):<8} {pooled_pct_lead:<16.1f} {pooled_median:<14.1f} {pooled_iqr:<14.1f} [{pooled_q25:.1f}, {pooled_q75:.1f}]")
print("="*100)

# =============================================================================
# Final Verdict
# =============================================================================
print("\n" + "="*100)
print("PHYSIOLOGICAL VERDICT")
print("="*100)

verdicts = []
verdict_pass = True

# Check 1: Peak frequency (use mean since median is quantized)
freq_mean_p = np.mean(all_freq)
if 160 <= freq_mean_p <= 190:
    verdicts.append(f"✅ Peak frequency: {freq_mean_p:.1f} Hz mean (expected 160-190 Hz)")
else:
    verdicts.append(f"⚠️  Peak frequency: {freq_mean_p:.1f} Hz mean (outside 160-190 Hz)")
    verdict_pass = False

# Check 2: Duration
dur_mean_p = np.mean(all_dur)
if 25 <= dur_mean_p <= 60:
    verdicts.append(f"✅ Duration: {dur_mean_p:.1f} ms mean (expected 25-60 ms)")
else:
    verdicts.append(f"⚠️  Duration: {dur_mean_p:.1f} ms mean (outside 25-60 ms)")
    verdict_pass = False

# Check 3: Purity
pur_mean_p = np.mean(all_purity)
if pur_mean_p >= 2.5:
    verdicts.append(f"✅ Spectral purity: {pur_mean_p:.2f} mean (good frequency specificity)")
else:
    verdicts.append(f"⚠️  Spectral purity: {pur_mean_p:.2f} mean (below 2.5)")
    verdict_pass = False

# Check 4: Coupling lag
if abs(pooled_median) <= 10:
    verdicts.append(f"✅ Coupling lag: {pooled_median:.1f} ms median (well-aligned)")
else:
    verdicts.append(f"⚠️  Coupling lag: {pooled_median:.1f} ms median (offset > 10ms)")
    verdict_pass = False

# Check 5: Coupling IQR
if pooled_iqr <= 25:
    verdicts.append(f"✅ Coupling IQR: {pooled_iqr:.1f} ms (tight jitter)")
else:
    verdicts.append(f"⚠️  Coupling IQR: {pooled_iqr:.1f} ms (high jitter)")
    verdict_pass = False

# Check 6: Frequency resolution note
verdicts.append(f"ℹ️  Freq resolution: ~9.8 Hz (only {len(set(all_freq))} unique values) - cosmetic issue only")

for v in verdicts:
    print(v)

print("\n" + "-"*100)
if verdict_pass:
    print("🎯 ALL CHECKS PASSED: Detected SWRs are a coherent physiological class!")
    print("   Detection is VALIDATED. Ready for synthetic generation.")
else:
    print("⚠️  Some checks flagged - review distributions before proceeding.")
print("="*100)
