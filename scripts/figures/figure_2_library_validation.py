"""
Figure 2 - synthetic library against empirical events.

Reads:  data_processed/swr_detection.pkl
        models/validation_null.pkl
Writes: figures/figure_2.{png,pdf,tiff}

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

from orexin_paths import DIRS, PROJECT_ROOT, ensure_dirs  # noqa: E402
from orexin_keys import ox_key  # noqa: E402,F401

ensure_dirs()

# =============================================================================
# FIGURE 2 — Synthetic library validation
#
# Library built from Samples 1-3 (training). Compared with the training
# sessions and with Sample 4 (held out, different animal). Sample 5 is
# excluded on detection quality before any comparison (see EXCLUDED).
# Synthetic data come from validation_null.pkl; the cell stops if a
# held-out session appears in the library's source_sessions.
#
# Panels: A session-level rates, B-E marginal event distributions.
#   The duration-vs-frequency panel was dropped; the three Spearman
#   correlations are still printed as a diagnostic, since the dependence is a
#   property of the generator worth knowing even when it is not shown.
# =============================================================================
import pickle
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wasserstein_distance, ks_2samp, spearmanr

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
except Exception:
    pass
try:
    from IPython.display import display
except Exception:
    def display(obj): pass

# Paths come from orexin_paths (see header). DIRS / FILES are already bound.
FIG_DIR = PROJECT_ROOT / "figures"          # same folder as Figures 3-5
FIG_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_PKL = DIRS["data_processed"] / "swr_detection.pkl"
LIB_PKL      = DIRS["models"] / "synthetic_libraries.pkl"
VAL_PKL      = DIRS["models"] / "validation_null.pkl"
CFG_PKL      = DIRS["models"] / "generator_config.pkl"

CONTEXT, OX_NULL, N_PRED, RNG_SEED = "Context_A", 1.0, 4000, 42
N_BOOT_KS = 300
MIN_N_COMPARE = 20
TIFF_DPI, TIFF_COMPRESS = 300, "tiff_lzw"

TRAINING_SESSIONS = ["Achilles_10252013", "Cicero_09102014", "Cicero_09172014"]
HELD_OUT_SESSIONS = ["Buddy_06272013"]
EXCLUDED = {
    "Gatsby_08282013": ("its ripple detection failed quality control (12 SWR-tier events, "
                        "0.1/min; all-tier ripple rate 1.4/min against 10-19/min in the "
                        "other sessions; 552 noise-vetoed events; 16 of 134 channels "
                        "vetoed), leaving too few events for a distributional comparison"),
}
FIG1_ORDER = TRAINING_SESSIONS + ["Buddy_06272013", "Gatsby_08282013"]

COL_EMP, COL_SYN, COL_SYN_D = "#e05a4b", "#5b9bd5", "#1a5fa8"
COL_HELD = "#66c28a"          # Sample 4; same colour in Figure 1
HELD_LS = (0, (4, 2))

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 9, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _safe_np(a): return np.array(a) if a is not None else np.array([])


def _fin(a):
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


def coupled_event_arrays_from_trial(trial):
    """Paired per-event (duration, frequency, cycles) from one synthetic trial."""
    for kd, kf, kc in [("swr_durations", "swr_frequencies", "swr_cycles"),
                       ("swr_event_durations", "swr_event_frequencies", "swr_event_cycles"),
                       ("coupled_ripple_durations", "coupled_ripple_frequencies",
                        "coupled_ripple_cycles")]:
        if kd in trial and kf in trial and kc in trial:
            d, f, c = (_safe_np(trial[k]).astype(float) for k in (kd, kf, kc))
            n = min(len(d), len(f), len(c))
            d, f, c = d[:n], f[:n], c[:n]
            ok = np.isfinite(d) & np.isfinite(f) & np.isfinite(c)
            return d[ok], f[ok], c[ok]
    raise KeyError("Synthetic trial has no per-event duration/frequency/cycle arrays.")


def pooled_coupled_event_arrays(trials):
    parts = [coupled_event_arrays_from_trial(t) for t in trials]
    parts = [p for p in parts if len(p[0])]
    cat = lambda i: np.concatenate([p[i] for p in parts]) if parts else np.array([])
    return cat(0), cat(1), cat(2)


def step_outline(ax, values, bins, **kw):
    """Density outline drawn as a line, so dash styles render on every matplotlib."""
    h, edges = np.histogram(np.asarray(values, float), bins=bins, density=True)
    xs = np.concatenate([[edges[0]], np.repeat(edges, 2)[1:-1], [edges[-1]]])
    ys = np.concatenate([[0.0], np.repeat(h, 2), [0.0]])
    return ax.plot(xs, ys, **kw)[0]


def require_key(trials, key):
    if not trials or key not in trials[0]:
        raise KeyError(f"Synthetic trials have no '{key}'.")


# ─── agreement statistics ────────────────────────────────────────────────────
def agreement_stats(e, s, n_boot=N_BOOT_KS, seed=0):
    """Matched-n KS against its own noise floor, plus SD-standardised distances.

    Both KS values use the same n = min(half of e, all of s), drawn without
    replacement: cross = e vs s, self = two disjoint halves of e (noise floor).
    """
    e, s = _fin(e), _fin(s)
    if len(e) < MIN_N_COMPARE or len(s) < MIN_N_COMPARE:
        return None
    rng = np.random.default_rng(seed)
    h = min(len(e) // 2, len(s))
    cross, selfs = [], []
    for _ in range(n_boot):
        idx = rng.permutation(len(e))
        e1, e2 = e[idx[:h]], e[idx[h:2 * h]]
        cross.append(ks_2samp(e1, rng.choice(s, size=h, replace=False)).statistic)
        selfs.append(ks_2samp(e1, e2).statistic)
    sd = float(np.std(e)) or np.nan
    return {"ks": float(np.median(cross)),
            "ks_ci": [float(x) for x in np.percentile(cross, [2.5, 97.5])],
            "ks_self": float(np.median(selfs)),
            "ks_self_ci": [float(x) for x in np.percentile(selfs, [2.5, 97.5])],
            "emd_raw": float(wasserstein_distance(e, s)),
            "emd_sd": float(wasserstein_distance(e, s) / sd),
            "dmu_raw": float(np.mean(s) - np.mean(e)),
            "dmu_sd": float((np.mean(s) - np.mean(e)) / sd),
            "n_e": int(len(e)), "n_s": int(len(s)), "n_match": h}


# ─── prediction intervals ────────────────────────────────────────────────────
# phi = max(within-trial, between-session): the wider dispersion, so bands are
# conservative. Training: other training sessions; held out: all of them.
def rate_band(null_trials, ref_values, duration_min, count_key, duration_sec, rng):
    """Gamma-Poisson prediction interval for an event rate (events/min)."""
    require_key(null_trials, count_key)
    counts = np.array([int(t[count_key]) for t in null_trials], float)
    mu, var = counts.mean(), counts.var(ddof=1)
    phi_w = max((var - mu) / mu ** 2, 0.0) if mu > 0 else 0.0
    ref = _fin(ref_values)
    phi_b = ref.var(ddof=1) / ref.mean() ** 2 if len(ref) >= 2 and ref.mean() > 0 else 0.0
    phi = max(phi_w, phi_b)
    n_tr = max(1, int(np.ceil(duration_min * 60.0 / duration_sec)))
    gains = rng.gamma(1.0 / phi, phi, size=N_PRED) if phi > 0 else np.ones(N_PRED)
    return rng.poisson(mu * n_tr * gains) / (n_tr * duration_sec) * 60.0, phi


def pcouple_band(null_trials, ref_values, duration_min, duration_sec, rng):
    """Beta-Binomial prediction interval for P(SPW has a coupled ripple)."""
    require_key(null_trials, "n_spw"); require_key(null_trials, "n_spw_with_ripple")
    spw = np.array([int(t["n_spw"]) for t in null_trials], float)
    cpl = np.array([int(t["n_spw_with_ripple"]) for t in null_trials], float)
    p = cpl.sum() / spw.sum() if spw.sum() > 0 else np.nan
    ref = _fin(ref_values)
    phi = max(ref.var(ddof=1) / (p * (1 - p)) if len(ref) >= 2 and 0 < p < 1 else 0, 1e-6)
    n_tr = max(1, int(np.ceil(duration_min * 60.0 / duration_sec)))
    n_spw = max(1, int(round(spw.mean() * n_tr)))
    a, b = p * (1 / phi - 1), (1 - p) * (1 / phi - 1)
    ps = rng.beta(max(a, 1e-3), max(b, 1e-3), size=N_PRED)
    return rng.binomial(n_spw, ps) / n_spw, phi


# ─── empirical ───────────────────────────────────────────────────────────────
with open(DETECTOR_PKL, "rb") as f:
    det = pickle.load(f)
DCFG = det.get("config", {})
by_name = {s["name"]: s for s in det["datasets"]}

_present = set(by_name)
LABEL_ORDER = [s for s in FIG1_ORDER if s in _present] + sorted(_present - set(FIG1_ORDER))
SESSION_LABELS = {nm: f"Sample {i + 1}" for i, nm in enumerate(LABEL_ORDER)}


def short(name): return SESSION_LABELS.get(name, str(name))
def tick_label(name): return short(name) + ("†" if name in HELD_OUT_SESSIONS else "")


_miss = [s for s in TRAINING_SESSIONS if s not in by_name]
if _miss:
    raise KeyError(f"Training sessions missing from the detector pickle: {_miss}")
HELD = [s for s in HELD_OUT_SESSIONS if s in by_name]


def extract(sess):
    ts = float(sess["nrem_intervals"]["total_sec"])
    swr, coup = sess["ripple_tiers"]["ripples_swr"], sess["coupling_swr"]
    n_spw, n_swr = len(sess["spw_events"]), len(swr)
    n_c = int(coup.get("n_coupled_spw", 0))
    sd = np.array([e["duration_ms"] for e in swr], float)
    sf = np.array([e.get("peak_freq", np.nan) for e in swr], float)
    cc = np.array([np.nan if e.get("n_cycles") is None else e["n_cycles"] for e in swr], float)
    lg = np.array(coup.get("coupling_lags_ms", []), float)
    ok = np.isfinite(sd) & np.isfinite(sf)
    arrays = {"swr_durs": sd, "swr_freqs": _fin(sf), "swr_cycles": _fin(cc),
              "coupling_lags": _fin(lg), "pair_d": sd[ok], "pair_f": sf[ok]}
    row = {"session": sess["name"], "duration_min": ts / 60.0,
           "spw_rate_per_min": n_spw / ts * 60.0 if ts > 0 else np.nan,
           "swr_rate_per_min": n_swr / ts * 60.0 if ts > 0 else np.nan,
           "p_couple": n_c / n_spw if n_spw > 0 else np.nan,
           "role": "training" if sess["name"] in TRAINING_SESSIONS else "held-out"}
    return arrays, row


def pool(names, per):
    keys = ("swr_durs", "swr_freqs", "swr_cycles", "coupling_lags", "pair_d", "pair_f")
    return {k: np.concatenate([per[n][k] for n in names]) if names else np.array([])
            for k in keys}


per_sess, rows = {}, []
for nm in TRAINING_SESSIONS + HELD:
    per_sess[nm], r = extract(by_name[nm])
    rows.append(r)
sess_df = pd.DataFrame(rows)
emp, emp_held = pool(TRAINING_SESSIONS, per_sess), pool(HELD, per_sess)
emp_pair, held_pair = (emp["pair_d"], emp["pair_f"]), (emp_held["pair_d"], emp_held["pair_f"])

# ─── synthetic ───────────────────────────────────────────────────────────────
SYN_SRC = VAL_PKL if VAL_PKL.exists() else LIB_PKL
if SYN_SRC == LIB_PKL:
    print(f"⚠ {VAL_PKL.name} not found; using the 200-trial agent library.")
with open(SYN_SRC, "rb") as f:
    lib = pickle.load(f)

LIB_SOURCES = lib.get("source_sessions")
if LIB_SOURCES is None:
    print("⚠ Library does not record its source sessions; held-out independence "
          "cannot be verified.")
else:
    _leak = sorted(set(LIB_SOURCES) & (set(HELD_OUT_SESSIONS) | set(EXCLUDED)))
    if _leak:
        raise ValueError(f"Library was built from held-out sessions {_leak}; "
                         f"rebuild it from the training sessions only.")

try:
    with open(CFG_PKL, "rb") as f:
        GCFG = pickle.load(f)
except Exception:
    GCFG = {}
_libcfg = lib.get("config", {})
GCFG = {**(_libcfg if isinstance(_libcfg, dict) else {}), **GCFG}

duration_sec = float(lib.get("duration_sec", 3.0))
null_trials = lib["libraries"]["null"][f"{CONTEXT}_ox{ox_key(OX_NULL)}"]
syn_d, syn_f, syn_c = pooled_coupled_event_arrays(null_trials)
syn_lags = _fin(np.concatenate([_safe_np(t.get("coupling_lags", [])) for t in null_trials]))
syn = {"swr_durs": syn_d, "swr_freqs": syn_f, "swr_cycles": syn_c, "coupling_lags": syn_lags}
syn_pair = (syn_d, syn_f)

MIN_CYC_GEN = GCFG.get("min_ripple_cycles")
MIN_CYC_DET = DCFG.get("ripple_min_cycles")

# ─── prediction bands ────────────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)
train_df = sess_df[sess_df["role"] == "training"]
METRICS = ["spw_rate_per_min", "swr_rate_per_min", "p_couple"]
COUNT_KEY = {"spw_rate_per_min": "n_spw", "swr_rate_per_min": "n_coupled"}
pred_bands = {c: {} for c in METRICS}
for _, row in sess_df.iterrows():
    sn = row["session"]
    ref_df = train_df[train_df["session"] != sn] if row["role"] == "training" else train_df
    for c in METRICS:
        if c == "p_couple":
            band, _ = pcouple_band(null_trials, ref_df[c].values, row["duration_min"],
                                   duration_sec, rng)
        else:
            band, _ = rate_band(null_trials, ref_df[c].values, row["duration_min"],
                                COUNT_KEY[c], duration_sec, rng)
        pred_bands[c][sn] = band

pctile = {sn: {c: float(np.mean(pred_bands[c][sn] <=
                               sess_df.loc[sess_df["session"] == sn, c].iloc[0])) * 100
               for c in METRICS} for sn in sess_df["session"]}


def coverage(names):
    return (sum(2.5 < pctile[sn][c] < 97.5 for sn in names for c in METRICS),
            len(names) * len(METRICS))


def _pct(p):
    if p >= 99: return ">99th"
    if p <= 1:  return "<1st"
    n = int(round(p))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ─── figure ──────────────────────────────────────────────────────────────────
def panel_title(ax, text):
    ax.set_title(text, fontweight="bold", loc="left", x=0.0, fontsize=12)


fig = plt.figure(figsize=(13.2, 11.6))
gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1.0, 0.10, 1.0, 1.0],
                       hspace=0.50, wspace=0.26,
                       left=0.075, right=0.965, top=0.945, bottom=0.055)

# A: session-level agreement
gsA = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.36)
panelA_cfg = [("spw_rate_per_min", "SPW rate (events/min)", "SPW rate"),
              ("swr_rate_per_min", "SWR rate (events/min)", "SWR rate"),
              ("p_couple",         "P(SPW→SWR)",       "SPW→SWR coupling")]
snames = list(sess_df["session"]); n_s = len(snames); n_train = len(TRAINING_SESSIONS)
for si, (col, ylabel, title) in enumerate(panelA_cfg):
    ax = fig.add_subplot(gsA[0, si])
    for s_idx, sn in enumerate(snames):
        bv = pred_bands[col][sn]
        ev = float(sess_df.loc[sess_df["session"] == sn, col].iloc[0])
        lo, hi = np.percentile(bv, [2.5, 97.5])
        vp = ax.violinplot(bv, positions=[s_idx], widths=0.62,
                           showmedians=True, showextrema=False)
        for pc in vp["bodies"]:
            pc.set_facecolor(COL_SYN); pc.set_alpha(0.45)
        vp["cmedians"].set_color(COL_SYN_D); vp["cmedians"].set_linewidth(2)
        ax.plot([s_idx, s_idx], [lo, hi], color=COL_SYN_D, lw=1.2, ls="--", alpha=0.75)
        mcol = COL_HELD if sn in HELD else COL_EMP
        ax.scatter([s_idx], [ev], color=mcol, zorder=6, s=70, marker="D",
                   edgecolor="white", linewidth=0.8)
        ax.annotate(_pct(pctile[sn][col]), xy=(s_idx, ev), xytext=(s_idx + 0.26, ev),
                    fontsize=9, color="#333333", va="center",
                    arrowprops=dict(arrowstyle="->", color=mcol, lw=0.8))
    if HELD:
        ax.axvline(n_train - 0.5, color="#888", lw=0.9, ls=":")
    ax.set_xlim(-0.6, n_s - 0.2)
    ax.set_xticks(range(n_s))
    ax.set_xticklabels([tick_label(sn).replace('Sample ', '') for sn in snames],
                       fontsize=10.5)
    ax.set_xlabel("Sample")
    ax.set_ylabel(ylabel, fontsize=11.5)
    panel_title(ax, title)
    if si == 0:
        ax.text(-0.26, 1.10, "A", transform=ax.transAxes, fontsize=17, fontweight="bold")

# key
ax_leg = fig.add_subplot(gs[1, :]); ax_leg.axis("off"); ax_leg.grid(False)
_key = [Line2D([0], [0], marker="D", color="w", markerfacecolor=COL_EMP, markersize=8,
               label="Training sessions"),
        Patch(fc=COL_SYN, alpha=0.45, label="Synthetic, null model (ox = 1.0)"),
        Line2D([0], [0], color=COL_SYN_D, ls="--", label="95% prediction interval")]
if HELD:
    _key.insert(1, Line2D([0], [0], marker="D", color=COL_HELD, ls=HELD_LS, lw=1.6,
                          markerfacecolor=COL_HELD, markersize=7,
                          label=f"Held-out session ({', '.join(tick_label(s) for s in HELD)})"))
ax_leg.legend(handles=_key, loc="center", ncol=len(_key), frameon=False, fontsize=10,
              handletextpad=0.6, columnspacing=1.8)

# B-E: marginal distributions
panelBE_cfg = [
    (gs[2, 0], "B", "SWR duration (ms)",   "SWR duration",   "swr_durs",      (15, 110), 26),
    (gs[2, 1], "C", "Cycles per SWR",      "Cycle count",    "swr_cycles",    (2, 18),   28),
    (gs[3, 0], "D", "Peak frequency (Hz)", "Peak frequency", "swr_freqs",     (145, 250), 28),
    (gs[3, 1], "E", "SPW→SWR lag (ms)", "Coupling lag", "coupling_lags", (-25, 60), 35),
]
stats_out = {}
for cell, letter, xlabel, title, key, xlim, nbins in panelBE_cfg:
    ax = fig.add_subplot(cell)
    ec, sc, hc = _fin(emp[key]), _fin(syn[key]), _fin(emp_held[key])
    bins = np.linspace(xlim[0], xlim[1], nbins)
    ax.hist(ec, bins=bins, density=True, alpha=0.70, color=COL_EMP,
            label=f"Training (n={len(ec)})")
    ax.hist(sc, bins=bins, density=True, alpha=0.55, color=COL_SYN,
            label=f"Synthetic (n={len(sc)})")
    if len(hc) >= MIN_N_COMPARE:
        step_outline(ax, hc, bins, lw=1.8, color=COL_HELD, ls=HELD_LS,
                     label=f"Held out, {', '.join(tick_label(s) for s in HELD)} (n={len(hc)})")
    if len(ec):
        ax.axvline(np.mean(ec), color=COL_EMP, lw=1.4, ls="--",
                   label=f"Training mean {np.mean(ec):.1f}")
    if len(sc):
        ax.axvline(np.mean(sc), color=COL_SYN_D, lw=1.4, ls="--",
                   label=f"Synthetic mean {np.mean(sc):.1f}")
    if key == "swr_cycles":
        if MIN_CYC_DET is not None:
            ax.axvline(MIN_CYC_DET, color="#333", lw=1.2, ls=":",
                       label=f"Detector floor ({MIN_CYC_DET})")
        if MIN_CYC_GEN is not None and MIN_CYC_GEN != MIN_CYC_DET:
            ax.axvline(MIN_CYC_GEN, color="#8f2f24", lw=1.2, ls=":",
                       label=f"Generator floor ({MIN_CYC_GEN})")
    ax.set_xlim(xlim)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
    seed = sum(map(ord, key))
    stats_out[key] = {"train": agreement_stats(emp[key], syn[key], seed=seed),
                      "held": agreement_stats(emp_held[key], syn[key], seed=seed + 1),
                      "held_vs_train": agreement_stats(emp_held[key], emp[key], seed=seed + 2)}
    panel_title(ax, title)
    ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95,
              borderpad=0.4, handlelength=1.6, labelspacing=0.3)
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=17, fontweight="bold")

# Duration-vs-frequency dependence: not a panel any more, still reported below.
ed, ef = emp_pair; sd_, sf_ = syn_pair; hd, hf = held_pair
rho_e = spearmanr(ef, ed).statistic if len(ed) > 10 else np.nan
rho_s = spearmanr(sf_, sd_).statistic if len(sd_) > 10 else np.nan
rho_h = spearmanr(hf, hd).statistic if len(hd) > 10 else np.nan

png = FIG_DIR / "figure_2.png"
pdf = FIG_DIR / "figure_2.pdf"
tif = FIG_DIR / "figure_2.tiff"
fig.savefig(png, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
try:
    fig.savefig(tif, dpi=TIFF_DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": TIFF_COMPRESS} if TIFF_COMPRESS else {})
    _tif_msg = str(tif)
except Exception as ex:
    _tif_msg = f"TIFF not written: {ex}"
display(fig); plt.show(); plt.close(fig)

# ─── caption numbers ─────────────────────────────────────────────────────────
def _fmt(r):
    if not r:
        return "n/a (too few events)"
    return (f"KS {r['ks']:.3f} [{r['ks_ci'][0]:.3f}, {r['ks_ci'][1]:.3f}] vs floor "
            f"{r['ks_self']:.3f}; EMD {r['emd_sd']:.2f} SD; mean diff {r['dmu_raw']:+.2f} "
            f"({r['dmu_sd']:+.2f} SD); matched n {r['n_match']}")


cov_t, cov_h = coverage(TRAINING_SESSIONS), coverage(HELD)
print("=" * 74)
print("FIGURE 2 CAPTION NUMBERS")
print("=" * 74)
print(f"  library: {SYN_SRC.name}, {len(null_trials)} trials, {len(syn_d)} synthetic SWRs; "
      f"sources: {', '.join(short(s) for s in (LIB_SOURCES or []))}")
print("  Sample key: " + ", ".join(f"{tick_label(s)} = {s}" for s in snames))
for sn in snames:
    row = sess_df.loc[sess_df["session"] == sn].iloc[0]
    print(f"  A  {tick_label(sn):<10}{row['duration_min']:.0f} min | "
          f"SPW {row['spw_rate_per_min']:.1f}/min ({_pct(pctile[sn]['spw_rate_per_min'])}) | "
          f"SWR {row['swr_rate_per_min']:.1f}/min ({_pct(pctile[sn]['swr_rate_per_min'])}) | "
          f"P(couple) {row['p_couple']:.3f} ({_pct(pctile[sn]['p_couple'])})")
print(f"     inside 95% interval: training {cov_t[0]}/{cov_t[1]}, held out {cov_h[0]}/{cov_h[1]}")
for key, lbl in [("swr_durs", "B duration"), ("swr_cycles", "C cycles"),
                 ("swr_freqs", "D frequency"), ("coupling_lags", "E lag")]:
    r = stats_out[key]
    print(f"  {lbl}")
    print(f"     training vs synthetic   {_fmt(r['train'])}")
    print(f"     held out vs synthetic   {_fmt(r['held'])}")
    print(f"     held out vs training    {_fmt(r['held_vs_train'])}")
print("  [not a panel] duration vs peak frequency, Spearman rho: "
      f"training {rho_e:+.3f}, held out {rho_h:+.3f}, synthetic {rho_s:+.3f}")
print("     The synthetic value is more negative because a generated event's duration")
print("     is its cycle count divided by its independently sampled peak frequency.")
print("     The agent reads cycle count, amplitude and lag, never duration.")
for nm, why in EXCLUDED.items():
    if nm in by_name:
        print(textwrap.fill(f"  Excluded: {short(nm)} ({nm}), because {why}.", width=76,
                            subsequent_indent="    "))
if MIN_CYC_GEN is not None and MIN_CYC_DET is not None and MIN_CYC_GEN != MIN_CYC_DET:
    print(f"  Note: detector cycle floor {MIN_CYC_DET}, generator floor {MIN_CYC_GEN} (panel C).")
print(f"\nSaved to {FIG_DIR}")
for _p in (png, pdf, tif):
    print(f"  {_p.name:<16}" + (f"{_p.stat().st_size / 1e6:.1f} MB" if _p.exists() else "NOT FOUND"))
if "not" in _tif_msg:
    print(f"  {_tif_msg}")
