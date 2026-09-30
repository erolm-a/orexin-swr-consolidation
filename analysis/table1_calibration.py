"""
Table 1 - reported antagonist effects against realized library statistics.

Reads:  models/calibration_libraries.pkl (generated on first run)
        models/synthetic_libraries.pkl (fallback)
Writes: tables/table1_calibration.csv
        tables/table1_calibration.md

Uses dedicated high-n libraries with an independent seed. Skewed
statistics are summarised by the median.

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

"""
Table 1 - Correspondence between reported antagonist effects, applied model
multipliers, and realized synthetic library statistics.

Standalone. Needs no upstream state to build the table; only
the optional library-generation step at the top uses `generator` and
`OREXIN_MODELS` if they happen to be defined.

Outputs
-------
  tables/table1_calibration.csv    full numeric table
  tables/table1_calibration.md     manuscript-ready markdown
  console                          provenance audit + flags
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR = DIRS["models"]
OUT_DIR    = DIRS["tables"]

CAL_PKL      = FILES["calibration_libraries"]
FALLBACK_PKL = FILES["libraries"]

CAL_TRIALS   = 2000     # trials per condition for the calibration libraries
CAL_SEED     = 1234     # independent of the agent libraries' seed
CAL_DURATION = 3.0

CONTEXT   = "Context_A"
OX_DRUG   = 0.3         # anchor reproducing the reported drug condition
OX_BASE   = 1.0         # anchor reproducing untreated baseline

N_BOOT    = 500         # bootstrap replicates for the realized CIs
BOOT_SEED = 0

DRUG_KEYS = ["sb_ox1r", "empa_ox2r", "dora"]


# ox_key comes from orexin_keys (see header).

# ---------------------------------------------------------------------------
# Step 0 - obtain calibration libraries
# ---------------------------------------------------------------------------
def load_libraries():
    """Return (libraries, duration_sec, provenance_note)."""
    if CAL_PKL.exists():
        with open(CAL_PKL, "rb") as f:
            d = pickle.load(f)
        return (d["libraries"], float(d.get("duration_sec", CAL_DURATION)),
                f"{CAL_PKL.name} (n_trials={d.get('n_trials', '?')}, "
                f"seed={d.get('seed', '?')})")

    g = globals()
    if "generator" in g and "OREXIN_MODELS" in g:
        print(f"calibration libraries not found; generating "
              f"{CAL_TRIALS} trials/condition (this takes a few minutes) ...")
        cal = {}
        for name in DRUG_KEYS:
            cal[name] = g["generator"].generate_library(
                g["OREXIN_MODELS"][name],
                orexin_levels=[OX_DRUG, OX_BASE],
                n_trials=CAL_TRIALS,
                duration_sec=CAL_DURATION,
                seed=CAL_SEED,
            )
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CAL_PKL, "wb") as f:
            pickle.dump({"libraries": cal, "duration_sec": CAL_DURATION,
                         "n_trials": CAL_TRIALS, "seed": CAL_SEED}, f)
        print(f"  written: {CAL_PKL}")
        return (cal, CAL_DURATION,
                f"{CAL_PKL.name} (n_trials={CAL_TRIALS}, seed={CAL_SEED})")

    print("!" * 78)
    print("WARNING: no calibration libraries and no generator in namespace.")
    print("Falling back to the agent library. Confidence intervals will be")
    print("wide enough to be uninformative. Run the generator cell first.")
    print("!" * 78)
    with open(FALLBACK_PKL, "rb") as f:
        d = pickle.load(f)
    return (d["libraries"], float(d.get("duration_sec", CAL_DURATION)),
            f"{FALLBACK_PKL.name} (FALLBACK - underpowered)")


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------
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


def pooled_rate_per_min(trials, count_key, duration_sec):
    total = sum(int(t.get(count_key, 0)) for t in trials)
    total_time = len(trials) * float(duration_sec)
    return (total / total_time) * 60.0 if total_time > 0 else np.nan


def pooled_p_couple(trials):
    tot_spw = sum(int(t.get("n_spw", 0)) for t in trials)
    tot_c = sum(int(t.get("n_spw_with_ripple", 0)) for t in trials)
    return (tot_c / tot_spw) if tot_spw > 0 else np.nan


def _concat_key(trials, key):
    parts = [_safe_np(t.get(key, [])) for t in trials]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts).astype(float) if parts else np.array([])


def summarize(trials, duration_sec):
    """
    All library statistics needed for the realized column.

    Skewed continuous quantities carry both a mean and a median estimator.
    The median is used in the table; the mean is retained for the audit.
    """
    amps = _concat_key(trials, "spw_amplitudes")
    sdur = _concat_key(trials, "spw_durations")
    rdur = _concat_key(trials, "swr_durations")
    lags = _concat_key(trials, "coupling_lags")
    rps = np.array(
        [t.get("mean_ripples_per_spw", np.nan) for t in trials], dtype=float
    )
    rps = rps[np.isfinite(rps) & (rps > 0)]
    return {
        "spw_rate":     pooled_rate_per_min(trials, "n_spw", duration_sec),
        "swr_rate":     pooled_rate_per_min(trials, "n_coupled", duration_sec),
        "p_coup":       pooled_p_couple(trials),
        "swr_freq":     _mean_or_nan(_concat_key(trials, "swr_frequencies")),
        "lag_sd":       _std_or_nan(lags),
        "rps":          float(np.mean(rps)) if len(rps) else np.nan,
        # skewed: median used in the table, mean kept for comparison
        "spw_amp":      _median_or_nan(amps),
        "spw_dur":      _median_or_nan(sdur),
        "swr_dur":      _median_or_nan(rdur),
        "spw_amp_mean": _mean_or_nan(amps),
        "spw_dur_mean": _mean_or_nan(sdur),
        "swr_dur_mean": _mean_or_nan(rdur),
    }


# Statistics estimated by the median rather than the mean
MEDIAN_STATS = {"spw_amp", "spw_dur", "swr_dur"}

# Statistics where a higher applied multiplier means a SMALLER realized
# dispersion, so the realized ratio must be inverted (precision = 1/spread).
INVERTED_STATS = {"lag_sd"}


def ratio(drug_stats, base_stats, stat):
    d, b = drug_stats[stat], base_stats[stat]
    if not (np.isfinite(d) and np.isfinite(b)) or b == 0 or d == 0:
        return np.nan
    return (b / d) if stat in INVERTED_STATS else (d / b)


def bootstrap_ratios(drug_trials, base_trials, duration_sec,
                     n_boot=N_BOOT, seed=BOOT_SEED):
    """Trial-level bootstrap. Returns {stat: (lo, hi)} at 95%."""
    rng = np.random.default_rng(seed)
    nd, nb = len(drug_trials), len(base_trials)
    acc = {}
    for _ in range(n_boot):
        d = [drug_trials[i] for i in rng.integers(0, nd, nd)]
        b = [base_trials[i] for i in rng.integers(0, nb, nb)]
        ds, bs = summarize(d, duration_sec), summarize(b, duration_sec)
        for stat in ds:
            acc.setdefault(stat, []).append(ratio(ds, bs, stat))
    out = {}
    for stat, vals in acc.items():
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        out[stat] = ((np.percentile(v, 2.5), np.percentile(v, 97.5))
                     if len(v) else (np.nan, np.nan))
    return out


# ---------------------------------------------------------------------------
# Row specification
# ---------------------------------------------------------------------------
ROWS = [
    dict(
        label="Sharp wave incidence", stat="spw_rate", provenance="Direct",
        reported_txt={"sb_ox1r": "-29 +/- 3%", "empa_ox2r": "-50 +/- 2%",
                      "dora": "-60 +/- 2%"},
        applied={"sb_ox1r": 0.71, "empa_ox2r": 0.50, "dora": 0.40},
    ),
    dict(
        label="Sharp wave amplitude", stat="spw_amp", provenance="Direct",
        reported_txt={"sb_ox1r": "-41 +/- 1%", "empa_ox2r": "-42 +/- 2%",
                      "dora": "n.s."},
        applied={"sb_ox1r": 0.59, "empa_ox2r": 0.58, "dora": 1.00},
    ),
    dict(
        label="Sharp wave duration", stat="spw_dur", provenance="Direct",
        reported_txt={"sb_ox1r": "-9 +/- 2%", "empa_ox2r": "-11 +/- 1%",
                      "dora": "n.s."},
        applied={"sb_ox1r": 0.91, "empa_ox2r": 0.89, "dora": 1.00},
    ),
    dict(
        label="Ripple duration", stat="swr_dur", provenance="Direct",
        reported_txt={"sb_ox1r": "n.s.", "empa_ox2r": "+12 +/- 3%",
                      "dora": "+26 +/- 5%"},
        applied={"sb_ox1r": 1.00, "empa_ox2r": 1.12, "dora": 1.26},
    ),
    dict(
        label="Intra-ripple frequency", stat="swr_freq", provenance="Direct",
        reported_txt={"sb_ox1r": "n.s.", "empa_ox2r": "n.s.", "dora": "n.s."},
        applied={"sb_ox1r": 1.00, "empa_ox2r": 1.00, "dora": 1.00},
    ),
    dict(
        label="SPW-ripple coupling probability", stat="p_coup",
        provenance="Derived from reported SPW:ripple ratio",
        reported_txt={"sb_ox1r": "SPW:ripple ratio -25%",
                      "empa_ox2r": "SPW:ripple ratio -25%",
                      "dora": "SPW:ripple ratio +29%"},
        applied={"sb_ox1r": 0.86, "empa_ox2r": 0.83, "dora": 1.29},
    ),
    dict(
        label="Ripple power", stat=None, provenance="Unconstrained; swept",
        reported_txt={"sb_ox1r": "not measured", "empa_ox2r": "not measured",
                      "dora": "not measured"},
        applied={"sb_ox1r": 0.70, "empa_ox2r": 0.60, "dora": 0.70},
    ),
    dict(
        label="Coupling-lag precision", stat="lag_sd",
        provenance="Unconstrained; realized effect <5%",
        reported_txt={"sb_ox1r": "not measured", "empa_ox2r": "not measured",
                      "dora": "not measured"},
        applied={"sb_ox1r": 1.00, "empa_ox2r": 0.80, "dora": 0.70},
    ),
]
# NOTE: the multi-ripple tendency row was removed in v3. The empirical
# ripples-per-SPW distribution is degenerate at 1, so the parameter has no
# effect. See the inertness check in the audit.

DRUGS = [
    ("sb_ox1r",   "SB-334867 (OX1R)"),
    ("empa_ox2r", "EMPA (OX2R)"),
    ("dora",      "TCS-1102 (DORA)"),
]

# Reported ripple incidence, as multiplier on baseline, with SEM
RIPPLE_INCIDENCE = {
    "sb_ox1r":   (0.55, 0.07),
    "empa_ox2r": (0.38, 0.05),
    "dora":      (0.51, 0.07),
}

# Reported change in the sharp wave : ripple ratio
RATIO_REPORTED = {"sb_ox1r": 0.75, "empa_ox2r": 0.75, "dora": 1.29}

# Internal null controls: applied multiplier is exactly 1.00, so the realized
# ratio should bracket 1.00. These calibrate the Monte Carlo noise floor.
NULL_CONTROLS = [
    ("dora",    "spw_amp",  "SPW amplitude, DORA"),
    ("dora",    "spw_dur",  "SPW duration, DORA"),
    ("sb_ox1r", "swr_dur",  "Ripple duration, SB-334867"),
    ("sb_ox1r", "lag_sd",   "Coupling-lag precision, SB-334867"),
]


def main():
    libraries, duration_sec, prov_note = load_libraries()

    base_key = f"{CONTEXT}_ox{ox_key(OX_BASE)}"
    drug_key = f"{CONTEXT}_ox{ox_key(OX_DRUG)}"

    # Per-model baseline: each drug model at its own untreated anchor
    base_trials = {m: libraries[m][base_key] for m, _ in DRUGS}
    drug_trials = {m: libraries[m][drug_key] for m, _ in DRUGS}

    base = {m: summarize(base_trials[m], duration_sec) for m, _ in DRUGS}
    drug = {m: summarize(drug_trials[m], duration_sec) for m, _ in DRUGS}

    print(f"bootstrapping realized CIs ({N_BOOT} replicates) ...")
    cis = {m: bootstrap_ratios(drug_trials[m], base_trials[m], duration_sec)
           for m, _ in DRUGS}

    records = []
    for row in ROWS:
        rec = {"Parameter": row["label"], "Provenance": row["provenance"]}
        for m, disp in DRUGS:
            app = row["applied"][m]
            rec[f"{disp} reported"] = row["reported_txt"][m]
            rec[f"{disp} applied"] = f"{app:.2f}"
            if row["stat"] is None:
                rec[f"{disp} realized"] = "--"
            else:
                r = ratio(drug[m], base[m], row["stat"])
                lo, hi = cis[m].get(row["stat"], (np.nan, np.nan))
                rec[f"{disp} realized"] = (
                    "--" if not np.isfinite(r)
                    else f"{r:.2f} [{lo:.2f}, {hi:.2f}]"
                )
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "table1_calibration.csv", index=False)
    with open(OUT_DIR / "table1_calibration.md", "w") as f:
        f.write(df.to_markdown(index=False))

    # -----------------------------------------------------------------------
    # Audit
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("TABLE 1 AUDIT  (v3: high-n calibration, median estimators)")
    print("=" * 78)
    print(f"source       : {prov_note}")
    print(f"baseline ref : each model's own ox={OX_BASE} anchor ({base_key})")
    print(f"drug anchor  : ox={OX_DRUG} ({drug_key})")
    print(f"trials/cell  : {len(drug_trials[DRUGS[0][0]])}")
    print(f"bootstrap    : {N_BOOT} replicates, trial-level resampling")
    print(f"estimator    : median for {sorted(MEDIAN_STATS)}, else mean/pooled")

    print("\n-- internal null controls (applied = 1.00 exactly) " + "-" * 26)
    print("   These set the Monte Carlo noise floor. Each should bracket 1.00.")
    floor = []
    for m, stat, lbl in NULL_CONTROLS:
        r = ratio(drug[m], base[m], stat)
        lo, hi = cis[m].get(stat, (np.nan, np.nan))
        ok = np.isfinite(lo) and lo <= 1.00 <= hi
        floor.append(abs(r - 1.0))
        print(f"   {lbl:<36} {r:.3f} [{lo:.3f}, {hi:.3f}] "
              f"{'OK' if ok else '<-- EXCLUDES 1.00'}")
    if floor:
        print(f"   observed noise floor (max |deviation|): {max(floor):.3f}")

    print("\n-- median vs mean estimator (skewed statistics) " + "-" * 29)
    print("   A multiplicative modulation scales both; divergence indicates")
    print("   finite-sample tail bias rather than a modelling difference.")
    for stat in ["spw_amp", "spw_dur", "swr_dur"]:
        for m, disp in DRUGS:
            r_med = ratio(drug[m], base[m], stat)
            r_mean = ratio(drug[m], base[m], stat + "_mean")
            print(f"   {stat:<10} {disp:<18} median={r_med:.3f}  "
                  f"mean={r_mean:.3f}  diff={r_mean - r_med:+.3f}")

    print("\n-- applied vs realized " + "-" * 54)
    for row in ROWS:
        if row["stat"] is None:
            print(f"  {row['label']:<32} not stored in library")
            continue
        for m, disp in DRUGS:
            app = row["applied"][m]
            r = ratio(drug[m], base[m], row["stat"])
            lo, hi = cis[m].get(row["stat"], (np.nan, np.nan))
            if not np.isfinite(r):
                continue
            covered = np.isfinite(lo) and lo <= app <= hi
            print(f"  {row['label']:<32} {disp:<18} applied={app:.2f}  "
                  f"realized={r:.2f} [{lo:.2f}, {hi:.2f}]"
                  f"{'' if covered else '  <-- CI EXCLUDES APPLIED'}")

    print("\n-- composite ripple incidence (REALIZED vs reported SEM) " + "-" * 19)
    for m, disp in DRUGS:
        r = ratio(drug[m], base[m], "swr_rate")
        lo, hi = cis[m].get("swr_rate", (np.nan, np.nan))
        target, sem = RIPPLE_INCIDENCE[m]
        inside = (target - sem) <= r <= (target + sem)
        print(f"  {disp:<18} realized {r:.3f} [{lo:.3f}, {hi:.3f}] | "
              f"reported {target:.2f} +/- {sem:.2f}  "
              f"{'OK' if inside else '<-- OUTSIDE SEM'}")

    print("\n-- coupling probability vs reported ratio " + "-" * 34)
    print("   Applied constants are corrected for the amplitude-dependent")
    print("   coupling term; the REALIZED value is what should match.")
    print("   A CI that also contains 1.00 is uninformative, not a pass.")
    for m, disp in DRUGS:
        r = ratio(drug[m], base[m], "p_coup")
        lo, hi = cis[m].get("p_coup", (np.nan, np.nan))
        tgt = RATIO_REPORTED[m]
        hits = np.isfinite(lo) and lo <= tgt <= hi
        null_in = np.isfinite(lo) and lo <= 1.00 <= hi
        note = "OK" if hits else "<-- MISSES"
        if hits and null_in:
            note += "  (but CI includes 1.00: underpowered)"
        print(f"  {disp:<18} realized {r:.3f} [{lo:.3f}, {hi:.3f}] | "
              f"reported {tgt:.2f}  {note}")

    print("\n-- multi-ripple inertness check " + "-" * 44)
    rps_all = _concat_key(base_trials["sb_ox1r"], "ripples_per_spw")
    rps_all = rps_all[rps_all > 0]
    if len(rps_all) == 0:
        print("   ripples_per_spw not stored; cannot check")
    elif np.all(rps_all == 1):
        print(f"   every SPW carries exactly one ripple (n={len(rps_all)}).")
        print("   Multi-ripple tendency is inert; row correctly omitted.")
    else:
        print(f"   P(>1 ripple) = {np.mean(rps_all > 1):.4f}, "
              f"mean = {rps_all.mean():.4f}")
        print("   Parameter is LIVE - reinstate the row in ROWS.")

    print("\n" + "=" * 78)
    print(f"written: {OUT_DIR / 'table1_calibration.csv'}")
    print(f"         {OUT_DIR / 'table1_calibration.md'}")
    return df


if __name__ == "__main__":
    df = main()
    print()
    print(df.to_string(index=False))
