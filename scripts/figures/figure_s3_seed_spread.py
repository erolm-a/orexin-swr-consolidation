"""
Supplementary Figure 3 - across-seed spread of DI and the floor.

Reads:  results/agent_model_results.pkl
Writes: figures/figS3_seed_spread.{png,pdf,tiff}

Supports the argument that the descending limb is not step-size
instability: replicate spread does not grow with OX2R.

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
# SUPPLEMENTARY FIGURE 3 — across-seed spread of DI, and the consolidation floor
#
# Supports the Discussion sentence that the descending limb is not step-size
# instability. An unstable optimiser produces seeds that diverge from one
# another, so across-seed spread would grow with OX2R and DI would fall below
# the no-consolidation floor. Neither happens.
#
#   A  every replicate, with the mean and its 95% interval, against the
#      no-consolidation floor
#   B  across-seed SD by OX2R with percentile intervals, and the ratio of the
#      SD at the top of the range to the SD at the optimum
#
# Reads agent_model_results.pkl. Standalone.
# =============================================================================
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PKL = FILES["results"]
OUTDIR = DIRS["figures"]
DI_KEY = "ox2r_di_arrays"
FLOOR_KEY = "no_consolidation_baseline"
N_BOOT = 10000
SEED = 42
TIFF_DPI, TIFF_COMPRESS = 600, "tiff_lzw"

print("=" * 74)
print("SUPPLEMENTARY FIGURE 3 — across-seed spread of DI")
print("=" * 74)

with open(PKL, "rb") as f:
    D = pickle.load(f)
if DI_KEY not in D:
    raise KeyError(f"'{DI_KEY}' not in the pickle. Keys: {sorted(D)}")

da = {float(k): np.asarray(v, float).ravel() for k, v in D[DI_KEY].items()}
OXS = np.array(sorted(da))
X = np.array([da[ox] for ox in OXS])              # levels x replicates
NREP = X.shape[1]
if len({len(da[ox]) for ox in OXS}) != 1:
    raise ValueError("Replicate counts differ between levels.")

mean = X.mean(axis=1)
sd = X.std(axis=1, ddof=1)
sem = sd / np.sqrt(NREP)
peak_i = int(np.argmax(mean))

# no-consolidation floor, however it was stored
floor = np.nan
if FLOOR_KEY in D:
    f = D[FLOOR_KEY]
    if isinstance(f, dict):
        f = f.get("di", f.get("mean", list(f.values())[0]))
    floor = float(np.mean(np.asarray(f, float)))
print(f"  levels     : {[f'{o:g}' for o in OXS]}")
print(f"  replicates : {NREP}")
print(f"  optimum    : OX2R = {OXS[peak_i]:g}, DI = {mean[peak_i]:.4f}")
print(f"  floor      : {floor:.4f}" if np.isfinite(floor) else
      f"  floor      : '{FLOOR_KEY}' not in the pickle; the floor line is omitted")

# ─── bootstrap: per-level SD, and the top-versus-optimum ratio ───────────────
rng = np.random.default_rng(SEED)
idx = rng.integers(0, NREP, size=(N_BOOT, NREP))
Xb = X[:, idx]                                     # levels x n_boot x replicates
sd_b = Xb.std(axis=2, ddof=1)                      # levels x n_boot
sd_ci = np.percentile(sd_b, [2.5, 97.5], axis=1)
ratio_b = sd_b[-1] / sd_b[peak_i]
ratio = sd[-1] / sd[peak_i]
r_ci = np.percentile(ratio_b, [2.5, 97.5])
p_increase = float((ratio_b > 1.0).mean())

# slope of SD on OX2R, as a single summary
slope_b = np.array([np.polyfit(OXS, sd_b[:, b], 1)[0] for b in range(N_BOOT)])
slope = float(np.polyfit(OXS, sd, 1)[0])
s_ci = np.percentile(slope_b, [2.5, 97.5])

print(f"\n  {'OX2R':<8}{'mean DI':<12}{'SD':<10}{'SD 95% CI':<22}{'SEM'}")
for i, ox in enumerate(OXS):
    print(f"  {ox:<8.2f}{mean[i]:<12.4f}{sd[i]:<10.4f}"
          f"{f'[{sd_ci[0, i]:.4f}, {sd_ci[1, i]:.4f}]':<22}{sem[i]:.4f}")

print(f"\n  SD at OX2R {OXS[-1]:g} / SD at the optimum {OXS[peak_i]:g} = {ratio:.3f} "
      f"[{r_ci[0]:.3f}, {r_ci[1]:.3f}]")
print(f"  P(ratio > 1) = {p_increase:.4f}")
print(f"  slope of SD on OX2R = {slope:+.5f} [{s_ci[0]:+.5f}, {s_ci[1]:+.5f}]")
if np.isfinite(floor):
    below = mean - floor
    print(f"  DI at OX2R {OXS[-1]:g} minus the floor = {below[-1]:+.4f}")
    print(f"  lowest DI on the sweep minus the floor = {below.min():+.4f} "
          f"(at OX2R {OXS[int(np.argmin(below))]:g})")

# ─── figure ──────────────────────────────────────────────────────────────────
C_MEAN, C_REP, C_FLOOR, C_SD = "#1a5fa8", "#9aa0a6", "#b1543a", "#1a5fa8"
C_T = "#1c1917"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.3))


def plabel(ax, s):
    ax.text(-0.14, 1.09, s, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", color=C_T)


# A: every replicate, the mean, and the floor
plabel(axA, "A")
for r in range(NREP):
    axA.plot(OXS, X[:, r], "-", color=C_REP, lw=0.5, alpha=0.30, zorder=1)
axA.fill_between(OXS, mean - 1.96 * sem, mean + 1.96 * sem, color=C_MEAN,
                 alpha=0.20, lw=0, zorder=2)
axA.plot(OXS, mean, "-o", color=C_MEAN, lw=2.0, ms=5, mfc="white", mew=1.4,
         zorder=4, label="mean and 95% interval")
if np.isfinite(floor):
    axA.axhline(floor, color=C_FLOOR, lw=1.4, ls="--", zorder=3,
                label=f"no-consolidation floor ({floor:.3f})")
axA.plot([], [], "-", color=C_REP, lw=0.8, alpha=0.6,
         label=f"individual replicates (n = {NREP})")
axA.set_xlabel("OX2R level")
axA.set_ylabel("Discrimination index")
axA.set_title("Every seed, against the floor", fontsize=12, loc="left", x=0.0)
axA.legend(frameon=False, fontsize=8.5, loc="lower center")

# B: across-seed SD
plabel(axB, "B")
axB.fill_between(OXS, sd_ci[0], sd_ci[1], color=C_SD, alpha=0.20, lw=0, zorder=1)
axB.plot(OXS, sd, "-o", color=C_SD, lw=2.0, ms=5, mfc="white", mew=1.4,
         zorder=3, label="across-seed SD, 95% interval")
axB.axvline(OXS[peak_i], color=C_T, lw=1.0, ls=":", zorder=2)
axB.text(OXS[peak_i], axB.get_ylim()[1], " optimum", fontsize=9, color=C_T,
         va="top", ha="left")
axB.set_ylim(0, max(sd_ci[1]) * 1.28)
axB.set_xlabel("OX2R level")
axB.set_ylabel("Across-seed SD of DI")
axB.set_title("Spread does not grow with OX2R", fontsize=12, loc="left", x=0.0)
axB.annotate(f"SD({OXS[-1]:g}) / SD({OXS[peak_i]:g}) = {ratio:.2f}\n"
             f"[{r_ci[0]:.2f}, {r_ci[1]:.2f}]",
             xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
             fontsize=9, color=C_T)
axB.legend(frameon=False, fontsize=8.5, loc="upper left")

fig.tight_layout()
OUTDIR.mkdir(parents=True, exist_ok=True)
png, pdf, tif = (OUTDIR / f"figure_S3_seed_variance.{e}"
                 for e in ("png", "pdf", "tiff"))
fig.savefig(png, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
try:
    fig.savefig(tif, dpi=TIFF_DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": TIFF_COMPRESS})
    _tif = f"{tif.name}  [{TIFF_DPI} dpi, {TIFF_COMPRESS}]"
except Exception as ex:
    _tif = f"TIFF not written: {ex}"
plt.show(); plt.close(fig)

print("\n" + "=" * 74)
print("CAPTION NUMBERS")
print("=" * 74)
print(f"  n = {NREP} replicates per level, common random numbers across levels")
print(f"  SD range across the sweep: {sd.min():.4f} to {sd.max():.4f}")
print(f"  SD at {OXS[-1]:g} relative to the optimum: {ratio:.2f} "
      f"[{r_ci[0]:.2f}, {r_ci[1]:.2f}], P(ratio > 1) = {p_increase:.3f}")
print(f"  slope of SD on OX2R: {slope:+.5f} [{s_ci[0]:+.5f}, {s_ci[1]:+.5f}]")
if np.isfinite(floor):
    print(f"  no-consolidation floor {floor:.4f}; DI at {OXS[-1]:g} is "
          f"{mean[-1]:.4f} ({mean[-1] - floor:+.4f} relative to the floor)")
print(f"\nSaved to {OUTDIR}")
for p in (png, pdf):
    print(f"  {p.name}")
print(f"  {_tif}")
print("=" * 74)
