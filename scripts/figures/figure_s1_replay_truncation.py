"""
Supplementary Figure 1 - task structure and replay truncation.

  A  the trace as a diagram: a shared trunk that forks into target and lure at
     the divergence dimension, with the replay event drawn under each segment
     in that segment's colour.
  B  what a replay event exposes as a function of where it terminates,
     computed with the model's own truncation rule rather than redrawn by hand.

How panel A is put together
---------------------------
ONE EVENT, ONE ENVELOPE. The three glyphs are a single sharp-wave ripple drawn
along the fork, not three separate events. envelope() is defined once over the
whole trace in dimension units: a fast rise to a peak inside the shared prefix,
then a slow decay. The branch glyphs are the decaying tail of that envelope,
which is why they start high at divergence and fall away. Drawing each segment
with its own fresh hump would say the opposite, that a new event begins at the
fork.

IDENTICAL BEFORE DIVERGENCE, DIFFERENT AFTER. One grey waveform runs under the
shared prefix: whichever trace is being replayed, the activity is the same and
nothing distinguishes them. Past divergence the target and lure glyphs carry
visibly different ripple patterns (F_TARGET against F_LURE), because that
difference is what the readout discriminates. Two identical squiggles in two
colours would say there is nothing to tell apart.

TRUNCATION. Both branch glyphs are cut at STOP_STEP: solid and filled up to it,
thin dashed outline past it. The dashed tail is content the readout never
receives, and its length comes from truncation(STOP_STEP), so it agrees with
bar STOP_STEP of panel B by construction.

COORDINATES. The branch boxes are drawn compressed (X_BRANCH..RATE_DIM standing
for dimensions N_SHARED..RATE_DIM), so dim_to_x() maps the envelope and the
termination point through that compression. Nothing in the branch half should
be placed by eye.

The waveforms are schematic rather than extracted LFP; Figure 1 carries the
real traces. The model assigns no different ripple dynamics to target and lure
either, and the two patterns differ in the drawing only to stand for different
replayed content. Both points belong in the caption and are printed at the end.

This script is standalone: it reimplements apply_truncation() instead of
importing the agent model, and asserts that the reimplementation matches the
constants it was given. Changing TASK_DIVERGENCE, RATE_DIM or SEQ_LEN in the
model means changing them here too; the assertions check internal consistency,
not agreement between files.

Writes: figures/figure_S1_task_structure.{png,pdf,tiff}

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

from orexin_paths import DIRS, ensure_dirs  # noqa: E402

ensure_dirs()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUTDIR = DIRS["figures"]
TIFF_DPI, TIFF_COMPRESS = 600, "tiff_lzw"

# ─── model constants, mirrored from the agent model ────────────────────────────────────────────────
RATE_DIM = 64
SEQ_LEN = 16
TASK_DIVERGENCE = 0.6
TRUNCATION_GAMMA = 1.0

N_SHARED = int(RATE_DIM * TASK_DIVERGENCE)                  # 38
N_UNIQUE = RATE_DIM - N_SHARED                              # 26
DIVERGE_STEP = max(1, int(SEQ_LEN * N_SHARED / RATE_DIM))   # 9
DIMS_PER_STEP = RATE_DIM / SEQ_LEN                          # 4

STOP_STEP = 13                   # where the example event in A terminates


def truncation(n_steps, gamma=TRUNCATION_GAMMA):
    """apply_truncation(), reimplemented: (n_shared_vis, n_unique_vis, q)."""
    if n_steps <= DIVERGE_STEP:
        n_sv = max(1, int(N_SHARED * (n_steps / DIVERGE_STEP)))
        return n_sv, 0, 0.0
    q = (n_steps - DIVERGE_STEP) / max(1, SEQ_LEN - DIVERGE_STEP)
    q = min(1.0, max(0.0, q))
    if gamma != 1.0:
        q = q ** gamma
    return N_SHARED, int(N_UNIQUE * q), q


assert DIVERGE_STEP == 9 and N_SHARED == 38 and N_UNIQUE == 26
assert truncation(DIVERGE_STEP)[2] == 0.0
assert abs(truncation(SEQ_LEN)[2] - 1.0) < 1e-12
assert DIVERGE_STEP < STOP_STEP < SEQ_LEN, \
    "STOP_STEP must sit past divergence and short of the end, or A shows nothing"

steps = np.arange(1, SEQ_LEN + 1)
vis = np.array([truncation(s) for s in steps], float)     # shared, unique, q

print("=" * 74)
print("SUPPLEMENTARY FIGURE 1 — task structure and replay truncation")
print("=" * 74)
print(f"  dimensions {RATE_DIM} = {N_SHARED} shared + {N_UNIQUE} trace-specific")
print(f"  steps {SEQ_LEN}, {DIMS_PER_STEP:g} dimensions per step, "
      f"divergence at step {DIVERGE_STEP}")
print(f"\n  {'stop step':<11}{'shared vis':<13}{'unique vis':<13}{'q'}")
for s, (sv, uv, q) in zip(steps, vis):
    mark = "  <- divergence" if s == DIVERGE_STEP else \
           "  <- example event in A" if s == STOP_STEP else ""
    print(f"  {s:<11.0f}{sv:<13.0f}{uv:<13.0f}{q:.3f}{mark}")

# ─── style, matched to Figures 1-5 ───────────────────────────────────────────
C_SHARED, C_TARGET, C_LURE = "#9aa0a6", "#3b6ea5", "#b1543a"
C_SHARED_D = "#6b7280"          # grey ripple, dark enough to read on the fill
C_T, C_GRID = "#1c1917", "#d4d4d8"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white"})

fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(8.8, 8.4), gridspec_kw={"height_ratios": [1.36, 1.0],
                                           "hspace": 0.18})


def plabel(ax, s, dx=-0.105, dy=1.07):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=16, fontweight="bold",
            va="top", color=C_T)


# ══ A: the trace as a diagram, with the event drawn along the fork ═══════════
Y_TOP, Y_BOT = 3.20, -2.58
axA.set_xlim(-3, RATE_DIM + 17)
axA.set_ylim(Y_BOT, Y_TOP)
axA.axis("off")
plabel(axA, "A")

BH = 0.42                                  # box half-height
X_BRANCH = N_SHARED + 3.2                  # left edge of the branch boxes
Y_TARGET_BOX, Y_TRUNK_BOX, Y_LURE_BOX = 1.90, 0.30, -1.32
Y_TARGET_SWR, Y_TRUNK_SWR, Y_LURE_SWR = 0.98, -0.62, -2.22

axA.plot([N_SHARED, N_SHARED], [-2.40, 2.56], color=C_T, lw=1.0, ls="--",
         zorder=1)
axA.text(N_SHARED + 0.9, 2.42, "divergence", ha="left", va="center",
         fontsize=9.5, color=C_T)

axA.add_patch(Rectangle((0, Y_TRUNK_BOX - BH), N_SHARED, 2 * BH,
                        fc=C_SHARED, ec="white", lw=1.2, zorder=3))
axA.text(N_SHARED / 2, Y_TRUNK_BOX, f"shared prefix  ({N_SHARED} dim)",
         ha="center", va="center", fontsize=10, color="white",
         fontweight="bold", zorder=4)

for ybox, col, lab in ((Y_TARGET_BOX, C_TARGET, "target"),
                       (Y_LURE_BOX, C_LURE, "lure")):
    # kept flat (rad 0.14) and behind the glyphs: a bulging arc crosses them
    axA.add_patch(FancyArrowPatch((N_SHARED, Y_TRUNK_BOX), (X_BRANCH, ybox),
                                  arrowstyle="-", mutation_scale=1,
                                  connectionstyle="arc3,rad=0.14",
                                  lw=1.3, color=col, zorder=1))
    axA.add_patch(Rectangle((X_BRANCH, ybox - BH), RATE_DIM - X_BRANCH, 2 * BH,
                            fc=col, ec="white", lw=1.2, zorder=3))
    axA.text((X_BRANCH + RATE_DIM) / 2, ybox, f"{lab}  ({N_UNIQUE} dim)",
             ha="center", va="center", fontsize=10, color="white",
             fontweight="bold", zorder=4)

# dimension axis, carrying the dimensions-per-step conversion that panel B needs
y = 2.70
axA.plot([0, RATE_DIM], [y, y], color=C_T, lw=1.0, zorder=2)
for t in np.arange(0, RATE_DIM + 1, DIMS_PER_STEP):
    axA.plot([t, t], [y - 0.07, y + 0.07], color=C_GRID, lw=0.8, zorder=1)
for t in (0, N_SHARED, RATE_DIM):
    axA.plot([t, t], [y - 0.13, y + 0.13], color=C_T, lw=1.2, zorder=2)
    axA.text(t, y + 0.20, f"{t:.0f}", ha="center", va="bottom", fontsize=9.5,
             color=C_T)
axA.text(RATE_DIM + 1.4, y, f"Dimension\n({DIMS_PER_STEP:g} per replay step)",
         ha="left", va="center", fontsize=9.5, color=C_T, linespacing=1.3)

# ─── the event ───────────────────────────────────────────────────────────────
ENV_PEAK_DIM, ENV_RISE, ENV_DECAY = 20.0, 9.0, 46.0
F_SHARED, F_TARGET, F_LURE = 0.30, 0.24, 0.38     # ripple cycles per dimension
SWR_HALF = 0.40


def envelope(d):
    """One envelope for the whole event: rise to a peak inside the shared
    prefix, then decay. The branch glyphs are this tail, not new bursts."""
    d = np.asarray(d, float)
    return np.where(d < ENV_PEAK_DIM,
                    np.exp(-0.5 * ((d - ENV_PEAK_DIM) / ENV_RISE) ** 2),
                    np.exp(-(d - ENV_PEAK_DIM) / ENV_DECAY))


def dim_to_x(d):
    """Dimension -> x. The branch boxes are drawn compressed, so the branch
    half of the trace has its own mapping."""
    d = np.asarray(d, float)
    return np.where(d <= N_SHARED, d,
                    X_BRANCH + (d - N_SHARED) / N_UNIQUE * (RATE_DIM - X_BRANCH))


def swr(ax, d0, d1, y0, col, ripple_col, freq, half=SWR_HALF, d_stop=None):
    """The event between dimensions d0 and d1, drawn at height y0. Past d_stop
    the event has ended: thin dashed outline, no fill."""
    d = np.linspace(d0, d1, 900)
    x = dim_to_x(d)
    env = envelope(d)
    top = y0 + 0.92 * half * env
    rip = y0 + half * np.sin(2 * np.pi * freq * (d - d0)) * env

    live = np.ones_like(d, dtype=bool) if d_stop is None else d <= d_stop
    ax.fill_between(x[live], y0, top[live], color=col, alpha=0.30, lw=0, zorder=2)
    ax.plot(x[live], top[live], color=col, lw=1.4, zorder=3)
    ax.plot(x[live], rip[live], color=ripple_col, lw=0.75, alpha=0.95, zorder=4)
    if not live.all():
        j = max(0, int(np.argmax(~live)) - 1)
        ax.plot(x[j:], top[j:], color=col, lw=0.8, ls=(0, (3, 2)), alpha=0.45,
                zorder=3)
        ax.plot(x[j:], rip[j:], color=ripple_col, lw=0.5, alpha=0.30, zorder=3)
    xe = float(dim_to_x(d1 if d_stop is None else min(d1, d_stop)))
    ax.plot([xe, xe], [y0 - 0.24, y0 + 0.44], color=C_T, lw=1.1, zorder=6)


N_UNIQUE_VIS = truncation(STOP_STEP)[1]
D_STOP = N_SHARED + N_UNIQUE_VIS
X_STOP = float(dim_to_x(D_STOP))

swr(axA, 0, N_SHARED, Y_TRUNK_SWR, C_SHARED, C_SHARED_D, F_SHARED)
swr(axA, N_SHARED, RATE_DIM, Y_TARGET_SWR, C_TARGET, C_TARGET, F_TARGET,
    d_stop=D_STOP)
swr(axA, N_SHARED, RATE_DIM, Y_LURE_SWR, C_LURE, C_LURE, F_LURE, d_stop=D_STOP)

axA.annotate(f"event ends (step {STOP_STEP})", xy=(X_STOP, Y_TARGET_SWR - 0.24),
             xytext=(X_STOP + 1.0, Y_TARGET_SWR - 0.60), fontsize=9, color=C_T,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.8, color=C_T, shrinkA=0,
                             shrinkB=2))
for ys in (Y_TARGET_SWR, Y_LURE_SWR):
    axA.text(RATE_DIM + 1.4, ys, "never replayed", ha="left", va="center",
             fontsize=8.8, color="0.45")

# ══ B: what an event exposes, by where it stops ══════════════════════════════
plabel(axB, "B", dx=-0.105, dy=1.10)
axB.bar(steps, vis[:, 0], width=0.78, color=C_SHARED, label="shared dimensions",
        zorder=2)
axB.bar(steps, vis[:, 1], bottom=vis[:, 0], width=0.78, color=C_TARGET,
        label="trace-specific dimensions", zorder=2)
axB.axvline(DIVERGE_STEP + 0.5, color=C_T, lw=1.1, ls="--", zorder=3)
axB.text(DIVERGE_STEP + 0.35, RATE_DIM * 0.97, "divergence step ",
         ha="right", va="top", fontsize=9.5, color=C_T)
axB.set_xticks(steps)
axB.set_xlim(0.4, SEQ_LEN + 0.6)
axB.set_ylim(0, RATE_DIM * 1.04)
axB.set_xlabel("Step at which the replay event terminates")
axB.set_ylabel("Dimensions exposed to the readout")
axB.legend(loc="upper left", frameon=False, fontsize=9)

axq = axB.twinx()
axq.plot(steps, vis[:, 2], "-o", color=C_LURE, lw=1.8, ms=4.5, mfc="white",
         mew=1.3, zorder=4, label="replay quality q")
axq.set_ylim(0, 1.04)
axq.set_ylabel("Replay quality q", color=C_LURE)
axq.tick_params(axis="y", colors=C_LURE)
axq.spines["right"].set_visible(True)
axq.spines["right"].set_color(C_LURE)
axq.spines["top"].set_visible(False)
axq.legend(loc="lower right", frameon=False, fontsize=9)

fig.subplots_adjust(left=0.115, right=0.88, top=0.95, bottom=0.075)
OUTDIR.mkdir(parents=True, exist_ok=True)
png, pdf, tif = (OUTDIR / f"figure_S1_task_structure.{e}"
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
print(f"  A  {RATE_DIM} dimensions = {N_SHARED} shared + {N_UNIQUE} "
      f"trace-specific; {SEQ_LEN} steps at {DIMS_PER_STEP:g} dimensions each;")
print(f"     divergence at step {DIVERGE_STEP} (dimension {N_SHARED})")
print(f"     the example event ends at step {STOP_STEP}: q = "
      f"{truncation(STOP_STEP)[2]:.3f}, {N_UNIQUE_VIS} of {N_UNIQUE} "
      f"trace-specific dimensions reach the readout;")
print(f"     the remaining {N_UNIQUE - N_UNIQUE_VIS} are the dashed tail")
print(f"  B  q = 0 through step {DIVERGE_STEP}; q rises to 1 across steps "
      f"{DIVERGE_STEP + 1} to {SEQ_LEN}")
print(f"     an event stopping at step {DIVERGE_STEP} exposes "
      f"{truncation(DIVERGE_STEP)[0]} shared and 0 trace-specific dimensions")
print(f"     step {DIVERGE_STEP + 1}: q = {truncation(DIVERGE_STEP + 1)[2]:.3f}, "
      f"{truncation(DIVERGE_STEP + 1)[1]} trace-specific dimensions")
print(f"  truncation exponent gamma = {TRUNCATION_GAMMA:g} (core model)")
print("\n  CAPTION MUST STATE:")
print("   - The waveforms in A are schematic. The recorded events are in")
print("     Figure 1; nothing in A is extracted LFP.")
print("   - The three glyphs are one replay event drawn along the fork, not")
print("     three separate events. Replay always begins at dimension 0; events")
print("     differ in where they stop, which is what B quantifies.")
print("   - The target and lure waveforms are drawn as different patterns to")
print("     stand for different replayed content. The model does not assign")
print("     different ripple dynamics to the two traces.")
print(f"\nSaved to {OUTDIR}")
for p in (png, pdf):
    print(f"  {p.name}")
print(f"  {_tif}")
print("=" * 74)

# =============================================================================
