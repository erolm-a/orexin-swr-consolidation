"""
Supplementary Figure 2 - learning-rule control.

Reads:  results/learning_rule_control.pkl   (analysis/learning_rule_control.py)
Writes: figures/figure_S2_learning_rule.{png,pdf,tiff}

The control answers one objection: that the inverted-U is an artefact of a
supervised delta rule rather than a property of the consolidation dynamics.
Only the readout differs between arms; task, replay plans, truncation, gating
and induction are identical, and all arms share random draws at each replicate
index.

  A   Discrimination index against OX2R, for the rules whose DI has a
      consistent sign. Bands are +/- 1 SEM over replicates. The optima of the
      delta and Hebbian arms are bracketed, with a paired-bootstrap interval on
      their difference.

  B   Alignment between the readout weights and the target memory vector, for
      all four rules, against the chance level for a random unit vector in the
      same number of dimensions.

  C   Curvature of each DI curve with its bootstrap interval. Negative is a
      peak, positive a trough. This is the panel that carries the claim.

Provenance is read from the pickle's `_meta` block and printed, so the figure
always states which run it was drawn from.

Per-replicate arrays
--------------------
Where the pickle carries them (`[rule]['replicates']`) this script draws SEM
bands, bootstraps the delta-Hebb optimum offset, and decides panel A membership
from a sign check rather than a hard-coded list. Where it does not - a control
written before those arrays were kept - it falls back to condition means with
no bands, prints which features were skipped, and says so on the figure.

Colour and legibility
---------------------
The four rules keep the palette used across Figures 1-5. Validated against the
colour checks, that palette has two known weaknesses: the Oja and Sanger hues
separate by only dE 6.5 under deuteranopia, and the delta grey falls below the
3:1 contrast floor. Both are permissible only with secondary encoding, so every
rule additionally carries its own marker shape, its own dash pattern and a
direct end-of-line label. Identity is therefore never carried by colour alone.
Changing the palette instead would have broken consistency with the other
figures, which matters more in a manuscript than in a standalone chart.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

PKL = DIRS["results"] / "learning_rule_control.pkl"
OUTDIR = DIRS["figures"]
STEM = "figure_S2_learning_rule"

RULES = ["delta", "hebb", "oja", "sanger"]
PANEL_A_FALLBACK = ["delta", "hebb", "oja"]   # used only without replicates

D_DIM_DEFAULT = 64
NREP_DEFAULT = 25

N_BOOT_OFFSET = 10_000
OFFSET_SEED = 20260925
SIGN_CONSISTENT = 0.9      # fraction of replicates one-signed to call DI usable

TIFF_DPI = 600

# --- palette: house hues, plus secondary encoding (see module docstring) ------
COL = {"delta": "#9aa0a6", "hebb": "#3b6ea5", "oja": "#b1543a", "sanger": "#3f8f6b"}
MRK = {"delta": "o", "hebb": "s", "oja": "^", "sanger": "D"}
DSH = {"delta": (0, ()), "hebb": (0, (5, 1.4)),
       "oja": (0, (1.4, 1.2)), "sanger": (0, (6, 1.3, 1.3, 1.3))}
SHORT = {"delta": "delta", "hebb": "Hebbian", "oja": "Oja", "sanger": "Sanger"}
LBL = {"delta": "Delta rule (supervised)",
       "hebb": "Reward-modulated Hebbian (global scalar)",
       "oja": "Oja (unsupervised, PC1)",
       "sanger": "Sanger (unsupervised, PC2)"}
C_T, C_REF, C_GRID = "#1c1917", "#9aa0a6", "#e7e5e4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.8, "axes.labelsize": 9.5,
    "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 120, "savefig.dpi": 300,
})


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

if not PKL.exists():
    sys.exit(f"{PKL} not found.\nRun: python analysis/learning_rule_control.py")

with open(PKL, "rb") as f:
    D = pickle.load(f)

meta = D.get("_meta", {})
D_DIM = int(meta.get("rate_dim", D_DIM_DEFAULT))
NREP = int(meta.get("n_reps", NREP_DEFAULT))
ALIGN_TARGET = meta.get("align_target", "mem_a")

print("=" * 78)
print("SUPPLEMENTARY FIGURE 2 - learning-rule control")
print("=" * 78)
if meta:
    print(f"  provenance : pipeline {meta.get('pipeline_version', '?')}, "
          f"{NREP} replicates, written {meta.get('created', '?')}")
    print(f"               alignment target '{ALIGN_TARGET}', d = {D_DIM}")
else:
    print(f"  no provenance block; assuming d = {D_DIM}, {NREP} replicates")

present = [r for r in RULES if r in D]
HAVE_REPS = all("replicates" in D[r] for r in present)
print(f"  rules      : {', '.join(present)}")
print(f"  replicates : {'stored' if HAVE_REPS else 'NOT stored - degraded figure'}")


def means(rule, metric):
    """(levels, condition means) from the stored summary."""
    mn = D[rule][metric]["means"]
    ox = np.array(sorted(float(k) for k in mn), float)
    return ox, np.array([float(mn[k]) for k in sorted(mn, key=float)], float)


def reps(rule, which):
    """(levels, array[n_levels, n_reps]) if the replicates were stored."""
    r = D[rule]["replicates"][which]
    ox = np.array(sorted(float(k) for k in r), float)
    return ox, np.array([r[k] for k in sorted(r, key=float)], float)


OXS = means(present[0], "stats_ALIGN")[0]


def sem_band(rule, which):
    """(mean, lo, hi) at +/- 1 SEM, or (mean, None, None) without replicates."""
    if not HAVE_REPS:
        key = "stats_DI" if which == "di" else "stats_ALIGN"
        return means(rule, key)[1], None, None
    _, a = reps(rule, which)
    m = a.mean(axis=1)
    s = a.std(axis=1, ddof=1) / np.sqrt(a.shape[1])
    return m, m - s, m + s


# --------------------------------------------------------------------------- #
# Which rules go in panel A
# --------------------------------------------------------------------------- #

def di_sign_consistent(rule) -> tuple[bool, float]:
    """
    A principal component's sign is undetermined, so DI under an unsupervised
    rule should be bimodal across seeds. If it is not, the sign is pinned by
    the encoding phase and DI is interpretable after all.
    """
    _, a = reps(rule, "di")
    frac = (a > 0).mean(axis=1)
    return bool(frac.min() > SIGN_CONSISTENT or frac.max() < 1 - SIGN_CONSISTENT), \
        float(np.mean(frac))


if HAVE_REPS:
    print("\n  DI sign check - fraction of replicates with DI > 0")
    print(f"    {'rule':<10}{'min':>8}{'max':>8}{'usable':>9}")
    panelA = []
    for r in present:
        if "stats_DI" not in D[r]:
            print(f"    {r:<10}{'-':>8}{'-':>8}{'no DI stored':>14}")
            continue
        _, a = reps(r, "di")
        frac = (a > 0).mean(axis=1)
        ok, _ = di_sign_consistent(r)
        print(f"    {r:<10}{frac.min():>8.2f}{frac.max():>8.2f}{str(ok):>9}")
        if ok:
            panelA.append(r)
    print("    membership decided from the data, not assumed")
else:
    panelA = [r for r in PANEL_A_FALLBACK if r in present and "stats_DI" in D[r]]
    print(f"\n  panel A membership assumed: {', '.join(panelA)}")

# --------------------------------------------------------------------------- #
# Chance alignment
# --------------------------------------------------------------------------- #

chance = float(np.sqrt(2.0 / (np.pi * D_DIM)))
chance_sd = chance * float(np.sqrt(np.pi / 2.0 - 1.0))
null_lo = chance - 1.96 * chance_sd / np.sqrt(NREP)
null_hi = chance + 1.96 * chance_sd / np.sqrt(NREP)

print(f"\n  chance E|cos| for d = {D_DIM}: {chance:.4f}")
print(f"  null band for an n = {NREP} mean: [{null_lo:.4f}, {null_hi:.4f}]")
print(f"\n  {'rule':<10}{'min ALIGN':>11}{'max ALIGN':>11}"
      f"{'max/chance':>12}  at chance?")
for r in present:
    _, m = means(r, "stats_ALIGN")
    inside = "YES" if (m.min() >= null_lo and m.max() <= null_hi) else "no"
    print(f"  {r:<10}{m.min():>11.4f}{m.max():>11.4f}"
          f"{m.max() / chance:>12.2f}  {inside}")

# --------------------------------------------------------------------------- #
# Paired bootstrap on the delta-Hebb optimum offset
# --------------------------------------------------------------------------- #

def paired_offset(rule_a="hebb", rule_b="delta"):
    """
    Bootstrap the difference in fitted optimum between two arms.

    Resamples REPLICATE INDICES and applies the same index set to both arms,
    which preserves the common random numbers they were generated with.
    """
    if not HAVE_REPS or rule_a not in present or rule_b not in present:
        return None
    ox, aa = reps(rule_a, "di")
    _, bb = reps(rule_b, "di")
    n = aa.shape[1]
    X = np.column_stack([np.ones_like(ox), ox, ox ** 2])
    rng = np.random.RandomState(OFFSET_SEED)

    def vtx(y):
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        if c[2] < 0:
            return float(np.clip(-c[1] / (2 * c[2]), ox[0], ox[-1]))
        return ox[-1] if c[1] > 0 else ox[0]

    d = np.empty(N_BOOT_OFFSET)
    for i in range(N_BOOT_OFFSET):
        k = rng.choice(n, n, True)
        d[i] = vtx(aa[:, k].mean(axis=1)) - vtx(bb[:, k].mean(axis=1))
    return {"point": vtx(aa.mean(axis=1)) - vtx(bb.mean(axis=1)),
            "ci_95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_negative": float(np.mean(d < 0))}


offset = paired_offset()
if offset:
    print(f"\n  Hebbian minus delta optimum: {offset['point']:+.4f} "
          f"[{offset['ci_95'][0]:+.4f}, {offset['ci_95'][1]:+.4f}]  "
          f"P(<0) = {offset['p_negative']:.3f}")
    print("  paired replicate bootstrap; common random numbers preserved")
else:
    p_d = D.get("delta", {}).get("stats_DI", {}).get("peak")
    p_h = D.get("hebb", {}).get("stats_DI", {}).get("peak")
    if p_d is not None and p_h is not None:
        print(f"\n  Hebbian minus delta optimum: {p_h - p_d:+.4f} "
              "(point estimate; the interval needs the replicates)")


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def plabel(ax, s, x=-0.17, y=1.10):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="left", color=C_T)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=C_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def draw_rule(ax, rule, which, x):
    """Draw one arm and return its end-point y, for the label pass."""
    m, lo, hi = sem_band(rule, which)
    if lo is not None:
        ax.fill_between(x, lo, hi, color=COL[rule], alpha=0.16, lw=0, zorder=2)
    ax.plot(x, m, color=COL[rule], lw=1.7, ls=DSH[rule], marker=MRK[rule],
            ms=4.6, mfc="white", mew=1.3, zorder=3, label=LBL[rule],
            clip_on=False)
    return m


def end_labels(ax, x_end, ends, min_gap_frac=0.055):
    """
    Place direct end-of-line labels, nudged apart where curves converge.

    Without this the labels collide wherever two arms finish at similar
    values, which is exactly where telling them apart matters most.
    """
    if not ends:
        return
    y0, y1 = ax.get_ylim()
    gap = min_gap_frac * (y1 - y0)
    order = sorted(ends.items(), key=lambda kv: kv[1])
    placed = {}
    prev = None
    for rule, y in order:
        yy = y if prev is None else max(y, prev + gap)
        placed[rule] = yy
        prev = yy
    # If the stack overflowed the top, slide the whole column back down.
    over = max(placed.values()) - (y1 - 0.01 * (y1 - y0))
    if over > 0:
        placed = {r: v - over for r, v in placed.items()}
    dx = 0.05 * (OXS[-1] - OXS[0])
    for rule, yy in placed.items():
        # a hairline connector where the label had to move off its curve
        if abs(yy - ends[rule]) > 0.15 * gap:
            ax.plot([x_end, x_end + dx], [ends[rule], yy], color=COL[rule],
                    lw=0.7, alpha=0.85, zorder=4, clip_on=False)
        ax.annotate(SHORT[rule], xy=(x_end + dx, yy), xycoords="data",
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=7.2, color=C_T, va="center", ha="left",
                    zorder=5, annotation_clip=False)


fig = plt.figure(figsize=(13.4, 4.15), facecolor="white")
gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.055, right=0.955,
                      top=0.86, bottom=0.30)

# ---- A: discrimination index ------------------------------------------------
axA = fig.add_subplot(gs[0, 0]); plabel(axA, "A"); tidy(axA)
_endsA = {r: draw_rule(axA, r, "di", OXS)[-1] for r in panelA}

pk = {r: float(D[r]["stats_DI"]["peak"]) for r in ("delta", "hebb")
      if r in present and "stats_DI" in D[r]}
if len(pk) == 2:
    xa, xb = sorted(pk.values())
    y0, y1 = axA.get_ylim(); span = y1 - y0
    axA.set_ylim(y0, y1 + 0.22 * span)
    # A bracket is only legible when the two optima are actually apart. When
    # they nearly coincide, one dotted line carries the location and the text
    # carries the number.
    sep = (xb - xa) / (OXS[-1] - OXS[0])
    if sep > 0.03:
        axA.axvspan(xa, xb, color=C_REF, alpha=0.20, lw=0, zorder=1)
        yb = y1 + 0.02 * span; h = 0.022 * span
        axA.plot([xa, xa, xb, xb], [yb - h, yb, yb, yb - h], color=C_T,
                 lw=0.9, zorder=4)
    else:
        axA.axvline((xa + xb) / 2, color=C_REF, lw=1.0, ls=":", zorder=1)
    if offset:
        txt = (f"optima {xa:.2f} / {xb:.2f}      "
               f"$\\Delta$ = {offset['point']:+.3f} "
               f"[{offset['ci_95'][0]:+.3f}, {offset['ci_95'][1]:+.3f}]")
    else:
        txt = f"optima {xa:.2f} / {xb:.2f}      $\\Delta$ = {xb - xa:+.3f}"
    axA.text(0.5, 0.99, txt, transform=axA.transAxes, fontsize=6.6,
             color=C_T, ha="center", va="top", zorder=5)


axA.set_xlabel("OX2R level")
axA.set_ylabel("Discrimination index")
axA.set_title("Memory outcome", fontsize=9.8, loc="left", fontweight="bold")
axA.set_xticks(OXS)
axA.set_xticklabels([f"{v:g}" for v in OXS], fontsize=7.4)
end_labels(axA, OXS[-1], _endsA)

# ---- B: alignment -----------------------------------------------------------
axB = fig.add_subplot(gs[0, 1]); plabel(axB, "B"); tidy(axB)
axB.axhspan(null_lo, null_hi, color=C_REF, alpha=0.26, lw=0, zorder=1)
axB.axhline(chance, color=C_REF, lw=0.9, ls="--", zorder=1)
_endsB = {r: draw_rule(axB, r, "align", OXS)[-1] for r in present}
axB.text(OXS[0], null_hi, f" chance, n = {NREP}", fontsize=6.6, color="#78716c",
         va="bottom", ha="left", zorder=5)
axB.set_xlabel("OX2R level")
_tgt = ("target memory vector" if ALIGN_TARGET == "mem_a"
        else "discriminative (A − B) axis")
axB.set_ylabel(f"|cos| with the {_tgt}")
axB.set_title("Information reaching the readout", fontsize=9.8, loc="left",
              fontweight="bold")
axB.set_ylim(0, None)
axB.set_xticks(OXS)
axB.set_xticklabels([f"{v:g}" for v in OXS], fontsize=7.4)
end_labels(axB, OXS[-1], _endsB)

# ---- C: curvature -----------------------------------------------------------
axC = fig.add_subplot(gs[0, 2]); plabel(axC, "C"); tidy(axC)
rows = [r for r in present if "stats_DI" in D[r]]
ys = np.arange(len(rows))[::-1]
lim = max(abs(np.array([D[r]["stats_DI"]["curvature"] for r in rows])).max(), 1e-3)
pad = 0.06 * lim
for y, r in zip(ys, rows):
    st = D[r]["stats_DI"]
    c = st["curvature"]
    axC.barh(y, c, height=0.52, color=COL[r], alpha=0.85,
             edgecolor="white", lw=1.2, zorder=3)
    # The value sits on the OPPOSITE side of zero from its bar, so it cannot
    # collide with the bar, another bar, or the rule labels on the left.
    lab = f"{c:+.4f}   P(q2≥0) = {st['p_nonneg']:.3f}"
    if c < 0:
        axC.text(pad, y, lab, fontsize=7, va="center", ha="left",
                 color=C_T, zorder=4)
    else:
        axC.text(-pad, y, lab, fontsize=7, va="center", ha="right",
                 color=C_T, zorder=4)
axC.axvline(0, color=C_T, lw=0.9, zorder=2)
axC.set_yticks(ys)
axC.set_yticklabels([SHORT[r] for r in rows], fontsize=8.5)
axC.set_xlabel("Quadratic coefficient of the DI curve")
axC.set_title("Peak or trough", fontsize=9.8, loc="left", fontweight="bold")
axC.set_xlim(-2.35 * lim, 2.35 * lim)
axC.set_ylim(-0.95, len(rows) - 0.35)
axC.annotate("", xy=(-1.3 * lim, -0.72), xytext=(-0.12 * lim, -0.72),
             arrowprops=dict(arrowstyle="->", lw=0.8, color="#a8a29e"))
axC.annotate("", xy=(1.3 * lim, -0.72), xytext=(0.12 * lim, -0.72),
             arrowprops=dict(arrowstyle="->", lw=0.8, color="#a8a29e"))
axC.text(-1.35 * lim, -0.72, "peak", fontsize=7, color="#78716c",
         ha="right", va="center")
axC.text(1.35 * lim, -0.72, "trough", fontsize=7, color="#78716c",
         ha="left", va="center")
axC.spines["left"].set_visible(False)
axC.tick_params(axis="y", length=0)

# ---- shared legend, so identity never rests on colour alone -----------------
handles = [Line2D([], [], color=COL[r], lw=1.7, ls=DSH[r], marker=MRK[r],
                  ms=4.6, mfc="white", mew=1.3, label=LBL[r]) for r in present]
fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
           frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, 0.055),
           columnspacing=1.8, handlelength=2.8)

note = (f"Bands: ± 1 SEM over {NREP} replicates."
        if HAVE_REPS else
        "Condition means; the stored control kept no per-replicate arrays, "
        "so no error bands, no interval on the optimum offset, and panel A "
        "membership is assumed rather than tested.")
fig.text(0.5, 0.005, note, ha="center", fontsize=6.8, color="#78716c",
         style="italic")

OUTDIR.mkdir(parents=True, exist_ok=True)
png, pdf = OUTDIR / f"{STEM}.png", OUTDIR / f"{STEM}.pdf"
fig.savefig(png, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
_tif = "TIFF not written"
try:
    tif = OUTDIR / f"{STEM}.tiff"
    fig.savefig(tif, dpi=TIFF_DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    _tif = tif.name
except Exception as ex:  # Pillow missing or too old for this keyword
    _tif = f"TIFF not written: {ex}   (pip install -U Pillow)"

try:
    from IPython.display import display
    display(fig)
except Exception:
    pass
plt.show()
plt.close(fig)

# --------------------------------------------------------------------------- #
# Caption numbers - read these into the caption rather than retyping them
# --------------------------------------------------------------------------- #

print("\n" + "=" * 78)
print("CAPTION NUMBERS")
print("=" * 78)
print(f"  replicates per condition        {NREP}")
print(f"  OX2R levels                     {', '.join(f'{v:g}' for v in OXS)}")
print(f"  readout dimension               {D_DIM}")
print(f"  chance |cos|                    {chance:.4f} "
      f"(null band {null_lo:.4f} to {null_hi:.4f})")
print(f"  panel A                         {', '.join(panelA)}")

print(f"\n  {'rule':<9}{'metric':<8}{'curvature':>12}{'P(q2>=0)':>11}"
      f"{'optimum':>10}{'95% CI':>18}{'P(interior)':>13}")
for r in present:
    for key, tag in (("stats_DI", "DI"), ("stats_ALIGN", "ALIGN")):
        if key not in D[r]:
            continue
        st = D[r][key]
        ci = f"[{st['peak_ci'][0]:.3f}, {st['peak_ci'][1]:.3f}]"
        print(f"  {r:<9}{tag:<8}{st['curvature']:>+12.4f}{st['p_nonneg']:>11.4f}"
              f"{st['peak']:>10.3f}{ci:>18}{st['p_interior']:>13.3f}")

if offset:
    print(f"\n  Hebbian minus delta optimum     {offset['point']:+.4f} "
          f"[{offset['ci_95'][0]:+.4f}, {offset['ci_95'][1]:+.4f}], "
          f"P(<0) = {offset['p_negative']:.3f}")

print("\n  Caption must state:")
if HAVE_REPS:
    print(f"    - bands are +/- 1 SEM over {NREP} replicates")
    print("    - the optimum offset interval is a paired replicate bootstrap")
    print("    - panel A membership follows the DI sign check, not assumption")
else:
    print("    - curves are condition means with no error bands")
    print("    - interval statistics come from the run that wrote the pickle")
for r in present:
    for key, tag in (("stats_DI", "DI"), ("stats_ALIGN", "ALIGN")):
        if key not in D[r]:
            continue
        st = D[r][key]
        lo, hi = st["peak_ci"]
        if abs(hi - lo) < 1e-9:
            print(f"    - {r} {tag}: the optimum CI is degenerate at "
                  f"{lo:.2f}, the sampled floor. Report it as censored, "
                  "not as precision.")
        elif lo <= OXS[0] + 1e-9 and hi >= OXS[-1] - 1e-9:
            print(f"    - {r} {tag}: the CI spans the whole sampled range, so "
                  "this arm supports no directional claim.")

print(f"\nSaved to {OUTDIR}")
for p in (png, pdf):
    print(f"  {p.name}")
print(f"  {_tif}")
print("=" * 78)
