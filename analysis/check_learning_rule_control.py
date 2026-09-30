"""
Is the stored learning-rule control compatible with the current pipeline?

The control was computed before the detector was revised (channel rule change,
reference sessions reanalysed, generator restricted to the training sessions).
Its numbers therefore rest on whatever synthetic library existed at the time.

The control's delta arm uses the same readout as the main experiment, on the
same task and the same OX2R grid. So if both rest on the same library, their
delta-rule DI curves should agree closely. If they do not, the library change
propagated and the control describes a different model from Figures 3-5.

Run with both pickles in place:

    python check_learning_rule_control.py

Reports a verdict. Nothing is modified.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

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

from orexin_paths import DIRS, FILES  # noqa: E402

CONTROL = DIRS["results"] / "learning_rule_control.pkl"

# Agreement thresholds on the delta-rule DI curve.
TOL_SHAPE = 0.05     # max |difference| in normalised curve shape
TOL_PEAK = 0.15      # max |difference| in peak location, OX2R units


def load(path: Path):
    if not path.exists():
        sys.exit(f"not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def main_delta_curve(D: dict) -> tuple[np.ndarray, np.ndarray]:
    """(levels, mean DI) for the main experiment's delta-rule sweep."""
    if "ox2r_di_arrays" not in D:
        sys.exit("results pickle has no 'ox2r_di_arrays'; cannot compare.")
    arrs = D["ox2r_di_arrays"]
    lv = np.array(sorted(float(k) for k in arrs), float)
    mn = np.array([np.mean(arrs[k] if k in arrs else arrs[str(k)])
                   for k in sorted(arrs, key=lambda x: float(x))], float)
    return lv, mn


def control_delta_curve(C: dict) -> tuple[np.ndarray, np.ndarray]:
    means = C["delta"]["stats_DI"]["means"]
    lv = np.array(sorted(float(k) for k in means), float)
    mn = np.array([means[k] for k in sorted(means, key=float)], float)
    return lv, mn


def norm(y: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1] so shape is compared independently of DI range."""
    r = y.max() - y.min()
    return (y - y.min()) / r if r > 0 else y * 0


def vertex(x: np.ndarray, y: np.ndarray) -> float:
    a, b, _ = np.polyfit(x, y, 2)
    return float(np.clip(-b / (2 * a), x.min(), x.max())) if a != 0 else float("nan")


def main() -> int:
    C = load(CONTROL)
    D = load(FILES["results"])

    print("=" * 74)
    print("LEARNING-RULE CONTROL — COMPATIBILITY WITH THE CURRENT PIPELINE")
    print("=" * 74)

    # ---- provenance ---------------------------------------------------------
    print("\n[1] Provenance recorded in the control")
    meta = [k for k in ("pipeline_version", "seed", "source_sessions",
                        "n_trials", "created") if k in C]
    print(f"    {meta if meta else 'none — no seed, session list, library or date'}")
    if not meta:
        print("    The control cannot be matched to a library by inspection, so")
        print("    the comparison below is the only available evidence.")

    # ---- grid ---------------------------------------------------------------
    cl, cy = control_delta_curve(C)
    ml, my = main_delta_curve(D)
    print("\n[2] OX2R grid")
    print(f"    control: {[f'{v:g}' for v in cl]}")
    print(f"    main   : {[f'{v:g}' for v in ml]}")
    shared = np.array(sorted(set(cl) & set(ml)), float)
    if len(shared) < 5:
        print(f"    [STOP] only {len(shared)} shared levels; cannot compare shape.")
        return 1
    print(f"    shared : {len(shared)} levels")

    ci = np.array([np.where(cl == v)[0][0] for v in shared])
    mi = np.array([np.where(ml == v)[0][0] for v in shared])
    cy_s, my_s = cy[ci], my[mi]

    # ---- shape --------------------------------------------------------------
    print("\n[3] Delta-rule DI curve, normalised shape")
    dn = np.abs(norm(cy_s) - norm(my_s))
    print(f"    {'ox':<8}{'control':>10}{'main':>10}{'|dshape|':>11}")
    for v, a, b, d in zip(shared, cy_s, my_s, dn):
        print(f"    {v:<8.2f}{a:>10.4f}{b:>10.4f}{d:>11.4f}")
    print(f"    max |dshape| = {dn.max():.4f}   (tolerance {TOL_SHAPE})")

    # ---- peak ---------------------------------------------------------------
    vc, vm = vertex(shared, cy_s), vertex(shared, my_s)
    dv = abs(vc - vm)
    print("\n[4] Fitted vertex")
    print(f"    control {vc:.3f}   main {vm:.3f}   |d| = {dv:.3f}"
          f"   (tolerance {TOL_PEAK})")

    # ---- sign ---------------------------------------------------------------
    cc = np.polyfit(shared, cy_s, 2)[0]
    cm = np.polyfit(shared, my_s, 2)[0]
    same_sign = (cc < 0) == (cm < 0)
    print("\n[5] Curvature sign")
    print(f"    control {cc:+.5f}   main {cm:+.5f}   "
          f"{'same' if same_sign else 'DIFFERENT'}")

    # ---- verdict ------------------------------------------------------------
    print("\n" + "=" * 74)
    if not same_sign:
        print("VERDICT: INCOMPATIBLE — the curvature sign differs. The control")
        print("describes a qualitatively different model. Do not use it.")
        return 1
    if dn.max() <= TOL_SHAPE and dv <= TOL_PEAK:
        print("VERDICT: COMPATIBLE. The delta arm of the control tracks the")
        print("current main experiment in shape, vertex and sign, so the library")
        print("revision did not materially propagate into this control. The")
        print("figure can be used, with its provenance stated in the caption.")
        return 0
    print("VERDICT: DIVERGENT. Same curvature sign, but shape or vertex moved")
    print("more than tolerance. The control rests on the earlier library. Either")
    print("re-run it against the current one, or state in the caption that it")
    print("was computed on the pre-revision library and report it as")
    print("qualitative support only.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
