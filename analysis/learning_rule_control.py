"""
Learning-rule control.

Runs the consolidation sweep under four readout rules and stores the curve
statistics that Supplementary Figure 2 draws. Produces
`results/learning_rule_control.pkl`.

This replaces an earlier standalone script that was not kept with the
pipeline. Its output was computed against a previous synthetic library and no
longer matches the current one, so it is regenerated here rather than reused.

What the control is for
-----------------------
The objection it answers: the inverted-U might be an artefact of a supervised
delta rule rather than a property of the consolidation dynamics. So the same
sweep is run with the learning rule swapped out, and the curvature compared.

Only the readout changes between arms. The task, memory pair, replay plans,
event statistics, truncation, gating and induction are identical, and every arm
shares the same random draws at each replicate index.

The four rules
--------------
  delta    supervised error correction: dw = lr * (target - pred) * x
           This is RateReadout, the rule used everywhere else in the pipeline.

  hebb     reward-modulated Hebbian: dw = lr * R * x, with R = 2*target - 1 a
           global scalar rather than a per-unit error. No error term, so no
           self-limiting behaviour; magnitudes run larger than delta.

  oja      Oja's rule: dw = lr * y * (x - y*w), y = w.x
           Unsupervised. Converges toward the leading principal component of
           the replayed inputs. The target is ignored.

  sanger   Generalized Hebbian (Sanger) with two components, reading out the
           SECOND. Unsupervised, and deliberately a near-null arm: PC2 of the
           replayed inputs carries little about the discriminative direction.

Metrics
-------
  DI       discrimination index, exactly as the main experiment computes it:
           mean predict(A) - mean predict(B) over 20 noisy probes.

  ALIGN    |cos| between the readout weight vector and the target memory
           vector. Chance for a random unit vector in d dimensions is
           sqrt(2 / (pi * d)), which is 0.0997 at RATE_DIM = 64.

DI is not stored for `sanger`: the sign of a principal component is
undetermined, so a signed difference of predictions has no consistent meaning
for that arm. Supplementary Figure 2 expects it to be absent.

Per-replicate arrays are stored alongside the summary statistics, under
`[rule]['replicates']`. The earlier control kept summaries only, which is why
the figure drawn from it had no error bands, no interval on the hebb-delta
optimum offset, and a hard-coded panel membership instead of a sign check.

Reconstruction choices
----------------------
Two details of the original were not recoverable and are set here. Both are
switches at the top of the file, so either can be revisited.

  ALIGN_TARGET = "discriminative"
      Alignment is measured against the A - B axis, the direction that
      separates the two memories. The alternative is the rewarded memory
      vector itself. Both are computed and stored on every run, and the
      printout compares them, because the choice changes the magnitudes for
      the unsupervised arms. The earlier figure's axis label named the A - B
      axis, and it is the quantity the panel is about.

  READOUT_SEED_OFFSET = 7777
      Every rule starts from the weight vector RateConsolidationAgent would
      have given the delta rule at that replicate, so the arms differ by rule
      and not by initialisation.

Validation
----------
With the delta readout this sweep is the main experiment. Run with
`--validate` it reproduces the sweep and checks the result against
`ox2r_di_arrays` in the results pickle. They must agree to floating-point
tolerance; if they do not, something in the pipeline has moved and the control
should not be trusted until it is understood.

Usage
-----
    python analysis/learning_rule_control.py --validate     # check, then run
    python analysis/learning_rule_control.py                # run
    python analysis/learning_rule_control.py --nreps 25     # fewer replicates
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
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

from orexin_paths import DIRS, FILES, PIPELINE_VERSION, ensure_dirs, stamp  # noqa: E402

# --------------------------------------------------------------------------- #
# Reconstruction switches — see the module docstring
# --------------------------------------------------------------------------- #

ALIGN_TARGET = "discriminative"   # "discriminative" or "mem_a"; both are
                                  # computed, this one is what the figure plots
_OTHER_TARGET = "mem_a" if ALIGN_TARGET == "discriminative" else "discriminative"
READOUT_SEED_OFFSET = 7777

RULES = ["delta", "hebb", "oja", "sanger"]
RULES_WITH_DI = ["delta", "hebb", "oja"]     # sanger's DI has no consistent sign

N_BOOT_CURV = 10_000
N_BOOT_PEAK = 5_000
STAT_SEED = 42

OUT = DIRS["results"] / "learning_rule_control.pkl"


# --------------------------------------------------------------------------- #
# Readouts
# --------------------------------------------------------------------------- #

class _BaseReadout:
    """Shared initialisation and sigmoid, so arms differ only in update()."""

    def __init__(self, dim, lr=0.3, seed=42):
        rng = np.random.RandomState(seed)
        self.w = (rng.randn(dim) * np.sqrt(2.0 / dim)).astype(np.float32)
        self.base_lr = lr
        self.dim = dim

    def predict(self, x):
        z = float(np.dot(self.w, x))
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def update(self, x, target, lr=None):  # pragma: no cover - overridden
        raise NotImplementedError


class HebbReadout(_BaseReadout):
    """Reward-modulated Hebbian. The modulator is a global scalar, not an error."""

    def update(self, x, target, lr=None):
        reward = 2.0 * float(target) - 1.0        # +1 rewarded, -1 unrewarded
        self.w += (lr if lr is not None else self.base_lr) * reward * x


class OjaReadout(_BaseReadout):
    """Oja's rule. Unsupervised; converges toward PC1 of the replayed inputs."""

    def update(self, x, target, lr=None):
        step = lr if lr is not None else self.base_lr
        y = float(np.dot(self.w, x))
        self.w += step * y * (x - y * self.w)


class SangerReadout(_BaseReadout):
    """
    Generalized Hebbian (Sanger) with two components; the readout is PC2.

    w1 tracks the leading component and is kept only to deflate the input for
    w2, which is what predict() and the alignment use.
    """

    def __init__(self, dim, lr=0.3, seed=42):
        super().__init__(dim, lr=lr, seed=seed)
        rng = np.random.RandomState(seed + 1)
        self.w1 = (rng.randn(dim) * np.sqrt(2.0 / dim)).astype(np.float32)

    def update(self, x, target, lr=None):
        step = lr if lr is not None else self.base_lr
        y1 = float(np.dot(self.w1, x))
        y2 = float(np.dot(self.w, x))
        self.w1 += step * y1 * (x - y1 * self.w1)
        self.w += step * y2 * (x - y1 * self.w1 - y2 * self.w)


READOUTS = {"hebb": HebbReadout, "oja": OjaReadout, "sanger": SangerReadout}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _abscos(w, t):
    nw, nt = np.linalg.norm(w), np.linalg.norm(t)
    if nw < 1e-12 or nt < 1e-12:
        return float("nan")
    return float(abs(np.dot(w, t) / (nw * nt)))


def alignments(w, mem_a, mem_b):
    """
    Both candidate alignment targets, from the same weight vector.

    The original addendum is gone and its two plausible targets give different
    answers for the unsupervised arms, so both are computed and stored. The
    earlier figure labelled its y axis "Alignment with A-B axis", which points
    at `discriminative`; the recovered magnitudes fit `mem_a` better. Rather
    than guess, measure both and decide from the printout.

      mem_a           |cos| with the rewarded memory vector
      discriminative  |cos| with the A - B axis, the direction that actually
                      separates the two memories
    """
    return {"mem_a": _abscos(w, mem_a.vector),
            "discriminative": _abscos(w, mem_a.vector - mem_b.vector)}


def chance_alignment(dim: int) -> tuple[float, float]:
    """Mean and sd of |cos| for a random unit vector in `dim` dimensions."""
    c = float(np.sqrt(2.0 / (np.pi * dim)))
    return c, c * float(np.sqrt(np.pi / 2.0 - 1.0))


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def run_rule_sweep(M, rule: str, oxs, nreps: int):
    """
    One arm of the control.

    Mirrors run_paired_sweep exactly — same task, same memory pair, same trial
    indices, same replay plans, same seeds — and changes only the readout.
    Returns {'di': {ox: array}, 'align': {ox: array}}.
    """
    task = M.TaskConfig.MWM(divergence=M.TASK_DIVERGENCE)
    di = {ox: [] for ox in oxs}
    al = {ox: [] for ox in oxs}          # ALIGN_TARGET, the one plotted
    ald = {ox: [] for ox in oxs}         # the other target, for comparison

    ref = M._get_trials(M.SWP_LIBRARY, M.MODEL_NAME_LIB, M.CONTEXT_LIB, 1.0)

    for rep in range(nreps):
        seed = rep * 1000
        ma, mb = M.RateMemory.create_pair(task.seq_len, 16, task.overlap,
                                          task.divergence_point, seed)
        ti = M._sample_crn(np.random.RandomState(seed + 123),
                           M.N_TRIAL_BLOCKS, len(ref))

        for ox in oxs:
            plan = M.build_replay_plan_ox(ox, task.seq_len, 16,
                                          np.random.RandomState(seed + 123),
                                          np.random.RandomState(seed + 500), ti)
            ag = M.RateConsolidationAgent(task.seq_len, 16, seed=seed)
            if rule != "delta":
                ag.readout = READOUTS[rule](ag.dim, lr=ag.base_lr,
                                            seed=seed + READOUT_SEED_OFFSET)

            # encode_ox2r=1.0 in the main experiment: encoding is held at
            # baseline so the sweep is consolidation-side only.
            ag.encode(ma, task.enc_str, task.enc_trials, ox2r=1.0)
            ag.encode(mb, task.enc_str * 0.3, max(1, task.enc_trials // 3), ox2r=1.0)
            ag.consolidate(plan, ox, M.CORE_SWITCHES, diverge_idx=M.DIVERGE_IDX)

            di[ox].append(float(ag.test(ma, mb)["discrimination_index"]))
            a = alignments(ag.readout.w, ma, mb)
            al[ox].append(a[ALIGN_TARGET])
            ald[ox].append(a[_OTHER_TARGET])

    return {"di": {ox: np.asarray(di[ox], float) for ox in oxs},
            "align": {ox: np.asarray(al[ox], float) for ox in oxs},
            "align_other": {ox: np.asarray(ald[ox], float) for ox in oxs}}


def curve_stats(M, da, oxs) -> dict:
    """The six fields Supplementary Figure 2 reads, from the pipeline's own stats."""
    curv = M.test_curvature(oxs, da, n_boot=N_BOOT_CURV, seed=STAT_SEED)
    peak = M.bootstrap_continuous_peak(da, oxs, n_boot=N_BOOT_PEAK, seed=STAT_SEED)
    interior = M.bootstrap_peak_location(da, oxs, n_boot=N_BOOT_PEAK, seed=STAT_SEED)
    return {
        "curvature": float(curv["curvature"]),
        "p_nonneg": float(curv["boot_sign_prob_nonneg"]),
        "peak": float(peak["mean"]),
        "peak_ci": [float(peak["ci_95"][0]), float(peak["ci_95"][1])],
        "p_interior": float(interior["p_interior"]),
        "means": {float(ox): float(np.mean(da[ox])) for ox in oxs},
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_delta(M, arm, oxs) -> bool:
    """The delta arm is the main experiment; it must reproduce the stored DI."""
    print("\n" + "-" * 74)
    print("VALIDATION — delta arm against the stored main experiment")
    print("-" * 74)

    if not FILES["results"].exists():
        print(f"  results pickle not found at {FILES['results']}; skipping.")
        return True
    with open(FILES["results"], "rb") as f:
        D = pickle.load(f)
    if "ox2r_di_arrays" not in D:
        print("  results pickle has no 'ox2r_di_arrays'; skipping.")
        return True

    stored = D["ox2r_di_arrays"]
    worst = 0.0
    print(f"  {'ox':<8}{'stored':>12}{'recomputed':>14}{'|d|':>12}")
    ok = True
    for ox in oxs:
        key = ox if ox in stored else str(ox)
        if key not in stored:
            print(f"  {ox:<8}  not in the stored sweep")
            ok = False
            continue
        s = float(np.mean(stored[key]))
        r = float(np.mean(arm["di"][ox]))
        d = abs(s - r)
        worst = max(worst, d)
        print(f"  {ox:<8.2f}{s:>12.6f}{r:>14.6f}{d:>12.2e}")

    print(f"\n  worst |difference| = {worst:.3e}")
    n_stored = len(stored[oxs[0] if oxs[0] in stored else str(oxs[0])])
    n_here = len(arm["di"][oxs[0]])
    if n_stored != n_here:
        print(f"  [note] replicate counts differ: stored {n_stored}, here {n_here}.")
        print("         Run with --nreps {} to compare like with like."
              .format(n_stored))
        return ok
    if worst < 1e-9:
        print("  PASS — identical to the stored sweep.")
        return ok
    if worst < 1e-3:
        print("  PASS — agrees to within numerical tolerance.")
        return ok
    print("  FAIL — the delta arm does not reproduce the main experiment.")
    print("  Do not use this control until the difference is understood.")
    return False


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nreps", type=int, default=50,
                    help="replicates per condition (default 50, matching the "
                         "main experiment)")
    ap.add_argument("--validate", action="store_true",
                    help="check the delta arm against the stored main experiment")
    ap.add_argument("--dry-run", action="store_true",
                    help="run and report, but do not write the pickle")
    args = ap.parse_args()

    ensure_dirs()

    print("=" * 74)
    print("LEARNING-RULE CONTROL")
    print("=" * 74)
    print("Loading the agent model (this runs the full pipeline stage) ...")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_agent", _repo_root() / "scripts" / "03_agent_model.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)

    oxs = list(M.OX_MAIN)
    dim = M.RATE_DIM
    ch, ch_sd = chance_alignment(dim)

    print("\nconfiguration")
    print(f"  OX2R grid          {[f'{v:g}' for v in oxs]}")
    print(f"  replicates         {args.nreps}")
    print(f"  readout dimension  {dim}")
    print(f"  chance alignment   {ch:.4f} (sd {ch_sd:.4f})")
    print(f"  alignment target   {ALIGN_TARGET}")

    arms = {}
    for rule in RULES:
        print(f"\n[{rule}] sweeping ...")
        arms[rule] = run_rule_sweep(M, rule, oxs, args.nreps)
        m_di = [np.mean(arms[rule]["di"][ox]) for ox in oxs]
        m_al = [np.mean(arms[rule]["align"][ox]) for ox in oxs]
        print(f"   DI    {m_di[0]:.4f} -> {max(m_di):.4f} -> {m_di[-1]:.4f}")
        print(f"   ALIGN {m_al[0]:.4f} -> {max(m_al):.4f} -> {m_al[-1]:.4f}"
              f"   (chance {ch:.4f})")

    ok = True
    if args.validate:
        ok = validate_delta(M, arms["delta"], oxs)

    print("\n" + "-" * 74)
    print("CURVE STATISTICS")
    print("-" * 74)
    out = {}
    hdr = (f"{'rule':<9}{'metric':<9}{'curvature':>12}{'p(c>=0)':>10}"
           f"{'peak':>8}{'95% CI':>18}{'p_int':>8}")
    print(hdr)
    print("-" * len(hdr))
    for rule in RULES:
        out[rule] = {}
        metrics = [("stats_ALIGN", "align")]
        if rule in RULES_WITH_DI:
            metrics.insert(0, ("stats_DI", "di"))
        for key, which in metrics:
            st = curve_stats(M, arms[rule][which], oxs)
            out[rule][key] = st
            ci = f"[{st['peak_ci'][0]:.2f}, {st['peak_ci'][1]:.2f}]"
            print(f"{rule:<9}{which.upper():<9}{st['curvature']:>+12.5f}"
                  f"{st['p_nonneg']:>10.4f}{st['peak']:>8.3f}{ci:>18}"
                  f"{st['p_interior']:>8.2f}")
        # Per-replicate arrays. The earlier control stored only the summaries,
        # which cost the figure its error bands, the paired bootstrap on the
        # hebb-delta optimum offset, and the DI sign check. Keeping them is
        # cheap and restores all three.
        out[rule]["stats_ALIGN_other"] = curve_stats(M, arms[rule]["align_other"], oxs)
        out[rule]["replicates"] = {
            "di": {float(ox): arms[rule]["di"][ox].tolist() for ox in oxs},
            "align": {float(ox): arms[rule]["align"][ox].tolist() for ox in oxs},
            "align_other": {float(ox): arms[rule]["align_other"][ox].tolist()
                            for ox in oxs},
        }

    # ---- provenance, which the earlier pickle lacked ------------------------
    out["_meta"] = {
        **stamp(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_reps": args.nreps,
        "ox_grid": [float(v) for v in oxs],
        "rate_dim": int(dim),
        "chance_alignment": ch,
        "align_target": ALIGN_TARGET,
        "align_target_other": _OTHER_TARGET,
        "readout_seed_offset": READOUT_SEED_OFFSET,
        "rules": list(RULES),
        "rules_with_di": list(RULES_WITH_DI),
        "n_boot_curvature": N_BOOT_CURV,
        "n_boot_peak": N_BOOT_PEAK,
        "stat_seed": STAT_SEED,
        "encode_ox2r": 1.0,
        "task_divergence": float(M.TASK_DIVERGENCE),
        "seq_len": int(M.TaskConfig.MWM(divergence=M.TASK_DIVERGENCE).seq_len),
    }

    print("\n" + "-" * 74)
    if not ok:
        print("VALIDATION FAILED - nothing written.")
        print("The delta arm did not reproduce the main experiment, so this")
        print("control would not describe the same pipeline as the figures.")
        print("Resolve the difference, or drop --validate if you are")
        print("deliberately running against a different results pickle.")
        return 1
    if args.dry_run:
        print("--dry-run: nothing written.")
    else:
        with open(OUT, "wb") as f:
            pickle.dump(out, f)
        print(f"written: {OUT}")
        print(f"  pipeline version {PIPELINE_VERSION}, {args.nreps} replicates")
        print("\nNext: python scripts/figures/figure_s2_learning_rule.py")

    if not ok:
        print("\nVALIDATION FAILED — see above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
