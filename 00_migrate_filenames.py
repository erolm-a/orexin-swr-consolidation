"""
00_migrate_filenames.py — one-time rename of the old versioned artefacts to the
version-free names used by orexin_paths.FILES.

Run this ONCE against an existing project directory before running any
other script. Point it at the directory with $OREXIN_PROJECT_ROOT. It is safe to re-run: files already
carrying the new name are left alone.

    python scripts/00_migrate_filenames.py --dry-run     # show what would move
    python scripts/00_migrate_filenames.py               # do it

The results pickle is the one to watch. Two filenames were in circulation:

    agent_model_v2.5.10_confirmed_results.pkl
    agent_model_v2.5.11_confirmed_results.pkl

If BOTH exist, this script refuses to guess. Inspect them, decide which is
authoritative, delete or archive the other, then re-run.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orexin_paths import DIRS, FILES, VERSIONS, ensure_dirs  # noqa: E402

# old filename(s) -> FILES key. Order matters only for the results pickle.
RENAMES: list[tuple[str, str, str]] = [
    # (directory key, old filename, FILES key)
    ("data_processed", "swr_detection_v5.4.5.pkl", "detector"),
    ("models", "synthetic_libraries_v5.8.3.pkl", "libraries"),
    ("models", "generator_config_v5.8.3.pkl", "generator_config"),
    ("models", "calibration_libraries_n2000.pkl", "calibration_libraries"),
]

AMBIGUOUS = [
    ("results", "agent_model_v2.5.11_confirmed_results.pkl", "results"),
    ("results", "agent_model_v2.5.10_confirmed_results.pkl", "results"),
]


def peek_version(path: Path) -> str:
    """Report whatever version fields a pickle carries, without validating."""
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        return f"<unreadable: {type(e).__name__}>"
    if not isinstance(d, dict):
        return "<not a dict>"
    bits = [f"{k}={d[k]!r}" for k in ("version", "generator_version",
                                      "detector_version", "library") if k in d]
    return ", ".join(bits) or "<no version field>"


def move(src: Path, dst: Path, dry: bool) -> None:
    if dry:
        print(f"  would move  {src.name}  ->  {dst.name}")
        return
    src.rename(dst)
    print(f"  moved       {src.name}  ->  {dst.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_dirs()

    print(f"project root: {DIRS['root']}")
    print(f"expected versions: {VERSIONS}\n")

    # ---- unambiguous renames ---------------------------------------------- #
    for dir_key, old_name, file_key in RENAMES:
        src = DIRS[dir_key] / old_name
        dst = FILES[file_key]
        if dst.exists():
            print(f"  already ok  {dst.name}   ({peek_version(dst)})")
            continue
        if not src.exists():
            print(f"  not found   {old_name}  (skipping)")
            continue
        print(f"  {old_name}: {peek_version(src)}")
        move(src, dst, args.dry_run)

    # ---- results pickle: refuse to guess ----------------------------------- #
    print()
    dst = FILES["results"]
    present = [(DIRS[d] / n) for d, n, _ in AMBIGUOUS if (DIRS[d] / n).exists()]

    if dst.exists():
        print(f"  already ok  {dst.name}   ({peek_version(dst)})")
        for p in present:
            print(f"  [note] old file still present, not touched: {p.name}")
    elif len(present) == 0:
        print("  not found   no results pickle under results/ (skipping)")
    elif len(present) == 1:
        src = present[0]
        print(f"  {src.name}: {peek_version(src)}")
        move(src, dst, args.dry_run)
    else:
        print("  [STOP] two results pickles found; refusing to guess which is real:")
        for p in present:
            size_mb = p.stat().st_size / 1e6
            print(f"     {p.name}")
            mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(" ", "seconds")
            print(f"       {size_mb:.1f} MB, modified {mtime}")
            print(f"       internal: {peek_version(p)}")
        print("\n  Decide which is authoritative, archive or delete the other,")
        print("  then re-run this script.")
        return 1

    print("\nNow run:  python orexin_paths.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
