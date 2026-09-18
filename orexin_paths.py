"""
orexin_paths.py — single source of truth for filesystem locations and for the
version of every pipeline artefact.

Filenames are version-free. The version lives inside each pickle instead, and
is checked on load. This is strictly safer than encoding it in the filename:
a filename mismatch is only caught if a human notices it, whereas
`load_artifact()` raises.

PROJECT_ROOT is $OREXIN_PROJECT_ROOT if set, otherwise the repository
directory. Nothing else is consulted: no cloud storage, no notebook runtime,
no machine-specific defaults. Working directories are created on demand and
are git-ignored, so a fresh clone runs without configuration.

To point the pipeline at data held elsewhere:

    export OREXIN_PROJECT_ROOT=/path/to/orexin_spw_r_project

Usage
-----
    from orexin_paths import DIRS, FILES, load_artifact, stamp

    D = load_artifact("results")        # version-checked on the way in

    with open(FILES["results"], "wb") as f:
        pickle.dump({**payload, **stamp("results")}, f)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

__all__ = [
    "PROJECT_ROOT", "DIRS", "FILES", "VERSIONS",
    "ensure_dirs", "find_file",
    "load_artifact", "check_version", "stamp",
]

# --------------------------------------------------------------------------- #
# Pipeline versions — the ONLY place these strings are written down
# --------------------------------------------------------------------------- #

VERSIONS: dict[str, str] = {
    "detector": "5.4.5",
    "generator": "5.8.3",
    "model": "2.5.11",
}

# --------------------------------------------------------------------------- #
# Root resolution
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parent


def _resolve_root() -> Path:
    """$OREXIN_PROJECT_ROOT if set, else the repository directory."""
    env = os.environ.get("OREXIN_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _REPO_ROOT


PROJECT_ROOT: Path = _resolve_root()

# --------------------------------------------------------------------------- #
# Directory map
# --------------------------------------------------------------------------- #

DIRS: dict[str, Path] = {
    "root": PROJECT_ROOT,
    "data_raw": PROJECT_ROOT / "data_raw",              # CRCNS LFP (never committed)
    "data_processed": PROJECT_ROOT / "data_processed",  # detector output
    "models": PROJECT_ROOT / "models",                  # synthetic libraries, configs
    "results": PROJECT_ROOT / "results",                # simulation results
    "figures": PROJECT_ROOT / "figures",                # publication figures
    "checkpoints": PROJECT_ROOT / "checkpoints",        # agent checkpoints
    "tables": PROJECT_ROOT / "tables",                  # Table 1, supplementary tables
}

# --------------------------------------------------------------------------- #
# Artefact filenames — version-free
# --------------------------------------------------------------------------- #

FILES: dict[str, Path] = {
    "detector": DIRS["data_processed"] / "swr_detection.pkl",
    "libraries": DIRS["models"] / "synthetic_libraries.pkl",
    "generator_config": DIRS["models"] / "generator_config.pkl",
    "calibration_libraries": DIRS["models"] / "calibration_libraries.pkl",
    "results": DIRS["results"] / "agent_model_results.pkl",
}

# Backwards-compatible alias for the old key name.
FILES["confirmed_results"] = FILES["results"]

# Which key inside each pickle carries the version, and which VERSIONS entry it
# must match. None = artefact carries no version stamp and is not checked.
_VERSION_FIELD: dict[str, tuple[str, str] | None] = {
    "detector": ("version", "detector"),
    "libraries": ("generator_version", "generator"),
    "generator_config": ("generator_version", "generator"),
    "results": ("version", "model"),
    "calibration_libraries": None,   # derived on demand, inherits generator version
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def ensure_dirs(verbose: bool = False) -> None:
    """Create the directory tree if it does not exist."""
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  {name}: {path}")


def find_file(filename: str) -> Path:
    """Locate `filename` anywhere in the known directory tree."""
    for path in DIRS.values():
        candidate = path / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{filename!r} not found under {PROJECT_ROOT}. "
        f"Set $OREXIN_PROJECT_ROOT or check that the artefact has been generated."
    )


def stamp(key: str) -> dict[str, str]:
    """
    Version fields to merge into an artefact before pickling:

        pickle.dump({**payload, **stamp("results")}, f)

    Guarantees the stamp matches what load_artifact() will check for.
    """
    spec = _VERSION_FIELD.get(key)
    if spec is None:
        return {}
    field, version_key = spec
    return {field: VERSIONS[version_key]}


def check_version(key: str, payload: dict, strict: bool = True) -> str | None:
    """
    Verify the version recorded inside a loaded artefact.

    Returns the version string found, or None if the artefact is unversioned.
    Raises ValueError on mismatch when strict.
    """
    spec = _VERSION_FIELD.get(key)
    if spec is None:
        return None
    field, version_key = spec
    expected = VERSIONS[version_key]
    found = payload.get(field) if isinstance(payload, dict) else None

    if found is None:
        msg = (
            f"{FILES[key].name} carries no {field!r} field. It was probably "
            f"written by an older script. Regenerate it, or pass strict=False."
        )
        if strict:
            raise ValueError(msg)
        print(f"  [warn] {msg}")
        return None

    if str(found) != expected:
        msg = (
            f"Version mismatch in {FILES[key].name}: file says {field}={found!r}, "
            f"code expects {expected!r}. Regenerate the artefact, or update "
            f"VERSIONS[{version_key!r}] in orexin_paths.py."
        )
        if strict:
            raise ValueError(msg)
        print(f"  [warn] {msg}")
    return str(found)


def load_artifact(key: str, strict: bool = True) -> dict:
    """Unpickle a named artefact and verify its recorded version."""
    path = FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the stage that produces it (see README), "
            f"or fetch it from the Zenodo archive."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    check_version(key, payload, strict=strict)
    return payload


if __name__ == "__main__":
    _src = ("$OREXIN_PROJECT_ROOT" if os.environ.get("OREXIN_PROJECT_ROOT")
            else "repository directory (default)")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"  source: {_src}\n")
    print("versions:")
    for k, v in VERSIONS.items():
        print(f"  {k:<12} {v}")
    print("\nartefacts:")
    for key in ("detector", "libraries", "generator_config",
                "calibration_libraries", "results"):
        path = FILES[key]
        if not path.exists():
            print(f"  [MISSING] {key:<22} {path}")
            continue
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            found = check_version(key, payload, strict=False)
            tag = f"v{found}" if found else "unversioned"
            print(f"  [ok     ] {key:<22} {path.name}  ({tag})")
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR  ] {key:<22} {path.name}: {type(e).__name__}: {e}")
