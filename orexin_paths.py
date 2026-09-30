"""
Filesystem locations and artefact identity for the pipeline.

PROJECT_ROOT is $OREXIN_PROJECT_ROOT if set, otherwise the repository
directory. Nothing else is consulted: no cloud storage, no notebook runtime,
no machine-specific defaults. Working directories are created on demand and
are git-ignored, so a fresh clone runs without configuration.

    export OREXIN_PROJECT_ROOT=/path/to/orexin_spw_r_project

Artefact filenames carry no version number. Each pickle is stamped with the
pipeline version instead, and the stamp is checked on load, so a stale
artefact raises rather than silently producing a figure from the wrong run.

    from orexin_paths import DIRS, FILES, load_artifact, stamp

    D = load_artifact("results")

    with open(FILES["results"], "wb") as f:
        pickle.dump({**payload, **stamp()}, f)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

__all__ = [
    "PROJECT_ROOT", "DIRS", "FILES", "PIPELINE_VERSION",
    "ensure_dirs", "find_file", "load_artifact", "check_version", "stamp",
]

# --------------------------------------------------------------------------- #
# One version for the whole pipeline, bumped per release
# --------------------------------------------------------------------------- #

PIPELINE_VERSION = "2.6.0"

_VERSION_FIELD = "pipeline_version"

# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #

# Imported as a module this is the file's directory; pasted into a
# notebook cell there is no __file__, so fall back to the working directory.
_REPO_ROOT = (Path(__file__).resolve().parent
              if "__file__" in globals() else Path.cwd().resolve())


def _resolve_root() -> Path:
    """$OREXIN_PROJECT_ROOT if set, else the repository directory."""
    env = os.environ.get("OREXIN_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _REPO_ROOT


PROJECT_ROOT: Path = _resolve_root()

# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #

DIRS: dict[str, Path] = {
    "root": PROJECT_ROOT,
    "data_raw": PROJECT_ROOT / "data_raw",              # recordings, never committed
    "data_processed": PROJECT_ROOT / "data_processed",  # detector output
    "models": PROJECT_ROOT / "models",                  # libraries and configs
    "results": PROJECT_ROOT / "results",                # simulation output
    "figures": PROJECT_ROOT / "figures",                # publication figures
    "tables": PROJECT_ROOT / "tables",                  # Table 1, supplements
    "checkpoints": PROJECT_ROOT / "checkpoints",        # agent checkpoints
}

# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #

FILES: dict[str, Path] = {
    "detector": DIRS["data_processed"] / "swr_detection.pkl",
    "libraries": DIRS["models"] / "synthetic_libraries.pkl",
    "generator_config": DIRS["models"] / "generator_config.pkl",
    "validation_null": DIRS["models"] / "validation_null.pkl",
    "calibration_libraries": DIRS["models"] / "calibration_libraries.pkl",
    "results": DIRS["results"] / "agent_model_results.pkl",
}

# Artefacts whose version stamp is checked on load. The calibration libraries
# are a cache regenerated on demand, so they carry no stamp.
_CHECKED = {"detector", "libraries", "generator_config", "validation_null", "results"}

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
    """Locate a file anywhere in the known directory tree."""
    for path in DIRS.values():
        candidate = path / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{filename!r} not found under {PROJECT_ROOT}. Set $OREXIN_PROJECT_ROOT, "
        f"or check that the artefact has been generated."
    )


def stamp() -> dict[str, str]:
    """Version field to merge into an artefact before pickling."""
    return {_VERSION_FIELD: PIPELINE_VERSION}


def check_version(key: str, payload, strict: bool = True) -> str | None:
    """
    Verify the version stamped inside a loaded artefact.

    Returns the version found, or None if this artefact is not version-checked.
    Raises ValueError on a mismatch when strict.
    """
    if key not in _CHECKED:
        return None
    found = payload.get(_VERSION_FIELD) if isinstance(payload, dict) else None

    if found is None:
        msg = (
            f"{FILES[key].name} carries no {_VERSION_FIELD!r} field. It predates "
            f"this release. Regenerate it, or pass strict=False."
        )
        if strict:
            raise ValueError(msg)
        print(f"  [warn] {msg}")
        return None

    if str(found) != PIPELINE_VERSION:
        msg = (
            f"Version mismatch in {FILES[key].name}: file says {found!r}, code is "
            f"{PIPELINE_VERSION!r}. Regenerate the artefact, or update "
            f"PIPELINE_VERSION in orexin_paths.py."
        )
        if strict:
            raise ValueError(msg)
        print(f"  [warn] {msg}")
    return str(found)


def load_artifact(key: str, strict: bool = True) -> dict:
    """Unpickle a named artefact and verify its version stamp."""
    path = FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the stage that produces it; see the README."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    check_version(key, payload, strict=strict)
    return payload


if __name__ == "__main__":
    _src = ("$OREXIN_PROJECT_ROOT" if os.environ.get("OREXIN_PROJECT_ROOT")
            else "repository directory (default)")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"  source: {_src}")
    print(f"  pipeline version: {PIPELINE_VERSION}\n")
    print("artefacts:")
    for key in FILES:
        path = FILES[key]
        if not path.exists():
            print(f"  [MISSING] {key:<22} {path}")
            continue
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            found = check_version(key, payload, strict=False)
            if found:
                tag = found
            elif key in _CHECKED:
                tag = "unstamped"
            else:
                tag = "not version-checked"
            print(f"  [ok     ] {key:<22} {path.name}  ({tag})")
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR  ] {key:<22} {path.name}: {type(e).__name__}: {e}")