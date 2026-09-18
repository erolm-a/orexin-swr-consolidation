"""
common.paths — one import to set up project paths in every pipeline script.

Use:
    from common.paths import setup
    DIRS, FILES, PROJECT_ROOT, VERSIONS, load_artifact = setup()

Replaces the duplicated sys.path / orexin_paths boilerplate that used to open
every script. ``common`` lives at the repository root; root-level scripts are
launched with the repo directory on sys.path, so ``import common`` resolves
without configuration.
"""

import sys
from pathlib import Path

# Make the repository root importable so `import orexin_paths` (and the other
# common.* modules) resolve no matter where the entry script lives.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from orexin_paths import (  # noqa: E402
    DIRS,
    FILES,
    PROJECT_ROOT,
    VERSIONS,
    check_version,
    ensure_dirs,
    find_file,
    load_artifact,
    stamp,
)


def setup():
    """
    Create the artefact directory tree and return the shared path tables.

    Call this at the top of every pipeline script:
        DIRS, FILES, PROJECT_ROOT, VERSIONS, load_artifact = setup()
    """
    ensure_dirs()
    return DIRS, FILES, PROJECT_ROOT, VERSIONS, load_artifact


__all__ = [
    "setup",
    "DIRS",
    "FILES",
    "PROJECT_ROOT",
    "VERSIONS",
    "ensure_dirs",
    "load_artifact",
    "find_file",
    "stamp",
    "check_version",
]