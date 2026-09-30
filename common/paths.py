"""
common.paths — one import to set up project paths in every pipeline script.

Use:
    from common.paths import setup
    DIRS, FILES, PROJECT_ROOT, PIPELINE_VERSION, load_artifact = setup()

The pipeline scripts import from ``orexin_paths`` directly; this module is the
convenience form for anything that only needs the path tables. ``common`` lives
at the repository root, and root-level scripts are launched with the repo
directory on sys.path, so ``import common`` resolves without configuration.
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
    PIPELINE_VERSION,
    PROJECT_ROOT,
    check_version,
    ensure_dirs,
    find_file,
    load_artifact,
    stamp,
)
from orexin_keys import ox_key  # noqa: E402


def setup():
    """
    Create the artefact directory tree and return the shared path tables.

    Call this at the top of a script that needs nothing else from orexin_paths:
        DIRS, FILES, PROJECT_ROOT, PIPELINE_VERSION, load_artifact = setup()
    """
    ensure_dirs()
    return DIRS, FILES, PROJECT_ROOT, PIPELINE_VERSION, load_artifact


__all__ = [
    "setup",
    "DIRS",
    "FILES",
    "PROJECT_ROOT",
    "PIPELINE_VERSION",
    "ensure_dirs",
    "load_artifact",
    "find_file",
    "stamp",
    "check_version",
    "ox_key",
]