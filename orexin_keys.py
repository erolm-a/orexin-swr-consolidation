"""
Canonical library key formatter.

The generator and the agent both index synthetic libraries by a string built
from the context name and the OX2R level. If the two ever disagreed about how
a float becomes a string, the agent would silently fail to find trials for a
level the generator had produced. Keeping one definition here removes that
possibility.

    0.30 -> "0.3"
    1.00 -> "1"
    0.85 -> "0.85"
    1.05 -> "1.05"

Library keys have the form f"{context}_ox{ox_key(level)}".
"""

from __future__ import annotations

__all__ = ["ox_key", "library_key"]


def ox_key(x: float) -> str:
    """OX2R level -> canonical string key, trailing zeros stripped."""
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def library_key(context: str, level: float) -> str:
    """Full library key for a context and OX2R level."""
    return f"{context}_ox{ox_key(level)}"