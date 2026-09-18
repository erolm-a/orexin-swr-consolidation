"""
common — shared modules for the orexin-SWR consolidation pipeline.

Single canon for the utilities, statistics and plotting helpers that used to be
copied between the pipeline scripts. Scripts import from these modules directly
(``from common.utils import ox_key``) or, for convenience, re-exported here.
"""

from . import paths, plotting, stats, utils
from .paths import (  # noqa: F401
    DIRS,
    FILES,
    PROJECT_ROOT,
    VERSIONS,
    check_version,
    ensure_dirs,
    find_file,
    load_artifact,
    setup,
    stamp,
)
from .plotting import panel_label, plabel, save_and_show  # noqa: F401
from .stats import (  # noqa: F401
    bootstrap_continuous_peak,
    bootstrap_peak_location,
    dist_stats,
    limb_decomposition,
    test_complexity_interaction,
    test_curvature,
    test_monotonicity,
    validate_peak_shift,
)
from .utils import (  # noqa: F401
    _cat,
    _mean_or_nan,
    _median_or_nan,
    _safe_np,
    _std_or_nan,
    bandpass,
    compute_isis_within_intervals,
    compute_min_duration_ms,
    coupled_event_arrays_from_trial,
    ox_key,
    pooled_coupled_event_arrays,
    pooled_p_couple,
    pooled_rate_from_trials,
    pooled_rate_per_min,
    ramp_to_one,
    safe_concat,
    sample_trunc_norm,
    stable_seed,
)

__all__ = [
    "setup", "DIRS", "FILES", "PROJECT_ROOT", "VERSIONS",
    "ensure_dirs", "load_artifact", "find_file", "stamp", "check_version",
    "plabel", "panel_label", "save_and_show",
    "test_curvature", "test_monotonicity", "bootstrap_continuous_peak",
    "bootstrap_peak_location", "validate_peak_shift", "limb_decomposition",
    "test_complexity_interaction", "dist_stats",
    "ox_key", "bandpass", "stable_seed", "safe_concat", "sample_trunc_norm",
    "compute_min_duration_ms", "compute_isis_within_intervals", "ramp_to_one",
    "_safe_np", "_mean_or_nan", "_median_or_nan", "_std_or_nan", "_cat",
    "coupled_event_arrays_from_trial", "pooled_coupled_event_arrays",
    "pooled_rate_from_trials", "pooled_rate_per_min", "pooled_p_couple",
]