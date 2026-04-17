"""rpy2 wrapper around nflfastR's calculate_win_probability."""
import os
import subprocess
import pandas as pd

try:
    # Auto-detect R_HOME from the system R installation so rpy2 uses API mode.
    if "R_HOME" not in os.environ:
        os.environ["R_HOME"] = subprocess.check_output(
            ["Rscript", "-e", "cat(R.home())"], text=True
        ).strip()

    import rpy2.robjects as _ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
except ModuleNotFoundError:
    _ro = None  # type: ignore
    pandas2ri = None  # type: ignore
    localconverter = None  # type: ignore
    importr = None  # type: ignore

# nflfastR's calculate_win_probability() requires these columns.
# home_team + posteam are used internally to build the home-team indicator.
WP_INPUT_COLS = [
    "home_team",
    "posteam",
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "receive_2h_ko",
    "spread_line",
]

_nflfastr = None


def _get_nflfastr():
    global _nflfastr
    if _nflfastr is None:
        _nflfastr = importr("nflfastR")
    return _nflfastr


def calculate_wp(states: pd.DataFrame) -> pd.Series:
    """
    Compute win probability for a batch of game states.

    Returns a Series with the same index as `states`.
    Initializes the R session lazily on first call.
    """
    nflfastr = _get_nflfastr()
    batch = states[WP_INPUT_COLS].reset_index(drop=True)
    with localconverter(_ro.default_converter + pandas2ri.converter):
        r_df = _ro.conversion.py2rpy(batch)
        r_result = nflfastr.calculate_win_probability(r_df)
        result_df = _ro.conversion.rpy2py(r_result)
    return pd.Series(result_df["wp"].values, index=states.index)
