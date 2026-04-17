"""rpy2 wrapper around nflfastR's calculate_win_probability."""
import pandas as pd

try:
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
except ModuleNotFoundError:
    pandas2ri = None  # type: ignore
    importr = None  # type: ignore

WP_INPUT_COLS = [
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
    r_df = pandas2ri.py2rpy(states[WP_INPUT_COLS].reset_index(drop=True))
    r_result = nflfastr.calculate_win_probability(r_df)
    result_df = pandas2ri.rpy2py(r_result)
    return pd.Series(result_df["wp"].values, index=states.index)
