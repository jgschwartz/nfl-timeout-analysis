"""Compute per-play counterfactual timeout value for each early timeout."""
import pandas as pd
from src.wp.calculator import calculate_wp, WP_INPUT_COLS

_SCRIMMAGE_PLAY_TYPES = {"pass", "run", "qb_kneel", "qb_spike"}


def compute_all_opportunities(
    pbp: pd.DataFrame,
    early_timeouts: pd.DataFrame,
) -> pd.DataFrame:
    """For every early timeout, compute WP gain at each remaining half play."""
    results = [
        _opportunities_for_timeout(pbp, row)
        for _, row in early_timeouts.iterrows()
    ]
    non_empty = [df for df in results if not df.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def _opportunities_for_timeout(pbp: pd.DataFrame, timeout: pd.Series) -> pd.DataFrame:
    half_qtrs = [1, 2] if timeout["qtr"] <= 2 else [3, 4]
    remaining = pbp[
        (pbp["game_id"] == timeout["game_id"])
        & (pbp["qtr"].isin(half_qtrs))
        & (pbp["half_seconds_remaining"] < timeout["half_seconds_remaining"])
        & (pbp["play_type"].isin(_SCRIMMAGE_PLAY_TYPES))
    ].sort_values("half_seconds_remaining", ascending=False).copy()

    if remaining.empty:
        return pd.DataFrame()

    # How much clock ran between the end of the previous play and the start of this one.
    # A timeout called here would have stopped the clock and recovered this time.
    prev_half_sec = remaining["half_seconds_remaining"].shift(
        1, fill_value=timeout["half_seconds_remaining"]
    )
    remaining["clock_runoff"] = (prev_half_sec - remaining["half_seconds_remaining"]).clip(lower=0)

    actual = remaining[WP_INPUT_COLS].copy()

    counterfactual = actual.copy()
    counterfactual["posteam_timeouts_remaining"] = (
        remaining["posteam_timeouts_remaining"] + 1
    ).clip(upper=3)
    counterfactual["half_seconds_remaining"] = (
        remaining["half_seconds_remaining"] + remaining["clock_runoff"]
    ).clip(upper=prev_half_sec.values)
    counterfactual["game_seconds_remaining"] = (
        remaining["game_seconds_remaining"] + remaining["clock_runoff"]
    )

    n = len(remaining)
    combined = pd.concat([actual, counterfactual], ignore_index=True)
    wp = calculate_wp(combined)

    return pd.DataFrame({
        "ref_timeout_play_id": timeout["play_id"],
        "play_id": remaining["play_id"].values,
        "half_seconds_remaining": remaining["half_seconds_remaining"].values,
        "clock_runoff": remaining["clock_runoff"].values,
        "actual_timeouts": remaining["posteam_timeouts_remaining"].values,
        "wp_gain_counterfactual": wp.iloc[n:].values - wp.iloc[:n].values,
    })
