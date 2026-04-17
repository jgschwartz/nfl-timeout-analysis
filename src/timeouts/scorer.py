"""Pair each EarlyTimeout with its best opportunity and compute opportunity cost."""
import pandas as pd
from src.wp.calculator import calculate_wp, WP_INPUT_COLS


def score_timeouts(
    early_timeouts: pd.DataFrame,
    pbp: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one TimeoutScore record per early timeout.

    Computes wp_gain_early and pairs each timeout with its maximum-value opportunity.
    """
    scored = _add_early_wp_gain(early_timeouts)

    best_opps = (
        opportunities
        .sort_values("wp_gain_counterfactual", ascending=False)
        .groupby("ref_timeout_play_id", sort=False)
        .first()
        .reset_index()
        .rename(columns={
            "ref_timeout_play_id": "play_id",
            "play_id": "best_opportunity_play_id",
            "half_seconds_remaining": "best_opportunity_time",
            "wp_gain_counterfactual": "best_opportunity_value",
        })[["play_id", "best_opportunity_play_id",
            "best_opportunity_time", "best_opportunity_value"]]
    )

    result = scored.merge(best_opps, on="play_id", how="left")
    result["opportunity_cost"] = result["best_opportunity_value"] - result["wp_gain_early"]
    result["verdict"] = result["opportunity_cost"].apply(
        lambda x: "justified" if x <= 0 else "wasteful"
    )
    return result


def _add_early_wp_gain(early_timeouts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute WP gain for each early timeout call.

    Avoided-penalty state: actual game state, posteam_timeouts_remaining as-is
    (already N-1 after the timeout was consumed in the PBP record).
    With-penalty state: ydstogo+5, yardline_100+5, posteam_timeouts_remaining+1 (N).
    """
    avoided = early_timeouts[WP_INPUT_COLS].copy()

    penalty = avoided.copy()
    penalty["ydstogo"] = (early_timeouts["ydstogo"] + 5).clip(upper=99)
    penalty["yardline_100"] = (early_timeouts["yardline_100"] + 5).clip(upper=99)
    penalty["posteam_timeouts_remaining"] = (
        early_timeouts["posteam_timeouts_remaining"] + 1
    ).clip(upper=3)

    n = len(early_timeouts)
    combined = pd.concat([avoided, penalty], ignore_index=True)
    wp = calculate_wp(combined)

    result = early_timeouts.copy()
    result["wp_gain_early"] = wp.iloc[:n].values - wp.iloc[n:].values
    return result


def save_scores(scores: pd.DataFrame, path: str = "data/processed/timeout_scores.parquet") -> None:
    """Persist TimeoutScore records for downstream analysis."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False)
