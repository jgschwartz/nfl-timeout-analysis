"""Identify early penalty-avoidance timeout events from NFL play-by-play data."""
import pandas as pd

EARLY_HALF_THRESHOLD = 300   # seconds; timeouts after this are clock management
GOAL_LINE_THRESHOLD = 5      # yards to end zone; inside this = strategic timeout
INJURY_PATTERN = r"(?i)\binjury\b"

_OUTPUT_COLS = [
    "game_id", "play_id", "season", "posteam", "coach",
    "qtr", "half_seconds_remaining", "game_seconds_remaining",
    "score_differential", "down", "ydstogo", "yardline_100",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "receive_2h_ko", "spread_line",
]


def detect_early_timeouts(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Return plays where an offensive team called an early timeout.

    Excludes injury timeouts, 4th-down timeouts, and goal-line timeouts.
    `posteam_timeouts_remaining` reflects the count after the timeout was
    consumed (i.e., N−1 where N was the count before the call).
    """
    mask = (
        (pbp["timeout"] == 1)
        & (pbp["timeout_team"] == pbp["posteam"])
        & (pbp["half_seconds_remaining"] > EARLY_HALF_THRESHOLD)
        & (pbp["down"] != 4)
        & (pbp["yardline_100"] > GOAL_LINE_THRESHOLD)
        & (pbp["qtr"] <= 4)
        & ~pbp["desc"].str.contains(INJURY_PATTERN, regex=True, na=False)
    )

    result = pbp.loc[mask, _OUTPUT_COLS].copy()
    result["excluded"] = False
    return result.reset_index(drop=True)
