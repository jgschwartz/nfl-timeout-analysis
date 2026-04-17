import pandas as pd
import pytest


@pytest.fixture
def sample_pbp():
    """Minimal PBP DataFrame with columns needed across all tests."""
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET"] * 6,
        "play_id": [100, 200, 300, 400, 500, 600],
        "season": [2023] * 6,
        "posteam": ["KC"] * 6,
        "defteam": ["DET"] * 6,
        "coach": ["Andy Reid"] * 6,
        "qtr": [1, 1, 2, 2, 3, 3],
        "half_seconds_remaining": [1800, 1500, 900, 600, 1800, 400],
        "game_seconds_remaining": [3600, 3300, 1800, 1500, 1800, 400],
        "score_differential": [0, 0, 0, 0, 7, 7],
        "down": [1, 2, 2, 1, 1, 1],
        "ydstogo": [10, 7, 7, 10, 10, 10],
        "yardline_100": [50, 43, 43, 50, 50, 50],
        "posteam_timeouts_remaining": [3, 3, 2, 2, 3, 3],
        "defteam_timeouts_remaining": [3, 3, 3, 3, 3, 3],
        "receive_2h_ko": [1, 1, 1, 1, 0, 0],
        "spread_line": [0.0] * 6,
        "timeout": [0, 1, 1, 0, 1, 1],
        "timeout_team": [None, "KC", "KC", None, "KC", "KC"],
        "play_type": ["pass", "no_play", "no_play", "pass", "no_play", "no_play"],
        "desc": [
            "Pass incomplete",
            "Timeout #2 by KC",
            "Timeout #1 by KC",
            "Run for 5",
            "Timeout #3 by KC",
            "Timeout #3 by KC INJURY",
        ],
    })
