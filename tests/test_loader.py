from unittest.mock import patch
import pandas as pd
import pytest
from src.data.loader import load_seasons, REQUIRED_FIELDS

@pytest.fixture
def pbp_df():
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET", "2023_01_KC_DET"],
        "play_id": [1, 2],
        "season": [2023, 2023],
        "posteam": ["KC", "KC"],
        "home_team": ["DET", "DET"],
        "wp": [0.55, 0.60],
        "half_seconds_remaining": [1800, 1750],
        "posteam_timeouts_remaining": [3, 3],
    })

@pytest.fixture
def schedules_df():
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET"],
        "home_team": ["DET"],
        "away_team": ["KC"],
        "home_coach": ["Dan Campbell"],
        "away_coach": ["Andy Reid"],
        "season": [2023],
    })

def test_load_seasons_joins_coach(pbp_df, schedules_df, tmp_path):
    with patch("src.data.loader.CACHE_DIR", tmp_path), \
         patch("src.data.loader.nflreadpy") as mock_nfl:
        mock_nfl.load_pbp.return_value = pbp_df
        mock_nfl.load_schedules.return_value = schedules_df
        result = load_seasons([2023])
    assert "coach" in result.columns
    assert result.loc[result["posteam"] == "KC", "coach"].iloc[0] == "Andy Reid"

def test_load_seasons_caches_parquet(pbp_df, schedules_df, tmp_path):
    with patch("src.data.loader.CACHE_DIR", tmp_path), \
         patch("src.data.loader.nflreadpy") as mock_nfl:
        mock_nfl.load_pbp.return_value = pbp_df
        mock_nfl.load_schedules.return_value = schedules_df
        load_seasons([2023])
        load_seasons([2023])  # second call should use cache
    assert mock_nfl.load_pbp.call_count == 1  # fetched only once

def test_validate_warns_on_missing_fields(pbp_df, schedules_df, tmp_path, capsys):
    pbp_df.loc[0, "wp"] = None
    with patch("src.data.loader.CACHE_DIR", tmp_path), \
         patch("src.data.loader.nflreadpy") as mock_nfl:
        mock_nfl.load_pbp.return_value = pbp_df
        mock_nfl.load_schedules.return_value = schedules_df
        load_seasons([2023])
    captured = capsys.readouterr()
    assert "Warning" in captured.out
