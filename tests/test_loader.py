from unittest.mock import patch
import pandas as pd
import pytest
from src.data.loader import load_seasons, REQUIRED_FIELDS


@pytest.fixture
def pbp_df():
    # PBP already contains home_coach/away_coach/home_team.
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET", "2023_01_KC_DET"],
        "play_id": [1, 2],
        "season": [2023, 2023],
        "posteam": ["KC", "KC"],
        "home_team": ["DET", "DET"],
        "away_team": ["KC", "KC"],
        "home_coach": ["Dan Campbell", "Dan Campbell"],
        "away_coach": ["Andy Reid", "Andy Reid"],
        "home_opening_kickoff": [1.0, 1.0],
        "wp": [0.55, 0.60],
        "half_seconds_remaining": [1800, 1750],
        "posteam_timeouts_remaining": [3, 3],
    })


def test_load_seasons_joins_coach(pbp_df, tmp_path):
    with patch("src.data.loader.CACHE_DIR", tmp_path), \
         patch("src.data.loader.nflreadpy") as mock_nfl:
        mock_nfl.load_pbp.return_value = pbp_df
        result = load_seasons([2023])
    assert "coach" in result.columns
    assert result.loc[result["posteam"] == "KC", "coach"].iloc[0] == "Andy Reid"


def test_load_seasons_caches_per_season(pbp_df, tmp_path):
    with patch("src.data.loader.CACHE_DIR", tmp_path), \
         patch("src.data.loader.nflreadpy") as mock_nfl:
        mock_nfl.load_pbp.return_value = pbp_df
        load_seasons([2023])
        load_seasons([2023])  # second call uses per-season cache
    assert mock_nfl.load_pbp.call_count == 1

    # Per-season cache file exists
    assert (tmp_path / "pbp_2023.parquet").exists()


def test_load_seasons_retries_on_failure(pbp_df, tmp_path):
    from unittest.mock import call
    call_count = 0

    def flaky_load_pbp(seasons):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Remote end closed connection")
        return pbp_df

    with patch("src.data.loader.CACHE_DIR", tmp_path), \
         patch("src.data.loader.nflreadpy") as mock_nfl, \
         patch("src.data.loader.time.sleep"):  # skip actual sleep
        mock_nfl.load_pbp.side_effect = flaky_load_pbp
        result = load_seasons([2023])

    assert call_count == 2  # failed once, succeeded on retry
    assert len(result) == len(pbp_df)


def test_validate_warns_on_missing_fields(pbp_df, tmp_path, capsys):
    pbp_df.loc[0, "wp"] = None
    with patch("src.data.loader.CACHE_DIR", tmp_path), \
         patch("src.data.loader.nflreadpy") as mock_nfl:
        mock_nfl.load_pbp.return_value = pbp_df
        load_seasons([2023])
    captured = capsys.readouterr()
    assert "Warning" in captured.out
