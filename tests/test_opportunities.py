from unittest.mock import patch
import pandas as pd
import pytest
from src.timeouts.opportunities import compute_all_opportunities


@pytest.fixture
def early_timeout():
    """Single early timeout record (play_id=200, qtr=1)."""
    return pd.Series({
        "game_id": "2023_01_KC_DET",
        "play_id": 200,
        "season": 2023,
        "posteam": "KC",
        "home_team": "DET",
        "qtr": 1,
        "half_seconds_remaining": 1500,
        "game_seconds_remaining": 3300,
        "score_differential": 0,
        "down": 2,
        "ydstogo": 7,
        "yardline_100": 43,
        "posteam_timeouts_remaining": 2,
        "defteam_timeouts_remaining": 3,
        "receive_2h_ko": 1,
        "spread_line": 0.0,
        "coach": "Andy Reid",
        "excluded": False,
    })


def test_returns_dataframe(sample_pbp, early_timeout):
    early_timeouts = early_timeout.to_frame().T.reset_index(drop=True)
    # sample_pbp: timeout at play_id=200, half_sec=1500, qtr=1
    # Remaining scrimmage plays in half 1 (qtrs 1-2) with half_sec < 1500:
    #   play_id=400, qtr=2, half_sec=600, play_type="pass" => n=1
    # combined = actual(1) + counterfactual(1) = 2 rows
    fake_wp = pd.Series([0.50, 0.51])
    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp):
        result = compute_all_opportunities(sample_pbp, early_timeouts)
    assert isinstance(result, pd.DataFrame)


def test_has_required_columns(sample_pbp, early_timeout):
    early_timeouts = early_timeout.to_frame().T.reset_index(drop=True)
    fake_wp = pd.Series([0.50, 0.51])
    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp):
        result = compute_all_opportunities(sample_pbp, early_timeouts)
    for col in ["ref_timeout_play_id", "play_id", "half_seconds_remaining",
                "clock_runoff", "actual_timeouts", "wp_gain_counterfactual"]:
        assert col in result.columns, f"Missing column: {col}"


def test_links_to_timeout(sample_pbp, early_timeout):
    early_timeouts = early_timeout.to_frame().T.reset_index(drop=True)
    fake_wp = pd.Series([0.50, 0.51])
    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp):
        result = compute_all_opportunities(sample_pbp, early_timeouts)
    assert (result["ref_timeout_play_id"] == 200).all()


def test_clock_runoff_non_negative(sample_pbp, early_timeout):
    early_timeouts = early_timeout.to_frame().T.reset_index(drop=True)
    fake_wp = pd.Series([0.50, 0.51])
    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp):
        result = compute_all_opportunities(sample_pbp, early_timeouts)
    assert (result["clock_runoff"] >= 0).all()


def test_empty_result_for_no_remaining_plays(early_timeout):
    """When no scrimmage plays remain after the timeout, return empty DataFrame."""
    tiny_pbp = pd.DataFrame({
        "game_id": ["2023_01_KC_DET"],
        "play_id": [200],
        "season": [2023],
        "posteam": ["KC"],
        "defteam": ["DET"],
        "qtr": [1],
        "half_seconds_remaining": [1500],
        "game_seconds_remaining": [3300],
        "score_differential": [0],
        "down": [2],
        "ydstogo": [7],
        "yardline_100": [43],
        "posteam_timeouts_remaining": [2],
        "defteam_timeouts_remaining": [3],
        "receive_2h_ko": [1],
        "spread_line": [0.0],
        "timeout": [1],
        "timeout_team": ["KC"],
        "play_type": ["no_play"],
        "desc": ["Timeout"],
    })
    early_timeouts = early_timeout.to_frame().T.reset_index(drop=True)
    with patch("src.timeouts.opportunities.calculate_wp") as mock_wp:
        result = compute_all_opportunities(tiny_pbp, early_timeouts)
    mock_wp.assert_not_called()
    assert result.empty
