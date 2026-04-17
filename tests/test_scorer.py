from unittest.mock import patch
import pandas as pd
import pytest
from src.timeouts.scorer import score_timeouts


@pytest.fixture
def early_timeouts():
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET"],
        "play_id": [200],
        "season": [2023],
        "posteam": ["KC"],
        "home_team": ["DET"],
        "coach": ["Andy Reid"],
        "qtr": [1],
        "half_seconds_remaining": [1500],
        "game_seconds_remaining": [3300],
        "score_differential": [0],
        "down": [2],
        "ydstogo": [7],
        "yardline_100": [43],
        "posteam_timeouts_remaining": [2],   # after timeout consumed (N-1)
        "defteam_timeouts_remaining": [3],
        "receive_2h_ko": [1],
        "spread_line": [0.0],
        "excluded": [False],
    })


@pytest.fixture
def opportunities():
    return pd.DataFrame({
        "ref_timeout_play_id": [200, 200, 200],
        "play_id": [300, 400, 500],
        "half_seconds_remaining": [900, 600, 400],
        "clock_runoff": [30, 25, 20],
        "actual_timeouts": [2, 2, 1],
        "wp_gain_counterfactual": [0.02, 0.08, 0.05],  # play 400 is best
    })


def test_has_required_columns(early_timeouts, opportunities):
    fake_wp = pd.Series([0.55, 0.52])  # avoided-penalty, with-penalty
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, opportunities)
    for col in ["play_id", "wp_gain_early", "best_opportunity_value",
                "best_opportunity_play_id", "best_opportunity_time",
                "opportunity_cost", "verdict"]:
        assert col in result.columns, f"Missing: {col}"


def test_best_opportunity_is_maximum(early_timeouts, opportunities):
    fake_wp = pd.Series([0.55, 0.52])
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, opportunities)
    assert result.iloc[0]["best_opportunity_play_id"] == 400
    assert result.iloc[0]["best_opportunity_value"] == pytest.approx(0.08)


def test_opportunity_cost_calculation(early_timeouts, opportunities):
    # wp_gain_early = 0.55 - 0.52 = 0.03; best_opp = 0.08; cost = 0.08 - 0.03 = 0.05
    fake_wp = pd.Series([0.55, 0.52])
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, opportunities)
    assert result.iloc[0]["opportunity_cost"] == pytest.approx(0.05)


def test_verdict_justified(early_timeouts, opportunities):
    # wp_gain_early = 0.10 > 0.08 best_opp → cost < 0 → justified
    fake_wp = pd.Series([0.60, 0.50])
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, opportunities)
    assert result.iloc[0]["verdict"] == "justified"


def test_verdict_wasteful(early_timeouts, opportunities):
    # wp_gain_early = 0.03 < 0.08 best_opp → cost > 0 → wasteful
    fake_wp = pd.Series([0.55, 0.52])
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, opportunities)
    assert result.iloc[0]["verdict"] == "wasteful"
