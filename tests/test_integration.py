"""End-to-end smoke test: minimal fake data + mocked WP calls."""
from unittest.mock import patch
import pandas as pd
import pytest

from src.timeouts.detector import detect_early_timeouts
from src.timeouts.opportunities import compute_all_opportunities
from src.timeouts.scorer import score_timeouts


@pytest.fixture
def full_pbp():
    """
    Realistic minimal PBP: one early timeout (play_id=200) in qtr 1,
    followed by several remaining scrimmage plays in the same half.
    """
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET"] * 8,
        "play_id": [100, 200, 300, 400, 500, 600, 700, 800],
        "season": [2023] * 8,
        "posteam": ["KC"] * 8,
        "home_team": ["DET"] * 8,
        "defteam": ["DET"] * 8,
        "coach": ["Andy Reid"] * 8,
        "qtr": [1] * 8,
        "half_seconds_remaining": [1800, 1700, 1600, 1500, 1200, 900, 600, 400],
        "game_seconds_remaining": [3600, 3500, 3400, 3300, 3000, 2700, 2400, 2200],
        "score_differential": [0] * 8,
        "down": [1, 2, 1, 2, 1, 2, 1, 2],
        "ydstogo": [10, 7, 10, 7, 10, 7, 10, 7],
        "yardline_100": [50, 43, 50, 43, 50, 43, 50, 43],
        "posteam_timeouts_remaining": [3, 2, 2, 2, 2, 2, 2, 2],
        "defteam_timeouts_remaining": [3] * 8,
        "receive_2h_ko": [1] * 8,
        "spread_line": [0.0] * 8,
        "timeout": [0, 1, 0, 0, 0, 0, 0, 0],
        "timeout_team": [None, "KC", None, None, None, None, None, None],
        "play_type": ["pass", "no_play", "pass", "run", "pass", "run", "pass", "run"],
        "desc": ["pass"] * 8,
    })


def test_full_pipeline_produces_scored_record(full_pbp):
    # detect_early_timeouts should find exactly 1 timeout (play_id=200)
    early = detect_early_timeouts(full_pbp)
    assert len(early) == 1

    # Remaining scrimmage plays: play_ids 300,400,500,600,700,800 (play_type pass/run)
    # half_seconds_remaining < 1700 (the timeout's value); that's all 6 remaining scrimmage plays
    # calculate_wp called once for opportunities: concat of 6 actual + 6 counterfactual = 12 rows
    # Then called once for scorer: 1 avoided + 1 penalty = 2 rows
    fake_wp_opps = pd.Series([0.50] * 12)   # 6 actual + 6 counterfactual
    fake_wp_score = pd.Series([0.55, 0.52])  # avoided-penalty, with-penalty

    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp_opps):
        opps = compute_all_opportunities(full_pbp, early)

    assert not opps.empty
    assert "wp_gain_counterfactual" in opps.columns
    assert (opps["ref_timeout_play_id"] == 200).all()

    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp_score):
        scores = score_timeouts(early, opps)

    assert len(scores) == 1
    assert scores.iloc[0]["verdict"] in {"justified", "wasteful"}
    assert "opportunity_cost" in scores.columns
    assert scores.iloc[0]["wp_gain_early"] == pytest.approx(0.03)  # 0.55 - 0.52


def test_full_pipeline_verdict_is_justified_when_early_gain_exceeds_best_opp(full_pbp):
    early = detect_early_timeouts(full_pbp)
    # Make early WP gain very high (0.20) so it beats any counterfactual opportunity
    fake_wp_opps = pd.Series([0.50] * 12)   # all counterfactual gains = 0
    fake_wp_score = pd.Series([0.70, 0.50])  # early gain = 0.20

    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp_opps):
        opps = compute_all_opportunities(full_pbp, early)

    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp_score):
        scores = score_timeouts(early, opps)

    assert scores.iloc[0]["verdict"] == "justified"
