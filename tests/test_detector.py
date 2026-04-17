import pytest
from src.timeouts.detector import detect_early_timeouts, EARLY_HALF_THRESHOLD


def test_detects_early_offensive_timeout(sample_pbp):
    result = detect_early_timeouts(sample_pbp)
    # play_id 200: qtr 1, half_sec 1500 > 300, timeout by posteam, down 2, yard 43
    assert 200 in result["play_id"].values


def test_excludes_injury_timeout(sample_pbp):
    result = detect_early_timeouts(sample_pbp)
    # play_id 600: contains "INJURY" in desc
    assert 600 not in result["play_id"].values


def test_excludes_timeouts_under_threshold(sample_pbp):
    result = detect_early_timeouts(sample_pbp)
    assert all(result["half_seconds_remaining"] > EARLY_HALF_THRESHOLD)


def test_excludes_fourth_down(sample_pbp):
    sample_pbp.loc[sample_pbp["play_id"] == 200, "down"] = 4
    result = detect_early_timeouts(sample_pbp)
    assert 200 not in result["play_id"].values


def test_excludes_goal_line(sample_pbp):
    sample_pbp.loc[sample_pbp["play_id"] == 200, "yardline_100"] = 3
    result = detect_early_timeouts(sample_pbp)
    assert 200 not in result["play_id"].values


def test_output_has_excluded_column(sample_pbp):
    result = detect_early_timeouts(sample_pbp)
    assert "excluded" in result.columns
    assert not result["excluded"].any()
