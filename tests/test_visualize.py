import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from src.analysis.visualize import (
    plot_opportunity_cost_heatmap,
    plot_cost_distribution,
    plot_wasteful_by_coach,
    plot_cost_vs_timeouts_remaining,
    plot_best_missed_opportunity_timeline,
)


@pytest.fixture
def timeout_scores():
    return pd.DataFrame({
        "play_id": range(20),
        "season": [2023] * 20,
        "coach": (["Andy Reid"] * 12) + (["Dan Campbell"] * 8),
        "score_differential": list(range(-5, 15)),
        "half_seconds_remaining": [1800 - i * 60 for i in range(20)],
        "posteam_timeouts_remaining": [3, 2, 1] * 6 + [3, 2],
        "wp_gain_early": [0.02 + i * 0.001 for i in range(20)],
        "best_opportunity_value": [0.03 + i * 0.002 for i in range(20)],
        "best_opportunity_play_id": range(100, 120),
        "best_opportunity_time": [900 - i * 30 for i in range(20)],
        "opportunity_cost": [0.01 + i * 0.001 for i in range(20)],
        "verdict": ["wasteful"] * 15 + ["justified"] * 5,
    })


def test_heatmap_returns_figure(timeout_scores):
    fig = plot_opportunity_cost_heatmap(timeout_scores)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_cost_distribution_returns_figure(timeout_scores):
    fig = plot_cost_distribution(timeout_scores)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_wasteful_by_coach_returns_figure(timeout_scores):
    fig = plot_wasteful_by_coach(timeout_scores, min_samples=5)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_cost_vs_timeouts_returns_figure(timeout_scores):
    fig = plot_cost_vs_timeouts_remaining(timeout_scores)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_timeline_returns_figure(timeout_scores):
    fig = plot_best_missed_opportunity_timeline(timeout_scores)
    assert isinstance(fig, plt.Figure)
    plt.close("all")
