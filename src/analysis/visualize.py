"""Charts for NFL timeout opportunity cost analysis."""
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd
import seaborn as sns


def plot_opportunity_cost_heatmap(scores: pd.DataFrame) -> matplotlib.figure.Figure:
    """Mean opportunity cost as a function of score_differential × half_seconds_remaining."""
    scores = scores.copy()
    scores["score_bucket"] = pd.cut(scores["score_differential"], bins=range(-21, 22, 7))
    scores["time_bucket"] = pd.cut(
        scores["half_seconds_remaining"],
        bins=[300, 600, 900, 1200, 1500, 1800],
        right=True,
    )
    pivot = scores.pivot_table(
        values="opportunity_cost",
        index="time_bucket",
        columns="score_bucket",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Mean Opportunity Cost by Score Differential and Time in Half")
    ax.set_xlabel("Score Differential (posteam perspective)")
    ax.set_ylabel("Half Seconds Remaining at Timeout Call")
    fig.tight_layout()
    return fig


def plot_cost_distribution(scores: pd.DataFrame) -> matplotlib.figure.Figure:
    """Histogram of opportunity costs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores["opportunity_cost"], bins=50, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Break-even")
    ax.set_xlabel("Opportunity Cost (WP)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Early Timeout Opportunity Costs")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_wasteful_by_coach(
    scores: pd.DataFrame, min_samples: int = 20
) -> matplotlib.figure.Figure:
    """% wasteful timeouts by coach, filtered to coaches with >= min_samples."""
    coach_stats = (
        scores.groupby("coach")
        .agg(
            total=("verdict", "count"),
            wasteful=("verdict", lambda s: (s == "wasteful").sum()),
        )
        .query("total >= @min_samples")
        .assign(pct_wasteful=lambda d: d["wasteful"] / d["total"] * 100)
        .sort_values("pct_wasteful", ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, max(4, len(coach_stats) * 0.35)))
    ax.barh(coach_stats["coach"], coach_stats["pct_wasteful"])
    ax.set_xlabel("% Wasteful Early Timeouts")
    ax.set_title(f"Wasteful Early Timeout Rate by Coach (min {min_samples} timeouts)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_cost_vs_timeouts_remaining(scores: pd.DataFrame) -> matplotlib.figure.Figure:
    """Box plot of opportunity cost by timeouts remaining at time of call."""
    groups = [
        scores.loc[
            scores["posteam_timeouts_remaining"] == n, "opportunity_cost"
        ].values
        for n in sorted(scores["posteam_timeouts_remaining"].unique())
    ]
    labels = sorted(scores["posteam_timeouts_remaining"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(groups, tick_labels=labels)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Timeouts Remaining at Call")
    ax.set_ylabel("Opportunity Cost (WP)")
    ax.set_title("Opportunity Cost vs. Timeouts Remaining")
    fig.tight_layout()
    return fig


def plot_best_missed_opportunity_timeline(scores: pd.DataFrame) -> matplotlib.figure.Figure:
    """Scatter: timeout call time (x) vs. best missed opportunity time (y), wasteful only."""
    wasteful = scores[scores["verdict"] == "wasteful"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        wasteful["half_seconds_remaining"],
        wasteful["best_opportunity_time"],
        alpha=0.4,
        s=20,
    )
    ax.plot([0, 1800], [0, 1800], "r--", linewidth=1, label="Diagonal (same moment)")
    ax.set_xlabel("Timeout Called At (half seconds remaining)")
    ax.set_ylabel("Best Opportunity At (half seconds remaining)")
    ax.set_title("When Was the Best Missed Opportunity?")
    ax.legend()
    fig.tight_layout()
    return fig
