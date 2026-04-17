"""Fetch NFL play-by-play data and cache locally as parquet."""
import pandas as pd
from pathlib import Path

# Lazily resolved so tests can patch before the real module is imported.
# Access via the module-level name so patch("src.data.loader.nflreadpy") works.
try:
    import nflreadpy
except ModuleNotFoundError:
    nflreadpy = None  # type: ignore[assignment]

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
REQUIRED_FIELDS = ["wp", "half_seconds_remaining", "posteam_timeouts_remaining"]
MISSING_FIELD_THRESHOLD = 0.05


def load_seasons(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """Load PBP for given seasons from cache or nflreadpy, with coach joined."""
    cache_path = CACHE_DIR / f"pbp_{min(seasons)}_{max(seasons)}.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    pbp = pd.concat(
        [nflreadpy.load_pbp([s]) for s in seasons],
        ignore_index=True,
    )
    schedules = nflreadpy.load_schedules(seasons)
    pbp = _join_coaches(pbp, schedules)
    _validate(pbp, seasons)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pbp.to_parquet(cache_path)
    return pbp


def _join_coaches(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Add `coach` column (head coach of the possession team) from schedules."""
    sched = (
        schedules[["game_id", "home_team", "away_team", "home_coach", "away_coach"]]
        .rename(columns={"home_team": "sched_home_team", "away_team": "sched_away_team"})
    )
    merged = pbp.merge(sched, on="game_id", how="left")
    merged["coach"] = merged.apply(
        lambda r: r["home_coach"] if r["posteam"] == r["sched_home_team"] else r["away_coach"],
        axis=1,
    )
    return merged.drop(columns=["home_coach", "away_coach", "sched_home_team", "sched_away_team"])


def _validate(pbp: pd.DataFrame, seasons: list[int]) -> None:
    """Print warnings for seasons with high rates of missing critical fields."""
    for season in seasons:
        season_df = pbp[pbp["season"] == season]
        missing_rate = season_df[REQUIRED_FIELDS].isna().any(axis=1).mean()
        if missing_rate > MISSING_FIELD_THRESHOLD:
            print(
                f"Warning: season {season} has {missing_rate:.1%} plays "
                "missing required fields"
            )
