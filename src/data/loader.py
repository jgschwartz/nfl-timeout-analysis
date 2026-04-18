"""Fetch NFL play-by-play data and cache locally as parquet."""
import time
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
DOWNLOAD_RETRIES = 3
RETRY_BACKOFF = 5.0  # seconds; doubles on each retry


def load_seasons(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """Load PBP for given seasons from per-season parquet cache or nflreadpy."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = [_load_one_season(s, force_refresh) for s in seasons]
    pbp = pd.concat(frames, ignore_index=True)
    _validate(pbp, seasons)
    return pbp


def _load_one_season(season: int, force_refresh: bool) -> pd.DataFrame:
    """Load a single season from cache, downloading with retries if needed."""
    cache_path = CACHE_DIR / f"pbp_{season}.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    df = _download_with_retry(season)
    df = _add_coach(df)
    df.to_parquet(cache_path)
    return df


def _download_with_retry(season: int) -> pd.DataFrame:
    """Download one season's PBP from nflreadpy, retrying on connection errors."""
    backoff = RETRY_BACKOFF
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            return _to_pandas(nflreadpy.load_pbp([season]))
        except Exception as exc:
            if attempt == DOWNLOAD_RETRIES:
                raise
            print(
                f"Download failed for {season} "
                f"(attempt {attempt}/{DOWNLOAD_RETRIES}): {exc}. "
                f"Retrying in {backoff:.0f}s…"
            )
            time.sleep(backoff)
            backoff *= 2


def _to_pandas(df) -> pd.DataFrame:
    """Convert a Polars DataFrame to pandas if needed; pass pandas through unchanged."""
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _add_coach(pbp: pd.DataFrame) -> pd.DataFrame:
    """Add `coach` column from the home_coach/away_coach columns already in PBP."""
    pbp = pbp.copy()
    pbp["coach"] = pbp.apply(
        lambda r: r["home_coach"] if r["posteam"] == r["home_team"] else r["away_coach"],
        axis=1,
    )
    return pbp


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
