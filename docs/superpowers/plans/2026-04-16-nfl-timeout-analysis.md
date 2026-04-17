# NFL Timeout Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that identifies early NFL timeouts called to avoid pre-snap penalties, computes the WP gain of each actual use, and compares it against the best available timeout use later in the same half.

**Architecture:** Five focused modules with a linear data flow — `loader → detector → calculator → opportunities → scorer` — plus a visualizer. All modules are tested with mocked external dependencies (nflreadpy, rpy2) to keep tests fast and environment-independent.

**Tech Stack:** Python 3.11, nflreadpy, rpy2 (nflfastR bridge), pandas, pyarrow, matplotlib, seaborn, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/data/loader.py` | Fetch PBP + schedules via nflreadpy, join coach, validate fields, cache as parquet |
| `src/wp/calculator.py` | rpy2 wrapper; initializes R session once; batches game states to nflfastR |
| `src/timeouts/detector.py` | Filter PBP to early offensive timeouts; apply exclusion rules |
| `src/timeouts/opportunities.py` | For each early timeout, compute counterfactual WP gain for every remaining half play |
| `src/timeouts/scorer.py` | Pair each timeout with its best opportunity; compute `opportunity_cost` and `verdict` |
| `src/analysis/visualize.py` | Generate heatmap and four supporting charts from `TimeoutScore` records |
| `tests/conftest.py` | Shared pytest fixtures: `sample_pbp`, `sample_schedules`, `sample_early_timeouts`, `sample_opportunities` |
| `tests/test_loader.py` | Unit tests for loader with mocked nflreadpy |
| `tests/test_calculator.py` | Unit tests for WP calculator with mocked rpy2 |
| `tests/test_detector.py` | Unit tests for timeout detection logic |
| `tests/test_opportunities.py` | Unit tests for opportunity computation with mocked calculate_wp |
| `tests/test_scorer.py` | Unit tests for scoring + opportunity cost |
| `tests/test_visualize.py` | Smoke tests for chart generation |

---

## Task 0: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/wp/__init__.py`, `src/timeouts/__init__.py`, `src/analysis/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/jaredschwartz/Documents/workspace/nfl-timeout-analysis
mkdir -p src/data src/wp src/timeouts src/analysis
mkdir -p tests data/processed reports notebooks
touch src/__init__.py src/data/__init__.py src/wp/__init__.py
touch src/timeouts/__init__.py src/analysis/__init__.py
touch tests/__init__.py
```

- [ ] **Step 2: Create `requirements.txt`**

```text
nflreadpy>=0.1.0
rpy2>=3.5.0
pandas>=2.0.0
pyarrow>=14.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
jupyter>=1.0.0
pytest>=7.0.0
```

- [ ] **Step 3: Create `.gitignore`**

```text
data/
*.parquet
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
reports/
.env
```

- [ ] **Step 4: Create virtual environment and install dependencies**

```bash
cd /Users/jaredschwartz/Documents/workspace/nfl-timeout-analysis
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages install without error. rpy2 requires R to be installed at the system level — if `brew install r` hasn't been run, do that first, then `install.packages(c("nflfastR", "nflreadr"))` from an R session.

- [ ] **Step 5: Verify pytest runs**

```bash
source .venv/bin/activate
pytest --collect-only
```

Expected: "no tests ran" (no tests exist yet).

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt .gitignore src/ tests/
git commit -m "feat: initial project scaffold"
```

---

## Task 1: Data Loader

**Files:**
- Create: `src/data/loader.py`
- Create: `tests/test_loader.py`

`★ Insight ─────────────────────────────────────`
Coach data is obtained from `nflreadpy.load_schedules()`, which has `home_coach`/`away_coach` columns per game — cleaner than a separate coach table. The schedule join works on `game_id`, which is present in both the PBP data and the schedules response.
`─────────────────────────────────────────────────`

- [ ] **Step 1: Write failing tests**

`tests/test_loader.py`:
```python
from unittest.mock import patch, MagicMock
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.loader'`

- [ ] **Step 3: Implement `src/data/loader.py`**

```python
"""Fetch NFL play-by-play data and cache locally as parquet."""
import pandas as pd
import nflreadpy
from pathlib import Path

CACHE_DIR = Path("data/processed")
REQUIRED_FIELDS = ["wp", "half_seconds_remaining", "posteam_timeouts_remaining"]


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
    sched = schedules[["game_id", "home_team", "away_team", "home_coach", "away_coach"]]
    merged = pbp.merge(sched, on="game_id", how="left")
    merged["coach"] = merged.apply(
        lambda r: r["home_coach"] if r["posteam"] == r["home_team"] else r["away_coach"],
        axis=1,
    )
    return merged.drop(columns=["home_coach", "away_coach"])


def _validate(pbp: pd.DataFrame, seasons: list[int]) -> None:
    """Print warnings for seasons with high rates of missing critical fields."""
    for season in seasons:
        season_df = pbp[pbp["season"] == season]
        missing_rate = season_df[REQUIRED_FIELDS].isna().any(axis=1).mean()
        if missing_rate > 0.05:
            print(
                f"Warning: season {season} has {missing_rate:.1%} plays "
                "missing required fields"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_loader.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: add data loader with nflreadpy fetch, parquet cache, and coach join"
```

---

## Task 2: Win Probability Calculator

**Files:**
- Create: `src/wp/calculator.py`
- Create: `tests/test_calculator.py`

`★ Insight ─────────────────────────────────────`
The R session is initialized once at module import time via a module-level `_nflfastr` singleton — starting R is expensive (~2–4 seconds). The `pandas2ri.activate()` call installs a global conversion hook so that pandas DataFrames are automatically converted when passed to R functions.
`─────────────────────────────────────────────────`

- [ ] **Step 1: Write failing tests**

`tests/test_calculator.py`:
```python
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

WP_COLS = [
    "score_differential", "half_seconds_remaining", "game_seconds_remaining",
    "down", "ydstogo", "yardline_100",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "receive_2h_ko", "spread_line",
]

@pytest.fixture
def sample_states():
    return pd.DataFrame({
        "score_differential": [0, 7],
        "half_seconds_remaining": [900, 600],
        "game_seconds_remaining": [1800, 1200],
        "down": [2, 1],
        "ydstogo": [7, 10],
        "yardline_100": [50, 40],
        "posteam_timeouts_remaining": [3, 2],
        "defteam_timeouts_remaining": [3, 3],
        "receive_2h_ko": [1, 0],
        "spread_line": [0.0, -3.0],
    })

def test_calculate_wp_returns_series_same_length(sample_states):
    mock_r_result = pd.DataFrame({"wp": [0.52, 0.65]})
    with patch("src.wp.calculator.importr"), \
         patch("src.wp.calculator.pandas2ri"), \
         patch("src.wp.calculator._get_nflfastr") as mock_get:
        mock_nflfastr = MagicMock()
        mock_nflfastr.calculate_win_probability.return_value = MagicMock()
        mock_get.return_value = mock_nflfastr
        with patch("src.wp.calculator.pandas2ri") as mock_p2r:
            mock_p2r.rpy2py.return_value = mock_r_result
            from src.wp.calculator import calculate_wp
            result = calculate_wp(sample_states)
    assert len(result) == len(sample_states)

def test_calculate_wp_index_matches_input(sample_states):
    sample_states.index = [10, 20]
    mock_r_result = pd.DataFrame({"wp": [0.52, 0.65]})
    with patch("src.wp.calculator.importr"), \
         patch("src.wp.calculator._get_nflfastr") as mock_get, \
         patch("src.wp.calculator.pandas2ri") as mock_p2r:
        mock_nflfastr = MagicMock()
        mock_nflfastr.calculate_win_probability.return_value = MagicMock()
        mock_get.return_value = mock_nflfastr
        mock_p2r.rpy2py.return_value = mock_r_result
        from src.wp import calculator
        import importlib; importlib.reload(calculator)
        result = calculator.calculate_wp(sample_states)
    assert list(result.index) == [10, 20]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_calculator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.wp.calculator'`

- [ ] **Step 3: Implement `src/wp/calculator.py`**

```python
"""rpy2 wrapper around nflfastR's calculate_win_probability."""
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()

WP_INPUT_COLS = [
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "receive_2h_ko",
    "spread_line",
]

_nflfastr = None


def _get_nflfastr():
    global _nflfastr
    if _nflfastr is None:
        _nflfastr = importr("nflfastR")
    return _nflfastr


def calculate_wp(states: pd.DataFrame) -> pd.Series:
    """
    Compute win probability for a batch of game states.

    Returns a Series with the same index as `states`.
    The R session is initialized lazily on first call.
    """
    nflfastr = _get_nflfastr()
    r_df = pandas2ri.py2rpy(states[WP_INPUT_COLS].reset_index(drop=True))
    r_result = nflfastr.calculate_win_probability(r_df)
    result_df = pandas2ri.rpy2py(r_result)
    return pd.Series(result_df["wp"].values, index=states.index)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_calculator.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wp/calculator.py tests/test_calculator.py
git commit -m "feat: add rpy2 wrapper for nflfastR win probability calculation"
```

---

## Task 3: Timeout Detector

**Files:**
- Create: `src/timeouts/detector.py`
- Create: `tests/conftest.py` (shared fixtures)
- Create: `tests/test_detector.py`

- [ ] **Step 1: Create `tests/conftest.py`**

```python
import pandas as pd
import pytest

@pytest.fixture
def sample_pbp():
    """Minimal PBP DataFrame with columns needed across all tests."""
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET"] * 6,
        "play_id": [100, 200, 300, 400, 500, 600],
        "season": [2023] * 6,
        "posteam": ["KC"] * 6,
        "defteam": ["DET"] * 6,
        "coach": ["Andy Reid"] * 6,
        "qtr": [1, 1, 2, 2, 3, 3],
        "half_seconds_remaining": [1800, 1500, 900, 600, 1800, 400],
        "game_seconds_remaining": [3600, 3300, 1800, 1500, 1800, 400],
        "score_differential": [0, 0, 0, 0, 7, 7],
        "down": [1, 2, 2, 1, 1, 1],
        "ydstogo": [10, 7, 7, 10, 10, 10],
        "yardline_100": [50, 43, 43, 50, 50, 50],
        "posteam_timeouts_remaining": [3, 3, 2, 2, 3, 3],
        "defteam_timeouts_remaining": [3, 3, 3, 3, 3, 3],
        "receive_2h_ko": [1, 1, 1, 1, 0, 0],
        "spread_line": [0.0] * 6,
        "timeout": [0, 1, 1, 0, 1, 1],
        "timeout_team": [None, "KC", "KC", None, "KC", "KC"],
        "play_type": ["pass", "no_play", "no_play", "pass", "no_play", "no_play"],
        "desc": [
            "Pass incomplete",
            "Timeout #2 by KC",
            "Timeout #1 by KC",
            "Run for 5",
            "Timeout #3 by KC",
            "Timeout #3 by KC INJURY",
        ],
    })
```

- [ ] **Step 2: Write failing tests**

`tests/test_detector.py`:
```python
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
    # play_id 300: half_seconds_remaining=900, but play_id 300 qtr=2 half_sec=900 > 300 -> included
    # play_id 500: half_sec=1800, qtr=3, timeout by KC -> included
    # Check that no timeouts with half_sec <= EARLY_HALF_THRESHOLD appear
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
    assert result["excluded"].all() == False
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_detector.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.timeouts.detector'`

- [ ] **Step 4: Implement `src/timeouts/detector.py`**

```python
"""Identify early penalty-avoidance timeout events from NFL play-by-play data."""
import pandas as pd

EARLY_HALF_THRESHOLD = 300  # seconds; timeouts after this are clock management
GOAL_LINE_THRESHOLD = 5     # yards to end zone; inside this = strategic timeout
INJURY_PATTERN = r"(?i)\binjury\b"

_OUTPUT_COLS = [
    "game_id", "play_id", "season", "posteam", "coach",
    "qtr", "half_seconds_remaining", "game_seconds_remaining",
    "score_differential", "down", "ydstogo", "yardline_100",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "receive_2h_ko", "spread_line",
]


def detect_early_timeouts(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Return plays where an offensive team called an early timeout.

    Excludes injury timeouts, 4th-down timeouts, and goal-line timeouts.
    The `posteam_timeouts_remaining` column reflects the count *after* the
    timeout was consumed (i.e., N−1 where N was the count before the call).
    """
    mask = (
        (pbp["timeout"] == 1)
        & (pbp["timeout_team"] == pbp["posteam"])
        & (pbp["half_seconds_remaining"] > EARLY_HALF_THRESHOLD)
        & (pbp["down"] != 4)
        & (pbp["yardline_100"] > GOAL_LINE_THRESHOLD)
        & ~pbp["desc"].str.contains(INJURY_PATTERN, regex=True, na=False)
    )

    result = pbp.loc[mask, _OUTPUT_COLS].copy()
    result["excluded"] = False
    return result.reset_index(drop=True)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_detector.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/timeouts/detector.py tests/conftest.py tests/test_detector.py
git commit -m "feat: add timeout detector with injury/4th-down/goal-line exclusions"
```

---

## Task 4: Opportunity Computer

**Files:**
- Create: `src/timeouts/opportunities.py`
- Create: `tests/test_opportunities.py`

`★ Insight ─────────────────────────────────────`
The key WP calculation here is: WP(M+1 timeouts, actual_time + clock_runoff) − WP(M timeouts, actual_time). The `clock_runoff` for a play is how much game clock elapsed *between the end of the previous play and the start of this one* — time that a well-timed timeout would have saved. Both calls are batched into a single DataFrame passed to `calculate_wp` to minimize R round-trips.
`─────────────────────────────────────────────────`

- [ ] **Step 1: Write failing tests**

`tests/test_opportunities.py`:
```python
from unittest.mock import patch
import pandas as pd
import pytest
from src.timeouts.opportunities import compute_all_opportunities

@pytest.fixture
def early_timeout(sample_pbp):
    """Single early timeout record from play_id=200."""
    return pd.Series({
        "game_id": "2023_01_KC_DET",
        "play_id": 200,
        "season": 2023,
        "posteam": "KC",
        "qtr": 1,
        "half_seconds_remaining": 1500,
        "game_seconds_remaining": 3300,
        "score_differential": 0,
        "down": 2,
        "ydstogo": 7,
        "yardline_100": 43,
        "posteam_timeouts_remaining": 2,  # after the timeout was called
        "defteam_timeouts_remaining": 3,
        "receive_2h_ko": 1,
        "spread_line": 0.0,
        "coach": "Andy Reid",
        "excluded": False,
    })

def test_compute_all_opportunities_returns_dataframe(sample_pbp, early_timeout):
    early_timeouts = early_timeout.to_frame().T.reset_index(drop=True)
    fake_wp = pd.Series([0.50, 0.52, 0.53, 0.55])  # actual + counterfactual for 2 plays
    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp):
        result = compute_all_opportunities(sample_pbp, early_timeouts)
    assert isinstance(result, pd.DataFrame)
    assert "ref_timeout_play_id" in result.columns
    assert "wp_gain_counterfactual" in result.columns

def test_opportunities_link_to_timeout(sample_pbp, early_timeout):
    early_timeouts = early_timeout.to_frame().T.reset_index(drop=True)
    fake_wp = pd.Series([0.50, 0.52, 0.53, 0.55])
    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp):
        result = compute_all_opportunities(sample_pbp, early_timeouts)
    assert (result["ref_timeout_play_id"] == 200).all()

def test_clock_runoff_is_non_negative(sample_pbp, early_timeout):
    early_timeouts = early_timeout.to_frame().T.reset_index(drop=True)
    fake_wp = pd.Series([0.50, 0.52, 0.53, 0.55])
    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp):
        result = compute_all_opportunities(sample_pbp, early_timeouts)
    assert (result["clock_runoff"] >= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_opportunities.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.timeouts.opportunities'`

- [ ] **Step 3: Implement `src/timeouts/opportunities.py`**

```python
"""Compute per-play counterfactual timeout value for each early timeout."""
import pandas as pd
from src.wp.calculator import calculate_wp, WP_INPUT_COLS

_SCRIMMAGE_PLAY_TYPES = {"pass", "run", "qb_kneel", "qb_spike"}


def compute_all_opportunities(
    pbp: pd.DataFrame,
    early_timeouts: pd.DataFrame,
) -> pd.DataFrame:
    """For every early timeout, compute WP gain at each remaining half play."""
    results = [
        _opportunities_for_timeout(pbp, row)
        for _, row in early_timeouts.iterrows()
    ]
    non_empty = [df for df in results if not df.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def _opportunities_for_timeout(pbp: pd.DataFrame, timeout: pd.Series) -> pd.DataFrame:
    half_qtrs = [1, 2] if timeout["qtr"] <= 2 else [3, 4]
    remaining = pbp[
        (pbp["game_id"] == timeout["game_id"])
        & (pbp["qtr"].isin(half_qtrs))
        & (pbp["half_seconds_remaining"] < timeout["half_seconds_remaining"])
        & (pbp["play_type"].isin(_SCRIMMAGE_PLAY_TYPES))
    ].sort_values("half_seconds_remaining", ascending=False).copy()

    if remaining.empty:
        return pd.DataFrame()

    # Clock runoff: time that ran between this play's start and the previous play's end.
    # A timeout called here would stop the clock and recover this time.
    prev_half_sec = remaining["half_seconds_remaining"].shift(1, fill_value=timeout["half_seconds_remaining"])
    remaining["clock_runoff"] = (prev_half_sec - remaining["half_seconds_remaining"]).clip(lower=0)

    # Build actual game states
    actual = remaining[WP_INPUT_COLS].copy()

    # Build counterfactual: +1 timeout, +clock_runoff seconds on clock
    counterfactual = actual.copy()
    counterfactual["posteam_timeouts_remaining"] = (
        remaining["posteam_timeouts_remaining"] + 1
    ).clip(upper=3)
    counterfactual["half_seconds_remaining"] = (
        remaining["half_seconds_remaining"] + remaining["clock_runoff"]
    ).clip(upper=prev_half_sec.values)
    counterfactual["game_seconds_remaining"] = (
        remaining["game_seconds_remaining"] + remaining["clock_runoff"]
    )

    n = len(remaining)
    combined = pd.concat([actual, counterfactual], ignore_index=True)
    wp = calculate_wp(combined)

    wp_actual = wp.iloc[:n].values
    wp_counterfactual = wp.iloc[n:].values

    return pd.DataFrame({
        "ref_timeout_play_id": timeout["play_id"],
        "play_id": remaining["play_id"].values,
        "half_seconds_remaining": remaining["half_seconds_remaining"].values,
        "clock_runoff": remaining["clock_runoff"].values,
        "actual_timeouts": remaining["posteam_timeouts_remaining"].values,
        "wp_gain_counterfactual": wp_counterfactual - wp_actual,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_opportunities.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/timeouts/opportunities.py tests/test_opportunities.py
git commit -m "feat: add opportunity computer with batched counterfactual WP calculation"
```

---

## Task 5: Scorer

**Files:**
- Create: `src/timeouts/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_scorer.py`:
```python
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
        "coach": ["Andy Reid"],
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
        "wp_gain_counterfactual": [0.02, 0.08, 0.05],  # play 400 is the best
    })

def test_score_timeouts_has_required_columns(early_timeouts, opportunities, sample_pbp):
    fake_wp = pd.Series([0.55, 0.52])  # avoided-penalty, with-penalty
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, sample_pbp, opportunities)
    required = [
        "play_id", "wp_gain_early", "best_opportunity_value",
        "best_opportunity_play_id", "best_opportunity_time",
        "opportunity_cost", "verdict",
    ]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"

def test_best_opportunity_is_maximum(early_timeouts, opportunities, sample_pbp):
    fake_wp = pd.Series([0.55, 0.52])
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, sample_pbp, opportunities)
    assert result.iloc[0]["best_opportunity_play_id"] == 400
    assert result.iloc[0]["best_opportunity_value"] == pytest.approx(0.08)

def test_opportunity_cost_calculation(early_timeouts, opportunities, sample_pbp):
    # wp_gain_early = 0.55 - 0.52 = 0.03; best_opp = 0.08; cost = 0.08 - 0.03 = 0.05
    fake_wp = pd.Series([0.55, 0.52])
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, sample_pbp, opportunities)
    assert result.iloc[0]["opportunity_cost"] == pytest.approx(0.05)

def test_verdict_justified_when_cost_nonpositive(early_timeouts, opportunities, sample_pbp):
    # Make early wp_gain > best opportunity value -> cost <= 0 -> justified
    fake_wp = pd.Series([0.60, 0.50])  # wp_gain_early = 0.10 > 0.08
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, sample_pbp, opportunities)
    assert result.iloc[0]["verdict"] == "justified"

def test_verdict_wasteful_when_cost_positive(early_timeouts, opportunities, sample_pbp):
    fake_wp = pd.Series([0.55, 0.52])  # wp_gain_early = 0.03 < 0.08 -> wasteful
    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp):
        result = score_timeouts(early_timeouts, sample_pbp, opportunities)
    assert result.iloc[0]["verdict"] == "wasteful"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scorer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.timeouts.scorer'`

- [ ] **Step 3: Implement `src/timeouts/scorer.py`**

```python
"""Pair each EarlyTimeout with its best opportunity and compute opportunity cost."""
import pandas as pd
from src.wp.calculator import calculate_wp, WP_INPUT_COLS


def score_timeouts(
    early_timeouts: pd.DataFrame,
    pbp: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one TimeoutScore record per early timeout.

    Computes wp_gain_early (avoided-penalty vs with-penalty WP delta) and
    pairs each timeout with its maximum-value remaining opportunity.
    """
    scored = _add_early_wp_gain(early_timeouts)

    best_opps = (
        opportunities.sort_values("wp_gain_counterfactual", ascending=False)
        .groupby("ref_timeout_play_id", sort=False)
        .first()
        .reset_index()
        .rename(columns={
            "ref_timeout_play_id": "play_id",
            "play_id": "best_opportunity_play_id",
            "half_seconds_remaining": "best_opportunity_time",
            "wp_gain_counterfactual": "best_opportunity_value",
        })[["play_id", "best_opportunity_play_id", "best_opportunity_time", "best_opportunity_value"]]
    )

    result = scored.merge(best_opps, on="play_id", how="left")
    result["opportunity_cost"] = result["best_opportunity_value"] - result["wp_gain_early"]
    result["verdict"] = result["opportunity_cost"].apply(
        lambda x: "justified" if x <= 0 else "wasteful"
    )
    return result


def _add_early_wp_gain(early_timeouts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute WP gain for each early timeout call.

    Avoided-penalty state: actual game state, posteam_timeouts_remaining (already N-1
    after timeout consumed).
    With-penalty state: ydstogo+5, yardline_100+5, posteam_timeouts_remaining+1 (N, as
    if timeout had not been called).
    """
    avoided = early_timeouts[WP_INPUT_COLS].copy()

    penalty = avoided.copy()
    penalty["ydstogo"] = (early_timeouts["ydstogo"] + 5).clip(upper=99)
    penalty["yardline_100"] = (early_timeouts["yardline_100"] + 5).clip(upper=99)
    penalty["posteam_timeouts_remaining"] = (
        early_timeouts["posteam_timeouts_remaining"] + 1
    ).clip(upper=3)

    n = len(early_timeouts)
    combined = pd.concat([avoided, penalty], ignore_index=True)
    wp = calculate_wp(combined)

    result = early_timeouts.copy()
    result["wp_gain_early"] = wp.iloc[:n].values - wp.iloc[n:].values
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scorer.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/timeouts/scorer.py tests/test_scorer.py
git commit -m "feat: add scorer that pairs timeouts with best opportunities and computes verdicts"
```

---

## Task 6: Visualizer

**Files:**
- Create: `src/analysis/visualize.py`
- Create: `tests/test_visualize.py`

- [ ] **Step 1: Write failing tests**

`tests/test_visualize.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_visualize.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.analysis.visualize'`

- [ ] **Step 3: Implement `src/analysis/visualize.py`**

```python
"""Charts and summary tables for NFL timeout opportunity cost analysis."""
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd
import seaborn as sns


def plot_opportunity_cost_heatmap(scores: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Heatmap: mean opportunity cost as a function of score_differential × half_seconds_remaining.
    Answers: under what conditions is it knowable that the early timeout is a mistake?
    """
    scores = scores.copy()
    scores["score_bucket"] = pd.cut(scores["score_differential"], bins=range(-21, 22, 7))
    scores["time_bucket"] = pd.cut(
        scores["half_seconds_remaining"], bins=[300, 600, 900, 1200, 1500, 1800], right=True
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
    """Histogram of opportunity costs across all early timeouts."""
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
    """Bar chart of % wasteful timeouts by coach (coaches with >= min_samples only)."""
    coach_stats = (
        scores.groupby("coach")
        .agg(total=("verdict", "count"), wasteful=("verdict", lambda s: (s == "wasteful").sum()))
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
    """Box plot of opportunity cost grouped by timeouts remaining at time of call."""
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [
        scores.loc[scores["posteam_timeouts_remaining"] == n, "opportunity_cost"].values
        for n in sorted(scores["posteam_timeouts_remaining"].unique())
    ]
    labels = sorted(scores["posteam_timeouts_remaining"].unique())
    ax.boxplot(groups, labels=labels)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Timeouts Remaining at Call")
    ax.set_ylabel("Opportunity Cost (WP)")
    ax.set_title("Opportunity Cost vs. Timeouts Remaining")
    fig.tight_layout()
    return fig


def plot_best_missed_opportunity_timeline(scores: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    For wasteful timeouts only: where in the half was the best missed opportunity?
    Scatter: actual timeout time (x) vs. best opportunity time (y).
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_visualize.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/visualize.py tests/test_visualize.py
git commit -m "feat: add five visualizations for timeout opportunity cost analysis"
```

---

## Task 7: Full Test Suite + Integration Smoke Test

**Files:**
- Modify: `tests/conftest.py` — add integration fixture
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration smoke test**

`tests/test_integration.py`:
```python
"""End-to-end smoke test using minimal fake data and mocked external calls."""
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from src.timeouts.detector import detect_early_timeouts
from src.timeouts.opportunities import compute_all_opportunities
from src.timeouts.scorer import score_timeouts

@pytest.fixture
def full_pbp():
    """Realistic minimal PBP with one early timeout and several remaining plays."""
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET"] * 8,
        "play_id": [100, 200, 300, 400, 500, 600, 700, 800],
        "season": [2023] * 8,
        "posteam": ["KC"] * 8,
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

def test_full_pipeline(full_pbp):
    fake_wp_early = pd.Series([0.55, 0.52])       # avoided, with-penalty
    fake_wp_opps = pd.Series([0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
                               0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58])

    early = detect_early_timeouts(full_pbp)
    assert len(early) == 1, f"Expected 1 early timeout, got {len(early)}"

    with patch("src.timeouts.opportunities.calculate_wp", return_value=fake_wp_opps):
        opps = compute_all_opportunities(full_pbp, early)
    assert not opps.empty
    assert "wp_gain_counterfactual" in opps.columns

    with patch("src.timeouts.scorer.calculate_wp", return_value=fake_wp_early):
        scores = score_timeouts(early, full_pbp, opps)

    assert len(scores) == 1
    assert scores.iloc[0]["verdict"] in {"justified", "wasteful"}
    assert "opportunity_cost" in scores.columns
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS (2 calculator + 3 loader + 6 detector + 3 opportunities + 5 scorer + 5 visualize + 1 integration = 25 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration smoke test for full pipeline"
```

---

## Task 8: Save Results + Notebook

**Files:**
- Create: `src/timeouts/scorer.py` — add `save_scores()` helper (append to existing file)
- Create: `notebooks/analysis.ipynb`

- [ ] **Step 1: Add `save_scores` to `src/timeouts/scorer.py`**

Append to the end of `src/timeouts/scorer.py`:
```python
def save_scores(scores: pd.DataFrame, path: str = "data/processed/timeout_scores.parquet") -> None:
    """Persist TimeoutScore records for downstream analysis."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False)
```

- [ ] **Step 2: Create `notebooks/analysis.ipynb`**

Create a notebook with the following cells (use `jupyter nbformat` or just create the file):

```json
{
 "cells": [
  {
   "cell_type": "code",
   "source": [
    "import sys\nsys.path.insert(0, '..')\n",
    "from src.data.loader import load_seasons\n",
    "from src.timeouts.detector import detect_early_timeouts\n",
    "from src.timeouts.opportunities import compute_all_opportunities\n",
    "from src.timeouts.scorer import score_timeouts, save_scores\n",
    "from src.analysis.visualize import (\n",
    "    plot_opportunity_cost_heatmap,\n",
    "    plot_cost_distribution,\n",
    "    plot_wasteful_by_coach,\n",
    "    plot_cost_vs_timeouts_remaining,\n",
    "    plot_best_missed_opportunity_timeline,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "SEASONS = list(range(2010, 2024))\n",
    "pbp = load_seasons(SEASONS)"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "early = detect_early_timeouts(pbp)\n",
    "print(f'{len(early)} early timeouts detected')"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "opps = compute_all_opportunities(pbp, early)\n",
    "print(f'{len(opps)} opportunities computed')"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "scores = score_timeouts(early, pbp, opps)\n",
    "save_scores(scores)\n",
    "print(scores['verdict'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "fig = plot_opportunity_cost_heatmap(scores)\n",
    "fig.savefig('../reports/heatmap.png', dpi=150, bbox_inches='tight')"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "for name, fig in [\n",
    "    ('distribution', plot_cost_distribution(scores)),\n",
    "    ('by_coach', plot_wasteful_by_coach(scores, min_samples=20)),\n",
    "    ('vs_timeouts', plot_cost_vs_timeouts_remaining(scores)),\n",
    "    ('timeline', plot_best_missed_opportunity_timeline(scores)),\n",
    "]:\n",
    "    fig.savefig(f'../reports/{name}.png', dpi=150, bbox_inches='tight')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest tests/ -v
```

Expected: All 25 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/timeouts/scorer.py notebooks/analysis.ipynb
git commit -m "feat: add save_scores helper and analysis notebook"
```

---

## Self-Review

### Spec Coverage

| Spec requirement | Covered by |
|-----------------|-----------|
| EarlyTimeout data structure | Task 3: detector.py |
| TimeoutOpportunity data structure | Task 4: opportunities.py |
| TimeoutScore data structure | Task 5: scorer.py |
| Penalty-avoidance classification (offensive, early, non-injury, non-4th, non-goal) | Task 3 |
| `wp_gain_early` = WP(avoided) − WP(with-penalty) | Task 5: `_add_early_wp_gain` |
| Clock correction in counterfactual | Task 4: `clock_runoff` |
| `opportunity_cost` = best_opp_value − wp_gain_early | Task 5 |
| Heatmap (opp cost × score_diff × time-in-half) | Task 6 |
| Histogram, coach bar chart, vs-timeouts box plot, timeline scatter | Task 6 |
| Parquet output | Task 8 |
| nflreadpy data fetch + parquet cache | Task 1 |
| rpy2 nflfastR wrapper with lazy R init | Task 2 |
| Coach join from schedules | Task 1 |
| Per-season field validation with warning | Task 1 |
| `EARLY_HALF_THRESHOLD` as configurable constant | Task 3 |
| Batch WP calls as dataframes | Tasks 4, 5 |

### Type Consistency

- `calculate_wp` input uses `WP_INPUT_COLS` (imported by both `opportunities.py` and `scorer.py`) — consistent.
- `detect_early_timeouts` output uses nflverse column names (`score_differential`, `posteam_timeouts_remaining`) throughout — scorer accesses these directly without renaming — consistent.
- `ref_timeout_play_id` in opportunities matches `play_id` in early_timeouts — used in scorer's merge — consistent.
- All fixtures in conftest.py match the column names expected by detector, opportunities, and scorer — consistent.

### No Placeholders Found

All steps contain actual code, commands, and expected outputs.
