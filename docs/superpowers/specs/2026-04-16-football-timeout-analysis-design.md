# NFL Timeout Quality Analysis — Design Spec
**Date:** 2026-04-16
**Phase:** 1 — Timeout Quality Modeling

---

## Goal

Determine whether NFL coaches calling timeouts early in a half (to avoid pre-snap penalties) are making good decisions given the information available at the time of the call. Quantify the opportunity cost of each early timeout use: the difference between the WP gain of the actual use versus the highest-value moment available later in the same half.

---

## Scope

**In scope (Phase 1):**
- Timeouts called with more than 5–6 minutes remaining in either half
- Classified as penalty-avoidance timeouts (assumed pre-snap, 5-yard penalty avoided)
- Full NFL dataset 1999–present, both halves, all four quarters
- Opportunity cost modeling and summary statistics

**Out of scope (Phase 2):**
- Coach grading / ranking (statistics broken down by coach are included, but comparative grading is deferred)
- Clock-management timeouts (called with <5–6 min remaining)
- Penalty-avoidance simulation at later counterfactual plays (clock correction only for now)

---

## Data Sources

| Source | Purpose | Access |
|--------|---------|--------|
| `nflreadpy` | Play-by-play data, 1999–present | `pip install nflreadpy` |
| nflverse coach table | Head coach by team/season | via `nflreadpy` supplemental tables |
| nflfastR (R) | `calculate_win_probability()` | via `rpy2` |

**Data validation:** On first load, flag seasons/games with missing `wp`, `half_seconds_remaining`, or `posteam_timeouts_remaining` fields. Early seasons (pre-2006) may have higher rates of missing data.

---

## Architecture

```
nfl-timeout-analysis/
├── data/                         # cached parquet files (gitignored)
├── src/
│   ├── data/
│   │   └── loader.py             # nflreadpy fetch + local parquet cache
│   ├── wp/
│   │   └── calculator.py         # rpy2 wrapper around nflfastR
│   ├── timeouts/
│   │   ├── detector.py           # identify and classify early timeout events
│   │   ├── opportunities.py      # compute per-play counterfactual timeout value
│   │   └── scorer.py             # pair each timeout with its best opportunity
│   └── analysis/
│       └── visualize.py          # charts and summary tables
├── notebooks/
│   └── analysis.ipynb
├── reports/                      # saved chart outputs
├── requirements.txt
└── README.md
```

**Data flow:**
1. `loader.py` fetches seasons via `nflreadpy`, caches as parquet, joins coach table
2. `detector.py` scans for timeout events, filters to early-half penalty-avoidance candidates, excludes injury/4th-down/goal-line
3. `calculator.py` is called to compute WP for modified game states (actual and hypothetical)
4. `opportunities.py` iterates remaining plays in each half after each timeout, computing counterfactual WP gain per play
5. `scorer.py` aggregates: pairs each `EarlyTimeout` with its max `TimeoutOpportunity`, produces `TimeoutScore`
6. `visualize.py` generates the primary heatmap and summary tables

---

## Data Structures

### EarlyTimeout
One record per actual early timeout event.

| Field | Description |
|-------|-------------|
| `game_id`, `play_id` | Identifiers |
| `season`, `team`, `coach` | Context |
| `qtr`, `half_seconds_remaining` | When in game |
| `score_diff` | Score differential (posteam perspective) |
| `down`, `ydstogo`, `yardline_100` | Field position |
| `timeouts_before_use` (N) | Timeouts the team had before this call |
| `wp_gain_early` | WP(avoided-penalty state, N−1 TO) − WP(+5yd penalty state, N TO) |
| `excluded` | Bool — injury/4th-down/goal-line |

### TimeoutOpportunity
One record per play remaining in the same half after an `EarlyTimeout`.

| Field | Description |
|-------|-------------|
| `ref_timeout_play_id` | Links back to `EarlyTimeout` |
| `play_id`, `half_seconds_remaining` | When in the half |
| `clock_runoff` | `prev_half_seconds_remaining − current_half_seconds_remaining` |
| `actual_timeouts` (M) | Timeouts team actually had at this play |
| `wp_gain_counterfactual` | WP(M+1 TO, time + clock_runoff) − WP(M TO, actual time) |

**Clock correction:** The counterfactual game state sets `half_seconds_remaining = actual + clock_runoff`, capped at the previous play's `half_seconds_remaining`. This simulates stopping the clock at that moment instead of letting it run. No penalty-yardage correction is applied at counterfactual plays (phase 2).

### TimeoutScore
One record per `EarlyTimeout` — the final scored output.

| Field | Description |
|-------|-------------|
| All `EarlyTimeout` fields | — |
| `best_opportunity_value` | max(`wp_gain_counterfactual`) over remaining half |
| `best_opportunity_play_id` | Which play was the best opportunity |
| `best_opportunity_time` | `half_seconds_remaining` of that play |
| `opportunity_cost` | `best_opportunity_value − wp_gain_early` |
| `verdict` | `"justified"` if `opportunity_cost ≤ 0`, else `"wasteful"` |

---

## WP Calculation (rpy2 Wrapper)

`calculator.py` wraps nflfastR's `calculate_win_probability()`. The R session is initialized once at module load to avoid per-call startup overhead.

**Inputs to WP function (per nflfastR spec):**
- `score_differential`, `half_seconds_remaining`, `game_seconds_remaining`
- `down`, `ydstogo`, `yardline_100`
- `posteam_timeouts_remaining`, `defteam_timeouts_remaining`
- `receive_2h_ko`, `spread_line`

**Two calls per early timeout use:**
1. Avoided-penalty state: actual game state, `posteam_timeouts_remaining = N−1`
2. With-penalty state: `ydstogo += 5`, `yardline_100 += 5`, `posteam_timeouts_remaining = N`

**Two calls per TimeoutOpportunity play:**
1. Counterfactual with saved timeout: `half_seconds_remaining = actual + clock_runoff`, `posteam_timeouts_remaining = M+1`
2. Actual state: `half_seconds_remaining = actual`, `posteam_timeouts_remaining = M`

Calls are batched as dataframes rather than row-by-row to minimize R round-trips.

---

## Timeout Classification

**Include (early penalty-avoidance):**
- `timeout_team == posteam` (offensive timeout)
- `half_seconds_remaining > 300` (more than 5 minutes left in half)
- `play_type` context suggests pre-snap situation

**Exclude:**
- Injury timeouts (detectable from play description text)
- `down == 4` (4th down — strategic playcall timeout)
- `yardline_100 <= 5` (goal-line — strategic playcall timeout)

---

## Outputs

**Artifact:** `TimeoutScore` records saved as `data/processed/timeout_scores.parquet`

**Summary statistics broken down by:** team, season, coach, score differential bucket, time-in-half bucket

**Primary visualization — heatmap:**
Opportunity cost (z-axis) as a function of `score_differential` (x) × `half_seconds_remaining` at time of call (y). This directly answers: *under what conditions is it knowable at call time that the early timeout is a mistake?*

**Supporting charts:**
- Distribution of opportunity costs (histogram)
- % wasteful by coach (bar chart, min sample threshold applied)
- Opportunity cost vs. timeouts remaining at time of call
- Timeline: where in the half was the "best missed opportunity" for each wasteful timeout

---

## Dependencies

**Python:**
- `nflreadpy` — data fetching
- `rpy2` — R bridge
- `pandas`, `pyarrow` — data handling
- `matplotlib`, `seaborn` — visualization
- `jupyter` — notebook

**R (must be installed separately):**
- `nflfastR`
- `nflreadr`

---

## Open Questions / Risks

1. **nflfastR `calculate_win_probability()` input format** — exact column names and required fields need to be verified against the R package docs before `calculator.py` is written.
2. **Data completeness pre-2006** — early seasons may lack some fields; implement a per-season validation pass and flag/exclude incomplete games rather than silently dropping rows.
3. **5-minute threshold** — the 5–6 minute cutoff for "early" timeouts is a design parameter. Should be exposed as a configurable constant, not hardcoded, so it can be tuned during analysis.
4. **Batch size for rpy2 calls** — WP must be computed for every play remaining in the half after each early timeout. With ~25 years of data this could be millions of calls. Batching into dataframes and caching results will be important for performance.
