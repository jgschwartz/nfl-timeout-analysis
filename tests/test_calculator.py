from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from src.wp.calculator import WP_INPUT_COLS


@pytest.fixture
def sample_states():
    return pd.DataFrame({
        "home_team": ["DET", "DET"],
        "posteam": ["KC", "KC"],
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


def _mock_ro(wp_values):
    mock_r_result = pd.DataFrame({"wp": wp_values})
    mock = MagicMock()
    mock.conversion.rpy2py.return_value = mock_r_result
    return mock


def test_calculate_wp_returns_series_same_length(sample_states):
    mock_ro = _mock_ro([0.52, 0.65])
    mock_nflfastr = MagicMock()
    with patch("src.wp.calculator._ro", mock_ro), \
         patch("src.wp.calculator.pandas2ri"), \
         patch("src.wp.calculator.localconverter"), \
         patch("src.wp.calculator._get_nflfastr", return_value=mock_nflfastr):
        from src.wp import calculator
        result = calculator.calculate_wp(sample_states)
    assert len(result) == len(sample_states)


def test_calculate_wp_index_matches_input(sample_states):
    sample_states = sample_states.copy()
    sample_states.index = [10, 20]
    mock_ro = _mock_ro([0.52, 0.65])
    mock_nflfastr = MagicMock()
    with patch("src.wp.calculator._ro", mock_ro), \
         patch("src.wp.calculator.pandas2ri"), \
         patch("src.wp.calculator.localconverter"), \
         patch("src.wp.calculator._get_nflfastr", return_value=mock_nflfastr):
        from src.wp import calculator
        result = calculator.calculate_wp(sample_states)
    assert list(result.index) == [10, 20]


def test_calculate_wp_calls_nflfastr(sample_states):
    mock_ro = _mock_ro([0.52, 0.65])
    mock_nflfastr = MagicMock()
    with patch("src.wp.calculator._ro", mock_ro), \
         patch("src.wp.calculator.pandas2ri"), \
         patch("src.wp.calculator.localconverter"), \
         patch("src.wp.calculator._get_nflfastr", return_value=mock_nflfastr):
        from src.wp import calculator
        calculator.calculate_wp(sample_states)
    mock_nflfastr.calculate_win_probability.assert_called_once()
