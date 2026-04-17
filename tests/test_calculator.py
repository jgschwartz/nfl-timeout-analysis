from unittest.mock import patch, MagicMock, PropertyMock
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


def _make_mock_nflfastr(wp_values):
    """Build a mock nflfastr that returns a DataFrame with a 'wp' column."""
    mock_r_result = pd.DataFrame({"wp": wp_values})
    mock_nflfastr = MagicMock()
    return mock_nflfastr, mock_r_result


def test_calculate_wp_returns_series_same_length(sample_states):
    mock_nflfastr, mock_r_result = _make_mock_nflfastr([0.52, 0.65])
    with patch("src.wp.calculator.importr"), \
         patch("src.wp.calculator.pandas2ri") as mock_p2r, \
         patch("src.wp.calculator._get_nflfastr", return_value=mock_nflfastr):
        mock_p2r.py2rpy.return_value = MagicMock()
        mock_p2r.rpy2py.return_value = mock_r_result
        from src.wp import calculator
        result = calculator.calculate_wp(sample_states)
    assert len(result) == len(sample_states)


def test_calculate_wp_index_matches_input(sample_states):
    sample_states = sample_states.copy()
    sample_states.index = [10, 20]
    mock_nflfastr, mock_r_result = _make_mock_nflfastr([0.52, 0.65])
    with patch("src.wp.calculator.importr"), \
         patch("src.wp.calculator.pandas2ri") as mock_p2r, \
         patch("src.wp.calculator._get_nflfastr", return_value=mock_nflfastr):
        mock_p2r.py2rpy.return_value = MagicMock()
        mock_p2r.rpy2py.return_value = mock_r_result
        from src.wp import calculator
        result = calculator.calculate_wp(sample_states)
    assert list(result.index) == [10, 20]


def test_calculate_wp_calls_nflfastr(sample_states):
    mock_nflfastr, mock_r_result = _make_mock_nflfastr([0.52, 0.65])
    with patch("src.wp.calculator.importr"), \
         patch("src.wp.calculator.pandas2ri") as mock_p2r, \
         patch("src.wp.calculator._get_nflfastr", return_value=mock_nflfastr):
        mock_p2r.py2rpy.return_value = MagicMock()
        mock_p2r.rpy2py.return_value = mock_r_result
        from src.wp import calculator
        calculator.calculate_wp(sample_states)
    mock_nflfastr.calculate_win_probability.assert_called_once()
