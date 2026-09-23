import numpy as np
import pandas as pd
import pytest

from backtester.metrics import (
    daily_returns,
    equity_curve_to_series,
    max_drawdown,
    sharpe_ratio,
    summarize,
    total_return,
)


def make_curve(values):
    dates = pd.date_range("2024-01-01", periods=len(values))
    return list(zip(dates, values))


def test_equity_curve_to_series_rejects_empty():
    with pytest.raises(ValueError):
        equity_curve_to_series([])


def test_daily_returns_length_is_one_less_than_input():
    equity = equity_curve_to_series(make_curve([100, 110, 105]))
    returns = daily_returns(equity)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)


def test_total_return_matches_first_to_last():
    equity = equity_curve_to_series(make_curve([100, 150]))
    assert total_return(equity) == pytest.approx(0.50)


def test_max_drawdown_on_known_path():
    # rises to 120, falls to 90 (25% off the 120 peak), recovers to 100
    equity = equity_curve_to_series(make_curve([100, 120, 90, 100]))
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_is_zero_for_monotonically_rising_equity():
    equity = equity_curve_to_series(make_curve([100, 110, 120, 130]))
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_sharpe_ratio_is_zero_for_constant_returns():
    equity = equity_curve_to_series(make_curve([100, 100, 100, 100]))
    returns = daily_returns(equity)
    assert sharpe_ratio(returns) == 0.0  # std is zero, must not divide by zero


def test_sharpe_ratio_is_positive_for_steady_gains():
    equity = equity_curve_to_series(make_curve([100, 101, 102, 103, 104]))
    returns = daily_returns(equity)
    assert sharpe_ratio(returns) > 0


def test_sharpe_ratio_handles_empty_returns():
    assert sharpe_ratio(pd.Series(dtype=float)) == 0.0


def test_summarize_returns_all_three_keys():
    result = summarize(make_curve([100, 110, 90, 105]))
    assert set(result) == {"total_return", "sharpe_ratio", "max_drawdown"}
    assert result["total_return"] == pytest.approx(0.05)