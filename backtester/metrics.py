"""Performance metrics computed from a portfolio's equity curve.

Kept separate from the portfolio itself: the portfolio only tracks state,
metrics only interpret it. That split makes it easy to test each in
isolation and to add new metrics without touching the portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def equity_curve_to_series(equity_curve: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    """Convert the portfolio's list of (timestamp, equity) pairs to a Series."""
    if not equity_curve:
        raise ValueError("equity_curve is empty")
    timestamps, values = zip(*equity_curve)
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="equity")


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple day-over-day percentage returns, dropping the first NaN."""
    return equity.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio from daily returns.

    risk_free_rate is annual; it's converted to a daily rate before
    subtracting. Returns 0.0 if volatility is zero, to avoid dividing by zero
    on a flat or single-point series.
    """
    if len(returns) == 0:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    std = excess.std()
    if std == 0:
        return 0.0
    return (excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.23)."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def total_return(equity: pd.Series) -> float:
    """Overall return from the first to the last equity value."""
    return (equity.iloc[-1] / equity.iloc[0]) - 1


def summarize(equity_curve: list[tuple[pd.Timestamp, float]]) -> dict[str, float]:
    """Convenience wrapper returning all headline metrics at once."""
    equity = equity_curve_to_series(equity_curve)
    returns = daily_returns(equity)
    return {
        "total_return": total_return(equity),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
    }