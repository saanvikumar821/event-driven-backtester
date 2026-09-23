import pandas as pd
import pytest

from backtester.events import Direction, FillEvent
from backtester.portfolio import Portfolio

TS = pd.Timestamp("2024-01-01")


def fill(direction: Direction, quantity: int, price: float, commission: float = 1.0) -> FillEvent:
    return FillEvent(timestamp=TS, symbol="AAA", direction=direction, quantity=quantity, price=price, commission=commission)


def test_rejects_non_positive_initial_cash():
    with pytest.raises(ValueError):
        Portfolio(initial_cash=0)


def test_buy_reduces_cash_and_adds_position():
    portfolio = Portfolio(initial_cash=10_000)
    portfolio.apply_fill(fill(Direction.BUY, quantity=10, price=50.0, commission=1.0))
    assert portfolio.cash == pytest.approx(10_000 - 500 - 1.0)
    assert portfolio.quantity_of("AAA") == 10


def test_sell_increases_cash_and_removes_position():
    portfolio = Portfolio(initial_cash=10_000)
    portfolio.apply_fill(fill(Direction.BUY, quantity=10, price=50.0, commission=1.0))
    portfolio.apply_fill(fill(Direction.SELL, quantity=10, price=55.0, commission=1.0))
    assert portfolio.quantity_of("AAA") == 0
    assert "AAA" not in portfolio.positions  # fully closed positions are removed
    assert portfolio.cash == pytest.approx(10_000 - 501 + 550 - 1.0)


def test_mark_to_market_reflects_cash_and_holdings():
    portfolio = Portfolio(initial_cash=10_000)
    portfolio.apply_fill(fill(Direction.BUY, quantity=10, price=50.0, commission=1.0))
    equity = portfolio.mark_to_market(TS, prices={"AAA": 60.0})
    # cash after buy: 10,000 - 500 - 1 = 9,499; holdings: 10 * 60 = 600
    assert equity == pytest.approx(9_499 + 600)


def test_equity_curve_records_one_point_per_call():
    portfolio = Portfolio(initial_cash=10_000)
    portfolio.mark_to_market(TS, prices={})
    portfolio.mark_to_market(TS + pd.Timedelta(days=1), prices={})
    assert len(portfolio.equity_curve) == 2


def test_commission_reduces_equity_on_both_sides():
    portfolio = Portfolio(initial_cash=10_000)
    portfolio.apply_fill(fill(Direction.BUY, quantity=10, price=50.0, commission=5.0))
    portfolio.apply_fill(fill(Direction.SELL, quantity=10, price=50.0, commission=5.0))
    # bought and sold at the same price, so the only loss is 10 in commission
    assert portfolio.cash == pytest.approx(10_000 - 10)