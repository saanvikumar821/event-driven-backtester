import pandas as pd
import pytest

from backtester.events import Direction, SignalEvent, SignalType
from backtester.risk import RiskManager

TS = pd.Timestamp("2024-01-01")


def signal(signal_type: SignalType) -> SignalEvent:
    return SignalEvent(timestamp=TS, symbol="AAA", signal_type=signal_type)


def test_rejects_invalid_max_position_pct():
    with pytest.raises(ValueError):
        RiskManager(max_position_pct=0)
    with pytest.raises(ValueError):
        RiskManager(max_position_pct=1.5)


def test_long_signal_sized_to_max_position_pct():
    risk = RiskManager(max_position_pct=0.10)
    order = risk.evaluate_signal(signal(SignalType.LONG), current_price=50.0, equity=100_000, current_quantity=0)
    assert order is not None
    assert order.direction == Direction.BUY
    assert order.quantity == 200  # 10% of 100,000 = 10,000; /50 = 200 shares


def test_long_signal_ignored_if_already_holding():
    risk = RiskManager(max_position_pct=0.10)
    order = risk.evaluate_signal(signal(SignalType.LONG), current_price=50.0, equity=100_000, current_quantity=10)
    assert order is None


def test_long_signal_skipped_if_cannot_afford_one_share():
    risk = RiskManager(max_position_pct=0.10)
    order = risk.evaluate_signal(signal(SignalType.LONG), current_price=1_000_000, equity=100_000, current_quantity=0)
    assert order is None


def test_exit_signal_sells_full_current_quantity():
    risk = RiskManager(max_position_pct=0.10)
    order = risk.evaluate_signal(signal(SignalType.EXIT), current_price=50.0, equity=100_000, current_quantity=200)
    assert order is not None
    assert order.direction == Direction.SELL
    assert order.quantity == 200


def test_exit_signal_ignored_if_nothing_held():
    risk = RiskManager(max_position_pct=0.10)
    order = risk.evaluate_signal(signal(SignalType.EXIT), current_price=50.0, equity=100_000, current_quantity=0)
    assert order is None