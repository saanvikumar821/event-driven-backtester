import pandas as pd
import pytest

from backtester.events import Direction, OrderEvent
from backtester.execution import SimulatedExecutionHandler

TS = pd.Timestamp("2024-01-01")


def order(direction: Direction, quantity: int = 100) -> OrderEvent:
    return OrderEvent(timestamp=TS, symbol="AAA", direction=direction, quantity=quantity)


def test_rejects_negative_commission():
    with pytest.raises(ValueError):
        SimulatedExecutionHandler(commission_per_trade=-1.0)


def test_rejects_negative_slippage():
    with pytest.raises(ValueError):
        SimulatedExecutionHandler(slippage_pct=-0.01)


def test_buy_fills_above_open_price():
    handler = SimulatedExecutionHandler(slippage_pct=0.01)
    fill = handler.execute_order(order(Direction.BUY), next_open_price=100.0)
    assert fill.price == pytest.approx(101.0)  # 1% worse than the open


def test_sell_fills_below_open_price():
    handler = SimulatedExecutionHandler(slippage_pct=0.01)
    fill = handler.execute_order(order(Direction.SELL), next_open_price=100.0)
    assert fill.price == pytest.approx(99.0)  # 1% worse than the open


def test_zero_slippage_fills_exactly_at_open():
    handler = SimulatedExecutionHandler(slippage_pct=0.0)
    fill = handler.execute_order(order(Direction.BUY), next_open_price=100.0)
    assert fill.price == 100.0


def test_commission_is_passed_through_unchanged():
    handler = SimulatedExecutionHandler(commission_per_trade=2.5)
    fill = handler.execute_order(order(Direction.BUY), next_open_price=100.0)
    assert fill.commission == 2.5


def test_quantity_and_symbol_preserved_from_order():
    handler = SimulatedExecutionHandler()
    fill = handler.execute_order(order(Direction.SELL, quantity=42), next_open_price=100.0)
    assert fill.quantity == 42
    assert fill.symbol == "AAA"