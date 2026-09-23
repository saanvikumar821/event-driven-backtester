"""Simulated execution: turns orders into fills with realistic costs.

Orders are filled at the *next* bar's open, not the price that triggered the
signal. A signal is generated from a bar's close, which you could not have
traded on at the time — you only know the close once the day is over. Filling
at the next open is what prevents this lookahead bias from leaking into
prices as well as into signals.
"""
from __future__ import annotations

from backtester.events import Direction, FillEvent, OrderEvent


class SimulatedExecutionHandler:
    """Fills orders against a supplied next-bar open price.

    Slippage is modelled as a fixed percentage that always works against you:
    buys fill slightly above the open, sells slightly below it.
    """

    def __init__(self, commission_per_trade: float = 1.0, slippage_pct: float = 0.0005) -> None:
        if commission_per_trade < 0:
            raise ValueError("commission_per_trade cannot be negative")
        if slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative")
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct

    def execute_order(self, order: OrderEvent, next_open_price: float) -> FillEvent:
        if order.direction == Direction.BUY:
            fill_price = next_open_price * (1 + self.slippage_pct)
        else:
            fill_price = next_open_price * (1 - self.slippage_pct)

        return FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            price=fill_price,
            commission=self.commission_per_trade,
        )