"""Tracks cash, positions, and equity over time from fills.

The portfolio is a pure bookkeeper: it only reacts to FillEvents and current
prices. It never decides what to trade — that's the strategy and risk
manager's job — so it can be tested with hand-built fills and no strategy at
all.
"""
from __future__ import annotations

import pandas as pd

from backtester.events import Direction, FillEvent


class Portfolio:
    """Cash and position ledger, with an equity curve recorded per bar."""

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.cash: float = initial_cash
        self.positions: dict[str, int] = {}
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    def quantity_of(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)

    def apply_fill(self, fill: FillEvent) -> None:
        """Update cash and holdings from an executed fill."""
        cost = fill.price * fill.quantity + fill.commission
        current = self.positions.get(fill.symbol, 0)

        if fill.direction == Direction.BUY:
            self.cash -= cost
            self.positions[fill.symbol] = current + fill.quantity
        else:
            self.cash += fill.price * fill.quantity - fill.commission
            self.positions[fill.symbol] = current - fill.quantity

        if self.positions[fill.symbol] == 0:
            del self.positions[fill.symbol]

    def mark_to_market(self, timestamp: pd.Timestamp, prices: dict[str, float]) -> float:
        """Record and return total equity (cash + holdings) at current prices."""
        holdings_value = sum(
            quantity * prices[symbol] for symbol, quantity in self.positions.items()
        )
        equity = self.cash + holdings_value
        self.equity_curve.append((timestamp, equity))
        return equity