"""Immutable event types that flow through the backtester.

Components communicate only by emitting and consuming events; none of them
reach into another component's state. Events are frozen dataclasses, so once
an event leaves its source it cannot be modified downstream.

Flow of a single trading day:

    MarketEvent -> Strategy -> SignalEvent -> Risk -> OrderEvent
                -> Execution -> FillEvent -> Portfolio
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SignalType(Enum):
    """What the strategy wants. Strategies express intent, never quantity."""

    LONG = "LONG"  # want to hold the asset
    EXIT = "EXIT"  # want to be flat in the asset


class Direction(Enum):
    """Side of an order or fill."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar for a single symbol."""

    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A new bar has arrived for every symbol at this timestamp."""

    timestamp: pd.Timestamp
    bars: Mapping[str, Bar]


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """Strategy output: which symbol it wants to hold or exit."""

    timestamp: pd.Timestamp
    symbol: str
    signal_type: SignalType


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """A sized order that has passed the pre-trade risk checks."""

    timestamp: pd.Timestamp
    symbol: str
    direction: Direction
    quantity: int


@dataclass(frozen=True, slots=True)
class FillEvent:
    """Confirmation that an order was executed, including costs."""

    timestamp: pd.Timestamp
    symbol: str
    direction: Direction
    quantity: int
    price: float  # execution price after slippage
    commission: float