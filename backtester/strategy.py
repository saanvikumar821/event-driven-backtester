"""Strategies turn market data into trading intent.

A Strategy only ever sees MarketEvents up to "now" — it has no access to the
portfolio, cash, or positions, and no way to see future bars. That separation
means a bug in a strategy cannot corrupt the books, and it cannot cheat by
using tomorrow's price.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from backtester.events import MarketEvent, SignalEvent, SignalType


class Strategy(ABC):
    """Base class every strategy implements."""

    @abstractmethod
    def on_market_event(self, event: MarketEvent) -> list[SignalEvent]:
        """Return zero or more signals in reaction to one day's bars."""
        raise NotImplementedError


class SmaCrossoverStrategy(Strategy):
    """Goes long a symbol while its fast SMA is above its slow SMA.

    Uses only closing prices seen so far — each symbol keeps its own rolling
    window, so a signal on day N depends only on days 1..N.
    """

    def __init__(self, symbols: list[str], fast_window: int = 10, slow_window: int = 50) -> None:
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self._closes: dict[str, deque[float]] = {
            symbol: deque(maxlen=slow_window) for symbol in symbols
        }
        # None = not yet holding an opinion; True = fast was above slow last time
        self._was_fast_above: dict[str, bool | None] = {symbol: None for symbol in symbols}

    def on_market_event(self, event: MarketEvent) -> list[SignalEvent]:
        signals: list[SignalEvent] = []
        for symbol, bar in event.bars.items():
            closes = self._closes[symbol]
            closes.append(bar.close)
            if len(closes) < self.slow_window:
                continue  # not enough history yet to compute both averages

            fast_avg = sum(list(closes)[-self.fast_window :]) / self.fast_window
            slow_avg = sum(closes) / len(closes)
            is_fast_above = fast_avg > slow_avg
            was_fast_above = self._was_fast_above[symbol]

            if was_fast_above is not None and is_fast_above != was_fast_above:
                signal_type = SignalType.LONG if is_fast_above else SignalType.EXIT
                signals.append(SignalEvent(timestamp=event.timestamp, symbol=symbol, signal_type=signal_type))

            self._was_fast_above[symbol] = is_fast_above

        return signals