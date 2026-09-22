import pandas as pd
import pytest

from backtester.events import Bar, MarketEvent, SignalType
from backtester.strategy import SmaCrossoverStrategy


def market_event(day: int, close: float) -> MarketEvent:
    bar = Bar(open=close, high=close, low=close, close=close, volume=1000.0)
    return MarketEvent(timestamp=pd.Timestamp("2024-01-01") + pd.Timedelta(days=day), bars={"AAA": bar})


def test_rejects_fast_window_not_smaller_than_slow():
    with pytest.raises(ValueError):
        SmaCrossoverStrategy(["AAA"], fast_window=10, slow_window=10)


def test_no_signal_before_enough_history():
    strategy = SmaCrossoverStrategy(["AAA"], fast_window=2, slow_window=4)
    signals = []
    for day, close in enumerate([100, 100, 100]):
        signals += strategy.on_market_event(market_event(day, close))
    assert signals == []  # only 3 bars seen, slow_window needs 4


def test_emits_long_when_fast_crosses_above_slow():
    strategy = SmaCrossoverStrategy(["AAA"], fast_window=2, slow_window=4)
    closes = [100, 100, 100, 100, 110, 120]  # flat, then a sharp rise
    signals = []
    for day, close in enumerate(closes):
        signals += strategy.on_market_event(market_event(day, close))
    assert len(signals) == 1
    assert signals[0].signal_type == SignalType.LONG
    assert signals[0].symbol == "AAA"


def test_emits_exit_after_long_when_fast_drops_back_below_slow():
    strategy = SmaCrossoverStrategy(["AAA"], fast_window=2, slow_window=4)
    closes = [100, 100, 100, 100, 110, 120, 90, 80]  # rise, then a sharp fall
    signals = []
    for day, close in enumerate(closes):
        signals += strategy.on_market_event(market_event(day, close))
    types = [s.signal_type for s in signals]
    assert types == [SignalType.LONG, SignalType.EXIT]


def test_no_duplicate_signal_while_state_unchanged():
    strategy = SmaCrossoverStrategy(["AAA"], fast_window=2, slow_window=4)
    closes = [100, 100, 100, 100, 110, 120, 130, 140]  # keeps rising after the cross
    signals = []
    for day, close in enumerate(closes):
        signals += strategy.on_market_event(market_event(day, close))
    assert len(signals) == 1  # only the initial LONG, no repeats while still above