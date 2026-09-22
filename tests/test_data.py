from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from backtester.data import HistoricDataHandler, validate_bars
from backtester.events import Bar


def make_frame(dates, start_price=100.0):
    n = len(dates)
    close = [start_price + i for i in range(n)]
    return pd.DataFrame(
        {
            "open": [c - 0.5 for c in close],
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [1000.0] * n,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def test_events_are_chronological():
    frame = make_frame(pd.date_range("2024-01-01", periods=5))
    timestamps = [event.timestamp for event in HistoricDataHandler({"AAA": frame})]
    assert len(timestamps) == 5
    assert timestamps == sorted(timestamps)


def test_symbols_are_aligned_on_common_dates():
    a = make_frame(pd.date_range("2024-01-01", periods=5))  # Jan 1-5
    b = make_frame(pd.date_range("2024-01-03", periods=5))  # Jan 3-7
    events = list(HistoricDataHandler({"AAA": a, "BBB": b}))
    assert len(events) == 3  # Jan 3, 4, 5
    assert set(events[0].bars) == {"AAA", "BBB"}


def test_bar_values_match_source_row():
    frame = make_frame(pd.date_range("2024-01-01", periods=3))
    event = next(iter(HistoricDataHandler({"AAA": frame})))
    assert event.bars["AAA"] == Bar(99.5, 101.0, 99.0, 100.0, 1000.0)


def test_events_are_immutable():
    frame = make_frame(pd.date_range("2024-01-01", periods=3))
    event = next(iter(HistoricDataHandler({"AAA": frame})))
    with pytest.raises(FrozenInstanceError):
        event.timestamp = pd.Timestamp("2030-01-01")


def test_validation_rejects_non_positive_prices():
    frame = make_frame(pd.date_range("2024-01-01", periods=3))
    frame.loc[frame.index[1], "close"] = -1.0
    with pytest.raises(ValueError):
        validate_bars(frame, "AAA")


def test_validation_rejects_duplicate_timestamps():
    frame = make_frame(pd.date_range("2024-01-01", periods=3))
    duplicated = pd.concat([frame, frame.iloc[[0]]]).sort_index()
    with pytest.raises(ValueError):
        validate_bars(duplicated, "AAA")


def test_handler_rejects_symbols_with_no_overlap():
    a = make_frame(pd.date_range("2024-01-01", periods=3))
    b = make_frame(pd.date_range("2024-02-01", periods=3))
    with pytest.raises(ValueError):
        HistoricDataHandler({"AAA": a, "BBB": b})