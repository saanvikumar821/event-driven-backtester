"""Historical data loading, validation, and chronological replay.

The DataHandler hands the rest of the system one bar at a time. Nothing
downstream ever receives the full DataFrame, so a strategy cannot peek at
future prices: lookahead bias is prevented structurally, not by discipline.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from backtester.events import Bar, MarketEvent

COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


def validate_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Raise if the data is malformed; return it unchanged otherwise."""
    if df.empty:
        raise ValueError(f"{symbol}: no data")
    if df.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate timestamps")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: timestamps are not sorted")
    if df[COLUMNS].isna().any().any():
        raise ValueError(f"{symbol}: contains missing values")
    if (df[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive prices")
    if (df["high"] < df["low"]).any():
        raise ValueError(f"{symbol}: high below low")
    return df


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily bars from Yahoo Finance (end date is exclusive).

    Prices are split- and dividend-adjusted, which is a simplification: fills
    use adjusted opens rather than the raw prices that actually traded.
    """
    import yfinance as yf  # imported here so tests do not need the network

    raw = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {symbol} ({start} to {end})")
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = COLUMNS
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df


def load_bars(
    symbols: Sequence[str],
    start: str,
    end: str,
    cache_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    """Load validated daily bars per symbol, caching downloads as CSV.

    Passing explicit start/end dates keeps runs reproducible: the same call
    always reads the same cached file.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = cache / f"{symbol}_{start}_{end}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="date", parse_dates=True)
        else:
            df = _download(symbol, start, end)
            df.to_csv(path)
        frames[symbol] = validate_bars(df, symbol)
    return frames


class HistoricDataHandler:
    """Replays aligned historical bars as a stream of MarketEvents.

    Symbols are aligned on the dates they all traded (an inner join), so every
    MarketEvent contains a bar for every symbol.
    """

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        if not data:
            raise ValueError("No data supplied")

        index: pd.Index | None = None
        for df in data.values():
            index = df.index if index is None else index.intersection(df.index)
        assert index is not None
        if len(index) == 0:
            raise ValueError("Symbols share no common dates")

        self.symbols: list[str] = list(data)
        self.index: pd.Index = index.sort_values()
        self._values = {
            symbol: df.loc[self.index, COLUMNS].to_numpy(dtype=float)
            for symbol, df in data.items()
        }

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Iterator[MarketEvent]:
        for i, timestamp in enumerate(self.index):
            bars = {symbol: Bar(*values[i]) for symbol, values in self._values.items()}
            yield MarketEvent(timestamp=timestamp, bars=bars)