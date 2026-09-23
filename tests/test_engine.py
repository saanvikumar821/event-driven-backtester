import pandas as pd

from backtester.data import HistoricDataHandler
from backtester.engine import Engine
from backtester.events import MarketEvent, SignalEvent, SignalType
from backtester.execution import SimulatedExecutionHandler
from backtester.portfolio import Portfolio
from backtester.risk import RiskManager
from backtester.strategy import Strategy


class ScriptedStrategy(Strategy):
    """Emits a pre-set signal on a specific day, for deterministic testing."""

    def __init__(self, signals_by_day: dict[int, SignalEvent]) -> None:
        self.signals_by_day = signals_by_day
        self._day = -1

    def on_market_event(self, event: MarketEvent) -> list[SignalEvent]:
        self._day += 1
        signal = self.signals_by_day.get(self._day)
        return [signal] if signal else []


def make_frame(closes):
    n = len(closes)
    return pd.DataFrame(
        {
            "open": [c - 1 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 2 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, name="date"),
    )


def build_engine(strategy, closes=(100, 100, 100, 100, 100)):
    data_handler = HistoricDataHandler({"AAA": make_frame(list(closes))})
    return Engine(
        data_handler=data_handler,
        strategy=strategy,
        risk_manager=RiskManager(max_position_pct=0.10),
        execution_handler=SimulatedExecutionHandler(commission_per_trade=1.0, slippage_pct=0.0),
        portfolio=Portfolio(initial_cash=100_000),
    )


def test_no_signals_means_flat_equity_curve_at_cash():
    strategy = ScriptedStrategy({})
    engine = build_engine(strategy)
    portfolio = engine.run()
    assert all(equity == 100_000 for _, equity in portfolio.equity_curve)


def test_signal_is_not_filled_on_the_same_bar():
    ts = pd.Timestamp("2024-01-01")
    strategy = ScriptedStrategy({0: SignalEvent(timestamp=ts, symbol="AAA", signal_type=SignalType.LONG)})
    engine = build_engine(strategy)
    engine.run()
    # if it filled on day 0, day 0's equity would already reflect a fill;
    # cash before any fill must equal the initial cash exactly
    first_equity = engine.portfolio.equity_curve[0][1]
    assert first_equity == 100_000


def test_signal_is_filled_on_the_next_bars_open():
    ts = pd.Timestamp("2024-01-01")
    strategy = ScriptedStrategy({0: SignalEvent(timestamp=ts, symbol="AAA", signal_type=SignalType.LONG)})
    engine = build_engine(strategy)
    engine.run()
    # closes are all 100, opens are all 99 (close - 1); should hold a position by day 1
    assert engine.portfolio.quantity_of("AAA") > 0


def test_exit_signal_closes_the_position():
    ts0 = pd.Timestamp("2024-01-01")
    ts1 = pd.Timestamp("2024-01-02")
    strategy = ScriptedStrategy(
        {
            0: SignalEvent(timestamp=ts0, symbol="AAA", signal_type=SignalType.LONG),
            2: SignalEvent(timestamp=ts1, symbol="AAA", signal_type=SignalType.EXIT),
        }
    )
    engine = build_engine(strategy)
    engine.run()
    assert engine.portfolio.quantity_of("AAA") == 0


def test_engine_runs_end_to_end_without_error_over_full_series():
    strategy = ScriptedStrategy({})
    engine = build_engine(strategy, closes=(100, 105, 95, 110, 90, 120))
    portfolio = engine.run()
    assert len(portfolio.equity_curve) == 6