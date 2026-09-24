"""Entry point: runs the SMA crossover strategy through the backtester
and reports results against a buy-and-hold benchmark.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from backtester.data import HistoricDataHandler, load_bars
from backtester.engine import Engine
from backtester.execution import SimulatedExecutionHandler
from backtester.metrics import equity_curve_to_series, summarize
from backtester.portfolio import Portfolio
from backtester.risk import RiskManager
from backtester.strategy import SmaCrossoverStrategy

SYMBOLS = ["AAPL", "MSFT"]
START = "2018-01-01"
END = "2024-01-01"
INITIAL_CASH = 100_000.0


def buy_and_hold_benchmark(data: dict[str, pd.DataFrame], initial_cash: float) -> pd.Series:
    """Equal-weight, buy-on-day-one benchmark using the same aligned dates."""
    handler = HistoricDataHandler(data)
    per_symbol_cash = initial_cash / len(handler.symbols)
    shares: dict[str, float] = {}
    values = []
    for i, event in enumerate(handler):
        if i == 0:
            shares = {symbol: per_symbol_cash / bar.open for symbol, bar in event.bars.items()}
        total = sum(shares[symbol] * bar.close for symbol, bar in event.bars.items())
        values.append((event.timestamp, total))
    return equity_curve_to_series(values)


def run_backtest() -> tuple[Portfolio, pd.Series]:
    data = load_bars(SYMBOLS, START, END)
    data_handler = HistoricDataHandler(data)

    engine = Engine(
        data_handler=data_handler,
        strategy=SmaCrossoverStrategy(SYMBOLS, fast_window=10, slow_window=50),
        risk_manager=RiskManager(max_position_pct=0.10),
        execution_handler=SimulatedExecutionHandler(commission_per_trade=1.0, slippage_pct=0.0005),
        portfolio=Portfolio(initial_cash=INITIAL_CASH),
    )
    portfolio = engine.run()
    benchmark = buy_and_hold_benchmark(data, INITIAL_CASH)
    return portfolio, benchmark


def report(portfolio: Portfolio, benchmark: pd.Series) -> None:
    strategy_equity = equity_curve_to_series(portfolio.equity_curve)
    strategy_metrics = summarize(portfolio.equity_curve)
    benchmark_metrics = summarize(list(zip(benchmark.index, benchmark.values)))

    print(f"{'Metric':<15}{'Strategy':>12}{'Buy & Hold':>14}")
    for key in ("total_return", "sharpe_ratio", "max_drawdown"):
        if key == "sharpe_ratio":
            print(f"{key:<15}{strategy_metrics[key]:>12.2f}{benchmark_metrics[key]:>14.2f}")
        else:
            print(f"{key:<15}{strategy_metrics[key]:>12.2%}{benchmark_metrics[key]:>14.2%}")

    fig, ax = plt.subplots(figsize=(10, 5))
    strategy_equity.plot(ax=ax, label="SMA crossover")
    benchmark.plot(ax=ax, label="Buy & hold")
    ax.set_title(f"Equity curve: {', '.join(SYMBOLS)} ({START} to {END})")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("equity_curve.png")
    print("\nSaved chart to equity_curve.png")


if __name__ == "__main__":
    portfolio, benchmark = run_backtest()
    report(portfolio, benchmark)