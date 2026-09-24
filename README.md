# event-driven-backtester
Event-driven backtesting engine in Python with modelled fees and slippage, risk limits, and performance metrics.

## Why event-driven

Rather than vectorising signals over an entire price history at once, this
engine processes one `MarketEvent` at a time, the same way a live trading
system would. A strategy only ever sees data up to the current bar, and every
order is filled on the *next* bar's open, never the bar that generated the
signal. Both of these are structural defences against lookahead bias, not
just conventions the strategy code happens to follow.


## Architecture

| Module | Responsibility |
|---|---|
| `backtester/events.py` | Immutable event types passed between components |
| `backtester/data.py` | Downloads, validates, caches, and replays historical bars |
| `backtester/strategy.py` | Strategy interface + an SMA crossover implementation |
| `backtester/risk.py` | Pre-trade position sizing and limits |
| `backtester/execution.py` | Simulated fills with slippage and commission |
| `backtester/portfolio.py` | Cash, positions, and equity curve tracking |
| `backtester/metrics.py` | Sharpe ratio, max drawdown, total return |
| `backtester/engine.py` | The event loop wiring everything together |

Each component only depends on the event types, not on each other's internals,
so any piece (e.g. the strategy) can be swapped or unit tested independently.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python main.py
```

This runs an SMA(10/50) crossover on AAPL and MSFT (2018-2024) against a
buy-and-hold benchmark, printing performance metrics and saving
`equity_curve.png`.

## Results (AAPL, MSFT, 2018-2024)

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total return | 25.41% | 370.76% |
| Sharpe ratio | 0.95 | 1.04 |
| Max drawdown | -6.73% | -32.31% |

The strategy returns far less than buy-and-hold over this period, which saw
an exceptional bull run in both stocks. This isn't a bug: the risk layer caps
any position at 10% of equity and moves to cash when the trend reverses, so
it structurally cannot match a fully-invested benchmark's upside. In exchange
it has a much smaller drawdown (-6.7% vs -32.3%), a genuine risk/return
trade-off rather than a better or worse result overall.

## Testing

```bash
pytest -q
```

45 unit tests covering event immutability, data validation, strategy signal
logic, risk sizing, execution fills, portfolio bookkeeping, metrics, and the
engine's fill-on-next-bar behaviour.

## Limitations and possible extensions

- Uses split/dividend-adjusted daily close and open prices, not the raw
  prices that actually traded.
- The 10% position cap limits any single new position but doesn't cap total
  portfolio exposure across multiple concurrent positions.
- Long-only, with a single strategy implemented (SMA crossover). The
  `Strategy` interface is designed so additional strategies can be added
  without changing the engine.
- No live or paper-trading execution handler yet; `SimulatedExecutionHandler`
  could be replaced with a broker adapter (e.g. Alpaca) behind the same
  interface.

## License

MIT
'@ | Set-Content -Encoding utf8 README.md