"""The event loop that ties every component together.

Each MarketEvent is processed in two stages:
  1. Fill any order queued from the *previous* bar, using *this* bar's open.
     This is what enforces the "you can only trade on the next bar's open"
     rule — a signal generated from day N's close cannot be filled until
     day N+1 exists.
  2. Feed this bar to the strategy, get any new signals, size them via the
     risk manager, and queue the resulting orders to be filled on the
     *following* bar.

The portfolio is marked to market once per bar, after any fill, so the
equity curve reflects the day's closing prices.
"""
from __future__ import annotations

from backtester.data import HistoricDataHandler
from backtester.events import MarketEvent, OrderEvent
from backtester.execution import SimulatedExecutionHandler
from backtester.portfolio import Portfolio
from backtester.risk import RiskManager
from backtester.strategy import Strategy


class Engine:
    def __init__(
        self,
        data_handler: HistoricDataHandler,
        strategy: Strategy,
        risk_manager: RiskManager,
        execution_handler: SimulatedExecutionHandler,
        portfolio: Portfolio,
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self._pending_orders: list[OrderEvent] = []

    def _fill_pending_orders(self, event: MarketEvent) -> None:
        """Fill orders queued from the previous bar at this bar's open."""
        for order in self._pending_orders:
            bar = event.bars.get(order.symbol)
            if bar is None:
                continue  # symbol didn't trade this bar; carry the order forward
            fill = self.execution_handler.execute_order(order, next_open_price=bar.open)
            self.portfolio.apply_fill(fill)
        self._pending_orders = [
            order for order in self._pending_orders if order.symbol not in event.bars
        ]

    def _generate_new_orders(self, event: MarketEvent) -> None:
        """Ask the strategy for signals on this bar, size them, queue the orders."""
        equity = self.portfolio.mark_to_market(
            event.timestamp, {symbol: bar.close for symbol, bar in event.bars.items()}
        )
        for signal in self.strategy.on_market_event(event):
            bar = event.bars[signal.symbol]
            order = self.risk_manager.evaluate_signal(
                signal,
                current_price=bar.close,
                equity=equity,
                current_quantity=self.portfolio.quantity_of(signal.symbol),
            )
            if order is not None:
                self._pending_orders.append(order)

    def run(self) -> Portfolio:
        for event in self.data_handler:
            self._fill_pending_orders(event)
            self._generate_new_orders(event)
        return self.portfolio