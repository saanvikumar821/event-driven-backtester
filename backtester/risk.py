"""Pre-trade risk gate: turns strategy intent into sized, limit-checked orders.

The strategy never sees cash or positions, and the risk layer never sees
prices beyond what it's given for the current bar. This is the one place
position sizing and limits are enforced.
"""
from __future__ import annotations

from backtester.events import Direction, OrderEvent, SignalEvent, SignalType


class RiskManager:
    """Sizes orders using a fixed fraction of current equity, with a cap.

    - LONG signals are ignored if already holding the symbol, or if the order
      would exceed max_position_pct of current equity.
    - EXIT signals are ignored if not currently holding the symbol.
    - Orders are only ever whole shares; if the allocation can't buy at least
      one share, the signal is skipped rather than rejected as an error.
    """

    def __init__(self, max_position_pct: float = 0.10) -> None:
        if not 0 < max_position_pct <= 1:
            raise ValueError("max_position_pct must be between 0 and 1")
        self.max_position_pct = max_position_pct

    def evaluate_signal(
        self,
        signal: SignalEvent,
        current_price: float,
        equity: float,
        current_quantity: int,
    ) -> OrderEvent | None:
        """Return an OrderEvent to send for execution, or None to reject/skip."""
        if signal.signal_type == SignalType.LONG:
            if current_quantity > 0:
                return None  # already holding; do not pyramid the position
            max_allocation = equity * self.max_position_pct
            quantity = int(max_allocation // current_price)
            if quantity < 1:
                return None  # too little equity to buy even one share
            return OrderEvent(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                direction=Direction.BUY,
                quantity=quantity,
            )

        if signal.signal_type == SignalType.EXIT:
            if current_quantity <= 0:
                return None  # nothing to sell
            return OrderEvent(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                direction=Direction.SELL,
                quantity=current_quantity,
            )

        raise ValueError(f"Unhandled signal type: {signal.signal_type}")