"""
Daily portfolio concentration / balance statistics.

Use with TradingEnv step info: allocation (per-symbol weights) and cash_weight.
"""

from __future__ import annotations

import math
from typing import Dict, List


def daily_balance_metrics(
    allocation: Dict[str, float],
    cash_weight: float,
) -> Dict[str, float]:
    """
    One row of stats for a single day.

    - hhi: Herfindahl–Hirschman on full book (all stock weights + cash). Range [1/n, 1].
      Lower ⇒ more spread out (more "balanced" in a diversification sense).
    - effective_n: 1 / hhi. Interpret as "how many equal-sized sleeves" the book resembles.
    - entropy: Shannon entropy of weights (nats). Higher ⇒ less concentrated.
    - max_weight / max_stock_weight: largest slice (any leg / stocks only).
    - n_stocks_held: count of stocks with positive weight.
    """
    stocks: List[float] = [max(0.0, float(v)) for v in allocation.values()]
    cash = max(0.0, float(cash_weight))
    full = stocks + [cash]
    total = sum(full)
    if total < 1e-12:
        return {
            "hhi": 1.0,
            "effective_n": 1.0,
            "entropy": 0.0,
            "max_weight": 0.0,
            "max_stock_weight": 0.0,
            "n_stocks_held": 0.0,
            "cash_weight": 0.0,
        }

    w = [x / total for x in full]
    hhi = float(sum(x * x for x in w))
    eff_n = float(1.0 / hhi) if hhi > 1e-12 else 1.0
    entropy = float(-sum(x * math.log(x + 1e-15) for x in w if x > 1e-15))
    max_w = float(max(w))
    sw = [x / total for x in stocks]
    max_stock = float(max(sw)) if sw else 0.0
    n_held = float(sum(1 for s in sw if s > 1e-6))

    return {
        "hhi": hhi,
        "effective_n": eff_n,
        "entropy": entropy,
        "max_weight": max_w,
        "max_stock_weight": max_stock,
        "n_stocks_held": n_held,
        "cash_weight": float(cash / total),
    }
