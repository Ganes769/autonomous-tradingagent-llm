"""
Evaluation Metrics for Trading Agent

Implements:
- Sharpe Ratio        — risk-adjusted return vs total volatility
- Sortino Ratio       — risk-adjusted return vs downside volatility
- Calmar Ratio        — annualised return / max drawdown
- Maximum Drawdown    — worst peak-to-trough decline
- Total Return
- Annualised Volatility
- Downside Deviation
- Win Rate
- Profit Factor
- Average Trade Duration (estimated from action labels)
- Convergence metrics (reward mean / std over training windows)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


class TradingMetrics:
    """Calculate comprehensive trading performance metrics."""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Composite interface
    # ------------------------------------------------------------------

    def calculate_all_metrics(
        self,
        portfolio_values: List[float],
        dates: Optional[List] = None,
        action_labels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        if len(portfolio_values) < 2:
            return self._zero_metrics()

        returns = self._returns(portfolio_values)
        ann_factor = self._annualisation_factor(returns, dates)

        metrics = {
            "total_return":       self.total_return(portfolio_values),
            "annualised_return":  self.annualised_return(portfolio_values, dates),
            "sharpe_ratio":       self.sharpe_ratio(returns, ann_factor),
            "sortino_ratio":      self.sortino_ratio(returns, ann_factor),
            "calmar_ratio":       self.calmar_ratio(portfolio_values, dates),
            "max_drawdown":       self.max_drawdown(portfolio_values),
            "volatility":         self.volatility(returns, ann_factor),
            "downside_deviation": self.downside_deviation(returns, ann_factor),
            "win_rate":           self.win_rate(returns),
            "profit_factor":      self.profit_factor(returns),
        }

        if action_labels:
            metrics.update(self.action_distribution(action_labels))

        return metrics

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self, returns: np.ndarray, ann_factor: float) -> float:
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        daily_rf = self.risk_free_rate / _TRADING_DAYS
        excess = returns - daily_rf
        return float((np.mean(excess) * ann_factor) / (np.std(returns) * np.sqrt(ann_factor) + 1e-8))

    def sortino_ratio(self, returns: np.ndarray, ann_factor: float) -> float:
        if len(returns) == 0:
            return 0.0
        daily_rf = self.risk_free_rate / _TRADING_DAYS
        excess = returns - daily_rf
        neg = returns[returns < daily_rf]
        if len(neg) == 0:
            return float(np.mean(excess) * ann_factor / 1e-8)
        dd = np.sqrt(np.mean(neg ** 2)) * np.sqrt(ann_factor)
        return float(np.mean(excess) * ann_factor / (dd + 1e-8))

    def calmar_ratio(
        self,
        portfolio_values: List[float],
        dates: Optional[List] = None,
    ) -> float:
        ann_ret = self.annualised_return(portfolio_values, dates)
        mdd = self.max_drawdown(portfolio_values)
        if mdd == 0:
            return 0.0
        return float(ann_ret / mdd)

    def max_drawdown(self, portfolio_values: List[float]) -> float:
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        dd = (peak - values) / (peak + 1e-8)
        return float(np.max(dd))

    def max_drawdown_duration(self, portfolio_values: List[float]) -> int:
        """Number of steps inside the longest drawdown period."""
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        in_dd = values < peak
        max_dur = 0
        cur_dur = 0
        for flag in in_dd:
            cur_dur = cur_dur + 1 if flag else 0
            max_dur = max(max_dur, cur_dur)
        return int(max_dur)

    def total_return(self, portfolio_values: List[float]) -> float:
        if len(portfolio_values) < 2:
            return 0.0
        return float((portfolio_values[-1] - portfolio_values[0]) / (portfolio_values[0] + 1e-8))

    def annualised_return(
        self,
        portfolio_values: List[float],
        dates: Optional[List] = None,
    ) -> float:
        total = self.total_return(portfolio_values)
        n = len(portfolio_values) - 1
        if n <= 0:
            return 0.0
        if dates and len(dates) > 1:
            years = (dates[-1] - dates[0]).days / 365.25
        else:
            years = n / _TRADING_DAYS
        if years <= 0:
            return 0.0
        return float((1 + total) ** (1 / years) - 1)

    def volatility(self, returns: np.ndarray, ann_factor: float) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.std(returns) * np.sqrt(ann_factor))

    def downside_deviation(self, returns: np.ndarray, ann_factor: float) -> float:
        if len(returns) == 0:
            return 0.0
        neg = returns[returns < 0]
        if len(neg) == 0:
            return 0.0
        return float(np.std(neg) * np.sqrt(ann_factor))

    def win_rate(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.mean(returns > 0))

    def profit_factor(self, returns: np.ndarray) -> float:
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float(gains) if gains > 0 else 0.0
        return float(gains / losses)

    def action_distribution(self, action_labels: List[str]) -> Dict[str, float]:
        total = len(action_labels)
        if total == 0:
            return {}
        from collections import Counter
        counts = Counter(action_labels)
        return {
            f"pct_{k.lower()}": v / total
            for k, v in counts.items()
        }

    # ------------------------------------------------------------------
    # Convergence / training analysis
    # ------------------------------------------------------------------

    @staticmethod
    def compute_convergence_stats(
        episode_rewards: List[float],
        window: int = 20,
    ) -> Dict[str, float]:
        """
        Compute rolling mean and std of episode rewards to assess convergence.

        Returns
        -------
        Dict with:
          rolling_mean_last   – mean over last `window` episodes
          rolling_std_last    – std over last `window` episodes
          improvement         – (mean_last_window - mean_first_window) / |mean_first_window|
          is_converged        – heuristic: std_last / |mean_last| < 0.15
        """
        if len(episode_rewards) < 2:
            return {
                "rolling_mean_last": 0.0,
                "rolling_std_last": 0.0,
                "improvement": 0.0,
                "is_converged": False,
            }

        arr = np.array(episode_rewards)
        last = arr[-window:]
        first = arr[:window]

        mean_last = float(np.mean(last))
        std_last = float(np.std(last))
        mean_first = float(np.mean(first))

        if abs(mean_first) < 1e-8:
            improvement = 0.0
        else:
            improvement = (mean_last - mean_first) / abs(mean_first)

        is_converged = bool(std_last / (abs(mean_last) + 1e-8) < 0.15)

        return {
            "rolling_mean_last": mean_last,
            "rolling_std_last": std_last,
            "improvement": float(improvement),
            "is_converged": is_converged,
        }

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compare_metrics(
        event_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Return delta dict: event_metrics - baseline_metrics for shared keys.
        Positive values = event model is better for that metric.
        Note: max_drawdown is inverted (lower is better → negative delta = better).
        """
        lower_is_better = {"max_drawdown", "volatility", "downside_deviation"}
        result = {}
        for k in event_metrics:
            if k in baseline_metrics:
                delta = event_metrics[k] - baseline_metrics[k]
                if k in lower_is_better:
                    delta = -delta   # flip sign so positive still means "event is better"
                result[f"{k}_delta"] = delta
        return result

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_metrics(self, metrics: Dict[str, float], label: str = ""):
        header = f"TRADING PERFORMANCE{' — ' + label if label else ''}"
        print("\n" + "=" * 60)
        print(header)
        print("=" * 60)
        _r = lambda k, fmt=".4f": f"{metrics.get(k, 0.0):{fmt}}"
        print(f"  Total Return:          {metrics.get('total_return', 0.0):>10.2%}")
        print(f"  Annualised Return:     {metrics.get('annualised_return', 0.0):>10.2%}")
        print(f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0.0):>10.3f}")
        print(f"  Sortino Ratio:         {metrics.get('sortino_ratio', 0.0):>10.3f}")
        print(f"  Calmar Ratio:          {metrics.get('calmar_ratio', 0.0):>10.3f}")
        print(f"  Max Drawdown:          {metrics.get('max_drawdown', 0.0):>10.2%}")
        print(f"  Annualised Volatility: {metrics.get('volatility', 0.0):>10.2%}")
        print(f"  Downside Deviation:    {metrics.get('downside_deviation', 0.0):>10.2%}")
        print(f"  Win Rate:              {metrics.get('win_rate', 0.0):>10.2%}")
        print(f"  Profit Factor:         {metrics.get('profit_factor', 0.0):>10.3f}")
        for k, v in metrics.items():
            if k.startswith("pct_"):
                print(f"  Action {k[4:].upper():<14}: {v:>10.2%}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _returns(self, portfolio_values: List[float]) -> np.ndarray:
        vals = np.array(portfolio_values, dtype=float)
        return np.diff(vals) / (vals[:-1] + 1e-8)

    def _annualisation_factor(
        self,
        returns: np.ndarray,
        dates: Optional[List],
    ) -> float:
        n = len(returns)
        if n == 0:
            return _TRADING_DAYS
        if dates and len(dates) > 1:
            try:
                days = (dates[-1] - dates[0]).days
                years = max(days / 365.25, 1e-3)
                return n / (years * 1.0)   # periods per year
            except Exception:
                pass
        return float(_TRADING_DAYS)

    def _zero_metrics(self) -> Dict[str, float]:
        return {
            "total_return": 0.0,
            "annualised_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "downside_deviation": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
