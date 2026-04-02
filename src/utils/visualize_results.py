"""
Visualization and Analysis Tools for Trading Agent Results

Shows:
- Portfolio value over time
- Sharpe/Sortino ratios
- Event impact analysis
- Comparison with baseline (no events)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import json
import os
from pathlib import Path
import logging

from src.utils.metrics import TradingMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Visualize and analyze trading agent results."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize visualizer."""
        self.metrics_calculator = TradingMetrics(risk_free_rate=risk_free_rate)
    
    def plot_portfolio_performance(
        self,
        portfolio_values: List[float],
        dates: Optional[List] = None,
        title: str = "Portfolio Performance",
        save_path: Optional[str] = None
    ):
        """
        Plot portfolio value over time.
        
        Args:
            portfolio_values: List of portfolio values
            dates: Optional list of dates
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        pv = list(portfolio_values)
        dt = list(dates) if dates else None
        if dt is not None:
            n = min(len(dt), len(pv))
            if n < len(pv) or n < len(dt):
                logger.warning(
                    "Trimming dates/values to %s points (was %s dates, %s values)",
                    n,
                    len(dt),
                    len(pv),
                )
            dt = dt[:n]
            pv = pv[:n]

        # Portfolio value over time
        ax1 = axes[0]
        if dt:
            ax1.plot(dt, pv, linewidth=2, label='Portfolio Value')
        else:
            ax1.plot(pv, linewidth=2, label='Portfolio Value')
        
        ax1.axhline(y=pv[0], color='r', linestyle='--', alpha=0.5, label='Initial Value')
        ax1.set_xlabel('Time' if not dt else 'Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns over time
        ax2 = axes[1]
        returns = self.metrics_calculator._returns(pv)
        if dt is not None and len(returns) > 0 and len(dt) == len(pv) and len(dt) >= len(returns) + 1:
            ax2.plot(dt[1 : len(returns) + 1], returns * 100, linewidth=1, alpha=0.7, color='green')
        else:
            ax2.plot(returns * 100, linewidth=1, alpha=0.7, color='green')
        
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time' if not dt else 'Date')
        ax2.set_ylabel('Daily Returns (%)')
        ax2.set_title('Daily Returns')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_comparison(
        self,
        results_with_events: Dict,
        results_without_events: Optional[Dict] = None,
        save_path: Optional[str] = None,
        label_a: Optional[str] = None,
        label_b: Optional[str] = None,
    ):
        """
        Compare two evaluation runs (e.g. event-based vs sentiment baseline).

        Uses ``label`` keys from each results dict when ``label_a`` / ``label_b``
        are not passed (as produced by ``evaluate_agent``).
        """
        name_a = label_a or results_with_events.get("label") or "Condition A"
        name_b = (
            label_b
            or (results_without_events or {}).get("label")
            or "Condition B"
        )

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Effectiveness comparison: event-based vs sentiment-based",
            fontsize=14,
            fontweight="bold",
        )

        # Portfolio comparison
        ax1 = axes[0, 0]
        dates = results_with_events.get("dates")
        pv_a = results_with_events["portfolio_values"]
        init = float(pv_a[0]) if pv_a else 0.0

        def _align_dates(pv, dlist):
            if not dlist:
                return None, pv
            n = min(len(dlist), len(pv))
            return dlist[:n], pv[:n]

        if dates:
            d_a, v_a = _align_dates(pv_a, dates)
            ax1.plot(d_a, v_a, label=name_a, linewidth=2, color="#1f77b4")
            if results_without_events:
                d_b, v_b = _align_dates(
                    results_without_events["portfolio_values"],
                    results_without_events.get("dates") or dates,
                )
                ax1.plot(d_b, v_b, label=name_b, linewidth=2, alpha=0.85, color="#ff7f0e")
        else:
            ax1.plot(pv_a, label=name_a, linewidth=2, color="#1f77b4")
            if results_without_events:
                ax1.plot(
                    results_without_events["portfolio_values"],
                    label=name_b,
                    linewidth=2,
                    alpha=0.85,
                    color="#ff7f0e",
                )

        if init > 0:
            ax1.axhline(y=init, color="gray", linestyle="--", alpha=0.6, label="Initial capital")
        ax1.set_xlabel("Date" if dates else "Step")
        ax1.set_ylabel("Portfolio value ($)")
        ax1.set_title("Figure 1. Portfolio value over the evaluation window")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Metrics comparison (clip Sortino for display when numerically exploded)
        ax2 = axes[0, 1]
        metrics_with = results_with_events.get("metrics", {})
        metrics_without = (
            results_without_events.get("metrics", {}) if results_without_events else {}
        )

        metric_names = ["sharpe_ratio", "sortino_ratio", "total_return"]
        metric_labels = ["Sharpe", "Sortino (capped†)", "Total return"]

        def _clip_sortino(v: float, cap: float = 10.0) -> float:
            if abs(v) > cap:
                return float(np.sign(v) * cap)
            return float(v)

        x = np.arange(len(metric_names))
        width = 0.35

        raw_w = [metrics_with.get(m, 0.0) for m in metric_names]
        raw_wo = (
            [metrics_without.get(m, 0.0) for m in metric_names]
            if results_without_events
            else [0.0] * len(metric_names)
        )
        values_with = [
            _clip_sortino(raw_w[1]) if i == 1 else raw_w[i]
            for i in range(len(metric_names))
        ]
        values_without = [
            _clip_sortino(raw_wo[1]) if i == 1 else raw_wo[i]
            for i in range(len(metric_names))
        ]

        ax2.bar(x - width / 2, values_with, width, label=name_a, alpha=0.85, color="#1f77b4")
        if results_without_events:
            ax2.bar(x + width / 2, values_without, width, label=name_b, alpha=0.85, color="#ff7f0e")

        ax2.set_ylabel("Value (return as decimal, e.g. 0.5 = 50%)")
        ax2.set_title("Figure 2. Key metrics side by side\n†Sortino shown in ±10 when unstable")
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_labels, fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(0, color="black", linewidth=0.5)
        
        # Drawdown comparison
        ax3 = axes[1, 0]
        drawdown_with = self._calculate_drawdown_series(
            results_with_events["portfolio_values"]
        )
        drawdown_without = (
            self._calculate_drawdown_series(results_without_events["portfolio_values"])
            if results_without_events
            else None
        )

        if dates:
            n_dd = min(len(dates), len(drawdown_with))
            d_dd = dates[:n_dd]
            dd_a = drawdown_with[:n_dd]
            ax3.fill_between(
                d_dd,
                dd_a * 100,
                0,
                alpha=0.45,
                label=name_a,
                color="#1f77b4",
            )
            if drawdown_without is not None:
                d_raw_b = results_without_events.get("dates") or dates
                n_dd_b = min(len(d_raw_b), len(drawdown_without))
                ax3.fill_between(
                    d_raw_b[:n_dd_b],
                    drawdown_without[:n_dd_b] * 100,
                    0,
                    alpha=0.45,
                    label=name_b,
                    color="#ff7f0e",
                )
        else:
            ax3.fill_between(
                range(len(drawdown_with)),
                drawdown_with * 100,
                0,
                alpha=0.45,
                label=name_a,
                color="#1f77b4",
            )
            if drawdown_without is not None:
                ax3.fill_between(
                    range(len(drawdown_without)),
                    drawdown_without * 100,
                    0,
                    alpha=0.45,
                    label=name_b,
                    color="#ff7f0e",
                )

        ax3.set_xlabel("Date" if dates else "Step")
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_title("Figure 3. Drawdown from running peak")
        ax3.legend(loc="lower right")
        ax3.grid(True, alpha=0.3)
        
        # Event impact analysis
        ax4 = axes[1, 1]
        if 'event_stats' in results_with_events:
            event_stats = results_with_events['event_stats']
            event_types = list(event_stats.keys())
            impacts = [event_stats[et].get('avg_impact', 0) for et in event_types]
            
            ax4.barh(event_types, impacts, alpha=0.7)
            ax4.set_xlabel('Average Impact on Returns')
            ax4.set_title("Figure 4. Event-type impact (if logged)")
            ax4.grid(True, alpha=0.3, axis="x")
        else:
            ax4.text(
                0.5,
                0.5,
                "Event-type aggregates not in JSON\n(use portfolio plots for comparison)",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=10,
            )
            ax4.set_title("Figure 4. Optional event breakdown")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {save_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def print_detailed_metrics(self, metrics: Dict, label: str = "Results"):
        """Print detailed metrics in a formatted way."""
        print("\n" + "="*60)
        print(f"{label.upper()} - DETAILED METRICS")
        print("="*60)
        print(f"Total Return:        {metrics.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio:        {metrics.get('sortino_ratio', 0):.3f}")
        print(f"Max Drawdown:         {metrics.get('max_drawdown', 0):.2%}")
        print(f"Volatility:           {metrics.get('volatility', 0):.2%}")
        print(f"Downside Deviation:   {metrics.get('downside_deviation', 0):.2%}")
        print("="*60 + "\n")
    
    def save_results(
        self,
        results: Dict,
        filepath: str,
        include_portfolio_values: bool = False
    ):
        """Save results to JSON file."""
        save_data = {
            'metrics': results.get('metrics', {}),
            'event_stats': results.get('event_stats', {}),
            'config': results.get('config', {})
        }
        
        if include_portfolio_values:
            save_data['portfolio_values'] = results.get('portfolio_values', [])
            save_data['dates'] = [str(d) for d in results.get('dates', [])] if results.get('dates') else None
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Saved results to {filepath}")
    
    def _calculate_drawdown_series(self, portfolio_values: List[float]) -> np.ndarray:
        """Calculate drawdown series."""
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / (peak + 1e-8)
        return drawdown


def analyze_training_logs(log_dir: str) -> Dict:
    """
    Analyze training logs from TensorBoard or CSV files.
    
    Args:
        log_dir: Directory containing training logs
        
    Returns:
        Dictionary with training statistics
    """
    # This would parse TensorBoard logs or CSV files
    # For now, return placeholder
    return {
        'episode_rewards': [],
        'losses': [],
        'learning_curve': []
    }


def compare_with_baseline(
    results_with_events: Dict,
    results_without_events: Dict
) -> Dict:
    """
    Compare results with and without event extraction.
    
    Returns:
        Dictionary with improvement metrics
    """
    metrics_with = results_with_events.get('metrics', {})
    metrics_without = results_without_events.get('metrics', {})
    
    improvements = {
        'sharpe_improvement': metrics_with.get('sharpe_ratio', 0) - metrics_without.get('sharpe_ratio', 0),
        'sortino_improvement': metrics_with.get('sortino_ratio', 0) - metrics_without.get('sortino_ratio', 0),
        'return_improvement': metrics_with.get('total_return', 0) - metrics_without.get('total_return', 0),
        'drawdown_reduction': metrics_without.get('max_drawdown', 0) - metrics_with.get('max_drawdown', 0),
    }
    
    print("\n" + "="*60)
    print("EVENT EXTRACTION IMPACT ANALYSIS")
    print("="*60)
    print(f"Sharpe Ratio Improvement:  {improvements['sharpe_improvement']:+.3f}")
    print(f"Sortino Ratio Improvement:  {improvements['sortino_improvement']:+.3f}")
    print(f"Return Improvement:         {improvements['return_improvement']:+.2%}")
    print(f"Drawdown Reduction:         {improvements['drawdown_reduction']:+.2%}")
    print("="*60 + "\n")
    
    return improvements
