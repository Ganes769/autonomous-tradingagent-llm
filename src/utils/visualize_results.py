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
        
        # Portfolio value over time
        ax1 = axes[0]
        if dates:
            ax1.plot(dates, portfolio_values, linewidth=2, label='Portfolio Value')
        else:
            ax1.plot(portfolio_values, linewidth=2, label='Portfolio Value')
        
        ax1.axhline(y=portfolio_values[0], color='r', linestyle='--', alpha=0.5, label='Initial Value')
        ax1.set_xlabel('Time' if not dates else 'Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns over time
        ax2 = axes[1]
        returns = self.metrics_calculator._calculate_returns(portfolio_values)
        if dates and len(dates) > len(returns):
            return_dates = dates[1:] if len(dates) == len(portfolio_values) else dates[:len(returns)]
            ax2.plot(return_dates, returns * 100, linewidth=1, alpha=0.7, color='green')
        else:
            ax2.plot(returns * 100, linewidth=1, alpha=0.7, color='green')
        
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time' if not dates else 'Date')
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
        save_path: Optional[str] = None
    ):
        """
        Compare performance with and without event extraction.
        
        Args:
            results_with_events: Dict with 'portfolio_values', 'dates', 'metrics'
            results_without_events: Optional baseline results
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio comparison
        ax1 = axes[0, 0]
        dates = results_with_events.get('dates')
        if dates:
            ax1.plot(dates, results_with_events['portfolio_values'], 
                    label='With Events', linewidth=2)
            if results_without_events:
                ax1.plot(dates, results_without_events['portfolio_values'],
                        label='Without Events', linewidth=2, alpha=0.7)
        else:
            ax1.plot(results_with_events['portfolio_values'], 
                    label='With Events', linewidth=2)
            if results_without_events:
                ax1.plot(results_without_events['portfolio_values'],
                        label='Without Events', linewidth=2, alpha=0.7)
        
        ax1.axhline(y=results_with_events['portfolio_values'][0], 
                    color='r', linestyle='--', alpha=0.5, label='Initial Value')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Value Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Metrics comparison
        ax2 = axes[0, 1]
        metrics_with = results_with_events.get('metrics', {})
        metrics_without = results_without_events.get('metrics', {}) if results_without_events else {}
        
        metric_names = ['sharpe_ratio', 'sortino_ratio', 'total_return']
        metric_labels = ['Sharpe Ratio', 'Sortino Ratio', 'Total Return']
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        values_with = [metrics_with.get(m, 0) for m in metric_names]
        values_without = [metrics_without.get(m, 0) for m in metric_names] if metrics_without else [0] * len(metric_names)
        
        ax2.bar(x - width/2, values_with, width, label='With Events', alpha=0.8)
        if results_without_events:
            ax2.bar(x + width/2, values_without, width, label='Without Events', alpha=0.8)
        
        ax2.set_ylabel('Value')
        ax2.set_title('Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Drawdown comparison
        ax3 = axes[1, 0]
        drawdown_with = self._calculate_drawdown_series(results_with_events['portfolio_values'])
        drawdown_without = self._calculate_drawdown_series(
            results_without_events['portfolio_values']
        ) if results_without_events else None
        
        if dates:
            ax3.fill_between(dates[:len(drawdown_with)], drawdown_with * 100, 0,
                            alpha=0.5, label='With Events', color='blue')
            if drawdown_without:
                ax3.fill_between(dates[:len(drawdown_without)], drawdown_without * 100, 0,
                                alpha=0.5, label='Without Events', color='red')
        else:
            ax3.fill_between(range(len(drawdown_with)), drawdown_with * 100, 0,
                            alpha=0.5, label='With Events', color='blue')
            if drawdown_without:
                ax3.fill_between(range(len(drawdown_without)), drawdown_without * 100, 0,
                                alpha=0.5, label='Without Events', color='red')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_title('Drawdown Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Event impact analysis
        ax4 = axes[1, 1]
        if 'event_stats' in results_with_events:
            event_stats = results_with_events['event_stats']
            event_types = list(event_stats.keys())
            impacts = [event_stats[et].get('avg_impact', 0) for et in event_types]
            
            ax4.barh(event_types, impacts, alpha=0.7)
            ax4.set_xlabel('Average Impact on Returns')
            ax4.set_title('Event Type Impact Analysis')
            ax4.grid(True, alpha=0.3, axis='x')
        else:
            ax4.text(0.5, 0.5, 'Event statistics not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Event Impact Analysis')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
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
