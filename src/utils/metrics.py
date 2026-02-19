"""
Evaluation Metrics for Trading Agent

Implements:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Total Return
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingMetrics:
    """Calculate trading performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(
        self,
        portfolio_values: List[float],
        dates: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics from portfolio value history.
        
        Args:
            portfolio_values: List of portfolio values over time
            dates: Optional list of dates (for annualization)
            
        Returns:
            Dictionary of metrics
        """
        if len(portfolio_values) < 2:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "volatility": 0.0,
                "downside_deviation": 0.0
            }
        
        returns = self._calculate_returns(portfolio_values)
        
        metrics = {
            "sharpe_ratio": self.sharpe_ratio(returns, dates),
            "sortino_ratio": self.sortino_ratio(returns, dates),
            "max_drawdown": self.max_drawdown(portfolio_values),
            "total_return": self.total_return(portfolio_values),
            "volatility": self.volatility(returns, dates),
            "downside_deviation": self.downside_deviation(returns)
        }
        
        return metrics
    
    def _calculate_returns(self, portfolio_values: List[float]) -> np.ndarray:
        """Calculate returns from portfolio values."""
        values = np.array(portfolio_values)
        returns = np.diff(values) / (values[:-1] + 1e-8)
        return returns
    
    def sharpe_ratio(
        self,
        returns: np.ndarray,
        dates: Optional[List] = None
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev of Returns
        
        Higher is better. Proves consistent, stable profits.
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize if dates provided
        if dates and len(dates) > 1:
            days = (dates[-1] - dates[0]).days
            periods_per_year = 252 if days > 100 else 365  # Trading days vs calendar days
            n_periods = len(returns)
            annualization_factor = np.sqrt(periods_per_year / n_periods)
            
            annual_mean = mean_return * periods_per_year
            annual_std = std_return * annualization_factor
            annual_rf = self.risk_free_rate
            
            sharpe = (annual_mean - annual_rf) / (annual_std + 1e-8)
        else:
            # Use daily risk-free rate approximation
            daily_rf = self.risk_free_rate / 252
            sharpe = (mean_return - daily_rf) / (std_return + 1e-8)
        
        return float(sharpe)
    
    def sortino_ratio(
        self,
        returns: np.ndarray,
        dates: Optional[List] = None
    ) -> float:
        """
        Calculate Sortino Ratio.
        
        Sortino Ratio = (Mean Return - Risk-Free Rate) / Downside Deviation
        
        Focuses on downside risk. Higher is better.
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_dev = self.downside_deviation(returns)
        
        if downside_dev == 0:
            return 0.0
        
        # Annualize if dates provided
        if dates and len(dates) > 1:
            days = (dates[-1] - dates[0]).days
            periods_per_year = 252 if days > 100 else 365
            n_periods = len(returns)
            annualization_factor = np.sqrt(periods_per_year / n_periods)
            
            annual_mean = mean_return * periods_per_year
            annual_dd = downside_dev * annualization_factor
            annual_rf = self.risk_free_rate
            
            sortino = (annual_mean - annual_rf) / (annual_dd + 1e-8)
        else:
            daily_rf = self.risk_free_rate / 252
            sortino = (mean_return - daily_rf) / (downside_dev + 1e-8)
        
        return float(sortino)
    
    def downside_deviation(self, returns: np.ndarray) -> float:
        """
        Calculate downside deviation (standard deviation of negative returns).
        
        Used in Sortino ratio calculation.
        """
        if len(returns) == 0:
            return 0.0
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        
        return float(np.std(negative_returns))
    
    def max_drawdown(self, portfolio_values: List[float]) -> float:
        """
        Calculate Maximum Drawdown.
        
        Max Drawdown = (Peak Value - Lowest Value After Peak) / Peak Value
        
        Lower is better. Measures worst crash from peak.
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / (peak + 1e-8)
        max_dd = np.max(drawdown)
        
        return float(max_dd)
    
    def total_return(self, portfolio_values: List[float]) -> float:
        """Calculate total return percentage."""
        if len(portfolio_values) < 2:
            return 0.0
        
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        return float((final_value - initial_value) / initial_value)
    
    def volatility(
        self,
        returns: np.ndarray,
        dates: Optional[List] = None
    ) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        
        std_return = np.std(returns)
        
        if dates and len(dates) > 1:
            days = (dates[-1] - dates[0]).days
            periods_per_year = 252 if days > 100 else 365
            n_periods = len(returns)
            annualization_factor = np.sqrt(periods_per_year / n_periods)
            annual_vol = std_return * annualization_factor
            return float(annual_vol)
        
        # Daily volatility
        return float(std_return)
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a readable format."""
        print("\n" + "="*50)
        print("TRADING PERFORMANCE METRICS")
        print("="*50)
        print(f"Total Return:        {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:.3f}")
        print(f"Max Drawdown:        {metrics['max_drawdown']:.2%}")
        print(f"Volatility:          {metrics['volatility']:.2%}")
        print(f"Downside Deviation:  {metrics['downside_deviation']:.2%}")
        print("="*50 + "\n")
