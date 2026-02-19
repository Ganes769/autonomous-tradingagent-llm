"""
Trading Environment for PPO Agent

Gymnasium-compatible environment that combines:
- Market data (OHLCV + technical indicators)
- Event features from Qwen LLM
- Portfolio state
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from src.data.market_data import MarketDataFetcher
from src.models.event_extractor import EventExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Trading environment for PPO agent.
    
    Action space: Continuous actions for portfolio weights [0, 1] for each asset + cash
    Observation space: Market features + event features + portfolio state
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        symbols: List[str],
        market_data_fetcher: MarketDataFetcher,
        event_extractor: EventExtractor,
        initial_cash: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.25,
        lookback_window: int = 60,
        news_data: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize trading environment.
        
        Args:
            symbols: List of stock tickers to trade
            market_data_fetcher: Market data fetcher instance
            event_extractor: Event extractor instance
            initial_cash: Starting cash amount
            transaction_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
            max_position_size: Maximum position size per asset (fraction of portfolio)
            lookback_window: Days of historical data for observation
            news_data: Dict mapping dates to news articles
        """
        super().__init__()
        
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.market_data_fetcher = market_data_fetcher
        self.event_extractor = event_extractor
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.news_data = news_data or {}
        
        # Fetch all market data
        logger.info("Loading market data...")
        self.market_data = market_data_fetcher.fetch_all()
        
        # Get common date range
        self.dates = self._get_common_dates()
        if len(self.dates) == 0:
            raise ValueError("No common dates found across all symbols")
        
        logger.info(f"Loaded {len(self.dates)} trading days")
        
        # Action space: portfolio weights for each asset + cash
        # Actions are normalized to sum to 1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets + 1,),  # +1 for cash
            dtype=np.float32
        )
        
        # Observation space: market features + event features + portfolio state
        market_feature_dim = 50  # From market_data.get_observation_features
        event_feature_dim = 6  # From event_extractor.encode_event_features
        portfolio_state_dim = self.n_assets + 1  # Holdings + cash
        
        obs_dim = (
            market_feature_dim * self.n_assets +  # Market features per asset
            event_feature_dim * self.n_assets +   # Event features per asset
            portfolio_state_dim                    # Portfolio state
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _get_common_dates(self) -> List[datetime]:
        """Get common trading dates across all symbols."""
        if not self.market_data:
            return []
        
        common_dates = None
        for symbol, df in self.market_data.items():
            if df.empty:
                continue
            dates = set(df.index)
            if common_dates is None:
                common_dates = dates
            else:
                common_dates = common_dates.intersection(dates)
        
        if common_dates is None:
            return []
        
        return sorted(list(common_dates))
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets)  # Number of shares per asset
        self.portfolio_value_history = [self.initial_cash]
        self.action_history = []
        self.reward_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Portfolio weights [w1, w2, ..., wn, w_cash]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to sum to 1
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # Get current date and prices
        if self.current_step >= len(self.dates):
            # Episode finished
            observation = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        current_date = self.dates[self.current_step]
        current_prices = self._get_current_prices(current_date)
        
        if current_prices is None:
            # Skip if no price data
            self.current_step += 1
            observation = self._get_observation()
            reward = 0.0
            terminated = False
            truncated = False
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        # Calculate current portfolio value
        portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        
        # Rebalance portfolio according to action
        target_values = action[:-1] * portfolio_value  # Exclude cash weight
        target_shares = target_values / (current_prices + 1e-8)
        
        # Calculate trades
        trades = target_shares - self.holdings
        
        # Apply transaction costs
        trade_value = np.abs(trades) * current_prices
        total_cost = np.sum(trade_value) * self.transaction_cost
        
        # Update holdings and cash
        self.holdings = target_shares.copy()
        self.cash = action[-1] * portfolio_value - total_cost
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        if self.current_step < len(self.dates):
            next_date = self.dates[self.current_step]
            next_prices = self._get_current_prices(next_date)
            
            if next_prices is not None:
                next_portfolio_value = self.cash + np.sum(self.holdings * next_prices)
                profit = next_portfolio_value - portfolio_value
                
                # Get event alignment bonus
                event_bonus = self._calculate_event_alignment_bonus(
                    current_date, action, current_prices
                )
                
                # Risk penalty (volatility)
                risk_penalty = self._calculate_risk_penalty(current_date)
                
                # Total reward
                reward = profit - total_cost - risk_penalty + event_bonus
                
                self.portfolio_value_history.append(next_portfolio_value)
            else:
                reward = 0.0
        else:
            reward = 0.0
        
        self.reward_history.append(reward)
        self.action_history.append(action.copy())
        
        # Check if episode is done
        terminated = self.current_step >= len(self.dates)
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_current_prices(self, date: datetime) -> Optional[np.ndarray]:
        """Get current prices for all symbols."""
        prices = []
        for symbol in self.symbols:
            if symbol not in self.market_data:
                return None
            df = self.market_data[symbol]
            date_data = df[df.index <= date]
            if date_data.empty:
                return None
            prices.append(date_data.iloc[-1]['close'])
        return np.array(prices)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        if self.current_step >= len(self.dates):
            current_date = self.dates[-1]
        else:
            current_date = self.dates[self.current_step]
        
        obs_features = []
        
        # Market features for each asset
        for symbol in self.symbols:
            market_features = self.market_data_fetcher.get_observation_features(
                symbol, current_date, self.lookback_window
            )
            obs_features.extend(market_features)
        
        # Event features for each asset
        events = self._get_events_for_date(current_date)
        for symbol in self.symbols:
            event_features = self.event_extractor.encode_event_features(events, symbol)
            event_feature_vec = [
                event_features['event_count'],
                event_features['weighted_direction'],
                event_features['avg_confidence'],
                event_features['short_term_signal'],
                event_features['mid_term_signal'],
                event_features['long_term_signal']
            ]
            obs_features.extend(event_feature_vec)
        
        # Portfolio state
        current_prices = self._get_current_prices(current_date)
        if current_prices is not None:
            portfolio_value = self.cash + np.sum(self.holdings * current_prices)
            position_values = self.holdings * current_prices
            position_weights = position_values / (portfolio_value + 1e-8)
            cash_weight = self.cash / (portfolio_value + 1e-8)
            
            obs_features.extend(position_weights.tolist())
            obs_features.append(cash_weight)
        else:
            obs_features.extend([0.0] * (self.n_assets + 1))
        
        return np.array(obs_features, dtype=np.float32)
    
    def _get_events_for_date(self, date: datetime) -> List[Dict]:
        """Get events for a specific date from news data."""
        date_str = date.strftime("%Y-%m-%d")
        if date_str not in self.news_data:
            return []
        
        news_articles = self.news_data[date_str]
        all_events = []
        
        for article in news_articles:
            events = self.event_extractor.extract_events(article)
            all_events.extend(events)
        
        return all_events
    
    def _calculate_event_alignment_bonus(
        self,
        date: datetime,
        action: np.ndarray,
        prices: np.ndarray
    ) -> float:
        """
        Calculate event alignment bonus.
        
        If agent reacts correctly to events, give bonus.
        """
        events = self._get_events_for_date(date)
        if not events:
            return 0.0
        
        bonus = 0.0
        
        for event in events:
            ticker = event.get("target", {}).get("ticker", "")
            if ticker not in self.symbols:
                continue
            
            symbol_idx = self.symbols.index(ticker)
            direction = event["direction"]
            confidence = event["confidence"]
            
            # Action weight for this asset (excluding cash)
            asset_weight = action[symbol_idx]
            cash_weight = action[-1]
            
            # Check alignment
            if direction == "up" and asset_weight > 0.1:  # Increased exposure
                bonus += confidence * 0.01 * asset_weight
            elif direction == "down" and cash_weight > 0.3:  # Reduced exposure (more cash)
                bonus += confidence * 0.01 * cash_weight
            elif direction == "uncertain" and cash_weight > 0.2:  # Cautious
                bonus += confidence * 0.005 * cash_weight
        
        return bonus
    
    def _calculate_risk_penalty(self, date: datetime) -> float:
        """Calculate risk penalty based on portfolio volatility."""
        if len(self.portfolio_value_history) < 20:
            return 0.0
        
        recent_values = np.array(self.portfolio_value_history[-20:])
        returns = np.diff(recent_values) / (recent_values[:-1] + 1e-8)
        volatility = np.std(returns)
        
        # Penalty increases with volatility
        return volatility * 100.0
    
    def _get_info(self) -> Dict:
        """Get info dictionary."""
        if self.current_step > 0 and self.current_step <= len(self.dates):
            current_date = self.dates[self.current_step - 1]
            current_prices = self._get_current_prices(current_date)
            if current_prices is not None:
                portfolio_value = self.cash + np.sum(self.holdings * current_prices)
            else:
                portfolio_value = self.initial_cash
        else:
            portfolio_value = self.initial_cash
        
        return {
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.tolist(),
            "step": self.current_step,
            "total_steps": len(self.dates)
        }
