"""
Market Data Fetcher

Fetches historical price data from Yahoo Finance or Stooq.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetches and processes market data for trading."""
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        frequency: str = "daily"
    ):
        """
        Initialize market data fetcher.
        
        Args:
            symbols: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency (daily, hourly)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.data_cache = {}
    
    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            DataFrame with OHLCV data
        """
        if symbol in self.data_cache:
            return self.data_cache[symbol]
        
        logger.info(f"Fetching data for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            interval = "1d" if self.frequency == "daily" else "1h"
            
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval=interval
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Calculate technical indicators
            df = self._add_technical_indicators(df)
            
            self.data_cache[symbol] = df
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols."""
        data = {}
        for symbol in self.symbols:
            df = self.fetch_data(symbol)
            if not df.empty:
                data[symbol] = df
        return data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data."""
        if df.empty:
            return df
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return df
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Fill NaN values
        df = df.bfill().fillna(0)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI when insufficient data
    
    def get_observation_features(
        self,
        symbol: str,
        date: datetime,
        lookback_window: int = 60
    ) -> np.ndarray:
        """
        Get feature vector for RL observation.
        
        Args:
            symbol: Stock ticker
            date: Current date
            lookback_window: Number of days to look back
            
        Returns:
            Feature vector
        """
        df = self.fetch_data(symbol)
        if df.empty:
            return np.zeros(50)  # Return zeros if no data
        
        # Filter data up to current date
        df_filtered = df[df.index <= date].tail(lookback_window)
        
        if df_filtered.empty:
            return np.zeros(50)
        
        # Extract features
        features = []
        
        # Price features (normalized)
        if 'close' in df_filtered.columns:
            close_prices = df_filtered['close'].values
            if len(close_prices) > 0:
                normalized_prices = (close_prices - close_prices.mean()) / (close_prices.std() + 1e-8)
                features.extend(normalized_prices[-20:])  # Last 20 days
                if len(features) < 20:
                    features.extend([0] * (20 - len(features)))
        
        # Technical indicators
        indicator_cols = [
            'sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_hist', 'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'volatility', 'volume_ratio'
        ]
        
        for col in indicator_cols:
            if col in df_filtered.columns:
                values = df_filtered[col].values
                if len(values) > 0:
                    features.append(values[-1])  # Latest value
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # Ensure fixed size
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50], dtype=np.float32)
