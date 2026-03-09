"""
Trading Environment for PPO Agent

Gymnasium-compatible environment that combines:
- Market data (OHLCV + technical indicators)           50-dim per asset
- Event features from LLM EventExtractor               10-dim per asset
- Horizon-aware signals from HorizonInterpreter         7-dim per asset
- Portfolio state                                       n_assets + 1

Action semantics (continuous Box):
  [w_1, ..., w_n, w_cash]  — target portfolio weights, sum-normalised to 1.
  The agent effectively chooses REBALANCE every step; the reward function
  penalises unnecessary churning via transaction costs.

Discrete action labels (for logging / analysis only):
  BUY  → asset weight increases significantly
  SELL → asset weight decreases significantly
  HOLD → asset weight barely changes
  REBALANCE → multiple assets change simultaneously
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from src.data.market_data import MarketDataFetcher
from src.models.event_extractor import EventExtractor
from src.models.horizon_interpreter import HorizonInterpreter, HORIZON_FEATURE_DIM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MARKET_FEATURE_DIM = 50     # from MarketDataFetcher.get_observation_features
EVENT_FEATURE_DIM = 10      # from EventExtractor.FEATURE_DIM


class TradingEnv(gym.Env):
    """
    Trading environment for PPO agent.

    Observation = market_features (50) + event_features (10) + horizon_features (7)
                  per asset  ×  n_assets  +  portfolio_state (n_assets + 1)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        symbols: List[str],
        market_data_fetcher: MarketDataFetcher,
        event_extractor,                        # EventExtractor | SentimentExtractor | Dummy
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.25,
        lookback_window: int = 60,
        news_data: Optional[Dict[str, List[str]]] = None,
        # Reward weights
        profit_weight: float = 1.0,
        event_alignment_weight: float = 0.3,
        risk_penalty_weight: float = 0.1,
        transaction_cost_penalty: float = 0.5,
        # Horizon interpreter
        horizon_interpreter: Optional[HorizonInterpreter] = None,
    ):
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

        # Reward weights
        self.profit_weight = profit_weight
        self.event_alignment_weight = event_alignment_weight
        self.risk_penalty_weight = risk_penalty_weight
        self.transaction_cost_penalty = transaction_cost_penalty

        # Horizon interpreter (shared or per-env)
        self.horizon_interpreter = horizon_interpreter or HorizonInterpreter(symbols)

        # Fetch market data
        logger.info("Loading market data...")
        self.market_data = market_data_fetcher.fetch_all()

        self.dates = self._get_common_dates()
        if len(self.dates) == 0:
            raise ValueError("No common dates found across all symbols")
        logger.info(f"Loaded {len(self.dates)} trading days")

        # ----- Action & Observation Spaces -----
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        per_asset_dim = MARKET_FEATURE_DIM + EVENT_FEATURE_DIM + HORIZON_FEATURE_DIM
        obs_dim = per_asset_dim * self.n_assets + (self.n_assets + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._prev_holdings: Optional[np.ndarray] = None
        self.reset()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets)
        self._prev_holdings = np.zeros(self.n_assets)
        self.portfolio_value_history: List[float] = [self.initial_cash]
        self.action_history: List[np.ndarray] = []
        self.reward_history: List[float] = []
        self.action_labels: List[str] = []

        self.horizon_interpreter.reset()

        obs = self._get_observation()
        return obs, self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, 0.0, 1.0)
        action = action / (action.sum() + 1e-8)

        if self.current_step >= len(self.dates):
            return self._get_observation(), 0.0, True, False, self._get_info()

        current_date = self.dates[self.current_step]
        current_prices = self._get_current_prices(current_date)

        if current_prices is None:
            self.current_step += 1
            return self._get_observation(), 0.0, False, False, self._get_info()

        portfolio_value = self.cash + np.sum(self.holdings * current_prices)

        # --- Ingest daily events into horizon interpreter ---
        events = self._get_events_for_date(current_date)
        self.horizon_interpreter.ingest_events(events, current_date)

        # --- Rebalance ---
        target_values = action[:-1] * portfolio_value
        target_shares = target_values / (current_prices + 1e-8)
        trades = target_shares - self.holdings

        # Apply max-position-size constraint
        max_value = self.max_position_size * portfolio_value
        target_shares = np.minimum(target_shares, max_value / (current_prices + 1e-8))

        trade_value = np.abs(trades) * current_prices
        total_cost = np.sum(trade_value) * self.transaction_cost

        self._prev_holdings = self.holdings.copy()
        self.holdings = target_shares.copy()
        self.cash = action[-1] * portfolio_value - total_cost
        self.cash = max(self.cash, 0.0)

        # --- Advance one day ---
        self.horizon_interpreter.step(days=1.0)
        self.current_step += 1

        # --- Compute reward ---
        reward = 0.0
        if self.current_step < len(self.dates):
            next_date = self.dates[self.current_step]
            next_prices = self._get_current_prices(next_date)

            if next_prices is not None:
                next_portfolio_value = self.cash + np.sum(self.holdings * next_prices)
                self.portfolio_value_history.append(next_portfolio_value)

                reward = self._compute_reward(
                    portfolio_value,
                    next_portfolio_value,
                    total_cost,
                    action,
                    current_date,
                )

        self.reward_history.append(reward)
        self.action_history.append(action.copy())
        self.action_labels.append(self._classify_action(action))

        terminated = self.current_step >= len(self.dates)
        obs = self._get_observation()
        return obs, reward, terminated, False, self._get_info()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        idx = min(self.current_step, len(self.dates) - 1)
        current_date = self.dates[idx]

        obs = []

        events = self._get_events_for_date(current_date)

        for symbol in self.symbols:
            # Market features (50-dim)
            mf = self.market_data_fetcher.get_observation_features(
                symbol, current_date, self.lookback_window
            )
            obs.extend(mf.tolist())

            # Event features (10-dim)
            ef = self.event_extractor.encode_event_feature_vector(events, symbol) \
                if hasattr(self.event_extractor, "encode_event_feature_vector") \
                else _legacy_encode(self.event_extractor, events, symbol)
            obs.extend(ef)

            # Horizon features (7-dim)
            hf = self.horizon_interpreter.encode_horizon_feature_vector(symbol)
            obs.extend(hf)

        # Portfolio state
        current_prices = self._get_current_prices(current_date)
        if current_prices is not None:
            pv = self.cash + np.sum(self.holdings * current_prices)
            pw = (self.holdings * current_prices) / (pv + 1e-8)
            cw = self.cash / (pv + 1e-8)
            obs.extend(pw.tolist())
            obs.append(cw)
        else:
            obs.extend([0.0] * (self.n_assets + 1))

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        prev_value: float,
        next_value: float,
        total_cost: float,
        action: np.ndarray,
        date: datetime,
    ) -> float:
        """
        Reward = profit_weight * normalised_profit
               - transaction_cost_penalty * normalised_cost
               - risk_penalty_weight * volatility_penalty
               + event_alignment_weight * alignment_bonus

        All components are normalised by initial_cash so the scale stays
        consistent regardless of portfolio size.
        """
        norm = self.initial_cash

        # 1. Profit signal
        profit = (next_value - prev_value) / norm
        r_profit = self.profit_weight * profit

        # 2. Transaction cost penalty
        r_cost = -self.transaction_cost_penalty * (total_cost / norm)

        # 3. Risk penalty (downside volatility of recent returns)
        r_risk = -self.risk_penalty_weight * self._volatility_penalty()

        # 4. Event alignment bonus
        r_event = self.event_alignment_weight * self._event_alignment_bonus(action)

        return float(r_profit + r_cost + r_risk + r_event)

    def _volatility_penalty(self) -> float:
        """Penalise downside volatility (semi-deviation) of recent portfolio."""
        if len(self.portfolio_value_history) < 10:
            return 0.0
        vals = np.array(self.portfolio_value_history[-20:])
        rets = np.diff(vals) / (vals[:-1] + 1e-8)
        neg = rets[rets < 0]
        if len(neg) == 0:
            return 0.0
        return float(np.std(neg) * 100.0)

    def _event_alignment_bonus(self, action: np.ndarray) -> float:
        """
        Use HorizonInterpreter composite signals to evaluate how well the
        agent's portfolio weights align with current event signals.
        Reward is averaged across all assets.
        """
        bonus = 0.0
        cash_weight = float(action[-1])
        for i, symbol in enumerate(self.symbols):
            asset_weight = float(action[i])
            alignment = self.horizon_interpreter.get_alignment_signal(
                symbol, asset_weight, cash_weight
            )
            bonus += alignment
        return bonus / max(self.n_assets, 1)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_common_dates(self) -> List[datetime]:
        if not self.market_data:
            return []
        common = None
        for df in self.market_data.values():
            if df.empty:
                continue
            s = set(df.index)
            common = s if common is None else common.intersection(s)
        return sorted(list(common)) if common else []

    def _get_current_prices(self, date: datetime) -> Optional[np.ndarray]:
        prices = []
        for symbol in self.symbols:
            if symbol not in self.market_data:
                return None
            df = self.market_data[symbol]
            sub = df[df.index <= date]
            if sub.empty:
                return None
            prices.append(sub.iloc[-1]["close"])
        return np.array(prices, dtype=np.float32)

    def _get_events_for_date(self, date: datetime) -> List[Dict]:
        date_str = date.strftime("%Y-%m-%d")
        articles = self.news_data.get(date_str, [])
        all_events = []
        for article in articles:
            all_events.extend(self.event_extractor.extract_events(article))
        return all_events

    def _classify_action(self, action: np.ndarray) -> str:
        """Map continuous action to a human-readable label."""
        if self._prev_holdings is None:
            return "HOLD"
        current_prices = self._get_current_prices(
            self.dates[min(self.current_step, len(self.dates) - 1)]
        )
        if current_prices is None:
            return "HOLD"

        pv = self.cash + np.sum(self.holdings * current_prices)
        if pv <= 0:
            return "HOLD"

        new_weights = (self.holdings * current_prices) / pv
        prev_pv = self.initial_cash if self._prev_holdings is None else (
            self.cash + np.sum(self._prev_holdings * current_prices)
        )
        if prev_pv <= 0:
            return "HOLD"
        old_weights = (self._prev_holdings * current_prices) / prev_pv

        delta = new_weights - old_weights
        big_buys = np.sum(delta > 0.05)
        big_sells = np.sum(delta < -0.05)

        if big_buys >= 2 or big_sells >= 2:
            return "REBALANCE"
        if big_buys == 1 and big_sells == 0:
            return "BUY"
        if big_sells == 1 and big_buys == 0:
            return "SELL"
        return "HOLD"

    def _get_info(self) -> Dict:
        idx = max(0, min(self.current_step - 1, len(self.dates) - 1))
        if self.current_step > 0:
            date = self.dates[idx]
            prices = self._get_current_prices(date)
            pv = self.cash + np.sum(self.holdings * prices) if prices is not None else self.initial_cash
        else:
            pv = self.initial_cash

        return {
            "portfolio_value": float(pv),
            "cash": float(self.cash),
            "holdings": self.holdings.tolist(),
            "step": self.current_step,
            "total_steps": len(self.dates),
            "last_action_label": self.action_labels[-1] if self.action_labels else "N/A",
        }


# ---------------------------------------------------------------------------
# Compatibility shim: support old 6-feature encode_event_features API
# ---------------------------------------------------------------------------

def _legacy_encode(extractor, events: List[Dict], symbol: str) -> List[float]:
    """Wrap old 6-key encode_event_features into the 10-dim vector with zeros."""
    d = extractor.encode_event_features(events, symbol)
    base = [
        d.get("event_count", 0.0),
        d.get("weighted_direction", 0.0),
        d.get("avg_confidence", 0.0),
        d.get("short_term_signal", 0.0),
        d.get("mid_term_signal", 0.0),
        d.get("long_term_signal", 0.0),
    ]
    # Pad to 10 dims
    base.extend([0.0] * (EVENT_FEATURE_DIM - len(base)))
    return base
