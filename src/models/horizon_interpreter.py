"""
Horizon-Aware Event Interpretation Module

Models the temporal propagation of market-moving events across
short- (days), mid- (weeks), and long-term (months) horizons using
exponential decay. Maintains a rolling state that can be queried at
each trading step for horizon-weighted signals.

Architecture:
  - Each event is assigned a half-life determined by its horizon tag.
  - At every step the stored event signals are attenuated by the decay
    factor for one day's elapsed time.
  - A blended signal vector is produced that combines all active
    horizon channels with configurable weights for the RL observation.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Half-life in trading days for each horizon bucket
HORIZON_HALF_LIFE: Dict[str, float] = {
    "short": 3.0,    # ~3 trading days (≈1 calendar week)
    "mid":   15.0,   # ~15 trading days (≈3 calendar weeks)
    "long":  60.0,   # ~60 trading days (≈3 calendar months)
}

# Weight of each horizon in the blended "composite" signal
HORIZON_BLEND_WEIGHTS: Dict[str, float] = {
    "short": 0.50,
    "mid":   0.30,
    "long":  0.20,
}

# Feature dimension contributed per ticker by this module
HORIZON_FEATURE_DIM = 7   # see encode_horizon_features docstring


def _decay_factor(half_life: float, days: float = 1.0) -> float:
    """Compute daily exponential decay multiplier for a given half-life."""
    return 0.5 ** (days / half_life)


class HorizonInterpreter:
    """
    Maintains decaying event signals per ticker across three time horizons.

    Usage
    -----
    interpreter = HorizonInterpreter(symbols)

    # At each trading day:
    interpreter.ingest_events(events, current_date)   # add new events
    interpreter.step()                                 # apply one-day decay
    features = interpreter.encode_horizon_features(ticker)
    """

    def __init__(
        self,
        symbols: List[str],
        horizon_weights: Optional[Dict[str, float]] = None,
        half_lives: Optional[Dict[str, float]] = None,
        max_history: int = 120,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.half_lives = half_lives or dict(HORIZON_HALF_LIFE)
        self.blend_weights = horizon_weights or dict(HORIZON_BLEND_WEIGHTS)
        self.max_history = max_history

        # Per-ticker, per-horizon: current decaying signal value [-1, +1]
        self._signals: Dict[str, Dict[str, float]] = {
            s: {h: 0.0 for h in HORIZON_HALF_LIFE} for s in self.symbols
        }

        # Per-ticker: last N composite signals for momentum calculation
        self._history: Dict[str, deque] = {
            s: deque(maxlen=max_history) for s in self.symbols
        }

        self._current_date: Optional[datetime] = None
        self._days_since_last: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_events(self, events: List[Dict], date: Optional[datetime] = None):
        """
        Absorb newly extracted events into the per-horizon signal buffers.

        Parameters
        ----------
        events : validated events from EventExtractor
        date   : trading date (used for logging; decay is step-based)
        """
        direction_map = {"up": 1.0, "down": -1.0, "uncertain": 0.0}

        for event in events:
            ticker = event.get("target", {}).get("ticker", "").upper()
            if ticker not in self.symbols:
                continue

            h = event.get("horizon", "short")
            direction = direction_map.get(event.get("direction", "uncertain"), 0.0)
            confidence = float(event.get("confidence", 0.5))
            magnitude = float(event.get("magnitude", 0.5))

            signal_delta = direction * confidence * magnitude

            # Additive injection; clipped to [-1, +1]
            self._signals[ticker][h] = np.clip(
                self._signals[ticker][h] + signal_delta, -1.0, 1.0
            )

            # Sector contagion: inject a diluted signal into peers
            if event.get("sector_contagion", False):
                sector = event.get("target", {}).get("sector", "")
                for peer in self.symbols:
                    if peer != ticker:
                        self._signals[peer][h] = np.clip(
                            self._signals[peer][h] + signal_delta * 0.25,
                            -1.0, 1.0,
                        )

    def step(self, days: float = 1.0):
        """
        Advance time by `days` trading days — applies exponential decay to
        every per-ticker, per-horizon signal and records composite history.
        """
        for ticker in self.symbols:
            for h, hl in self.half_lives.items():
                factor = _decay_factor(hl, days)
                self._signals[ticker][h] *= factor

            composite = self._composite_signal(ticker)
            self._history[ticker].append(composite)

    def encode_horizon_features(self, ticker: str) -> Dict[str, float]:
        """
        Return a 7-element feature dict for the RL observation vector.

        Features
        --------
        short_signal   – current short-horizon decayed signal
        mid_signal     – current mid-horizon decayed signal
        long_signal    – current long-horizon decayed signal
        composite      – blend-weighted combination of all horizons
        momentum       – mean change in composite over last 5 steps
        trend          – sign of (composite - mean composite over 20 steps)
        signal_energy  – L2 norm across horizon channels (overall activity)
        """
        t = ticker.upper()
        if t not in self.symbols:
            return {k: 0.0 for k in [
                "short_signal", "mid_signal", "long_signal",
                "composite", "momentum", "trend", "signal_energy",
            ]}

        short_s = self._signals[t]["short"]
        mid_s = self._signals[t]["mid"]
        long_s = self._signals[t]["long"]
        composite = self._composite_signal(t)

        hist = list(self._history[t])
        if len(hist) >= 5:
            diffs = np.diff(hist[-5:])
            momentum = float(np.mean(diffs))
        else:
            momentum = 0.0

        if len(hist) >= 20:
            trend = float(np.sign(composite - np.mean(hist[-20:])))
        else:
            trend = 0.0

        signal_energy = float(np.sqrt(short_s**2 + mid_s**2 + long_s**2))

        return {
            "short_signal":  float(short_s),
            "mid_signal":    float(mid_s),
            "long_signal":   float(long_s),
            "composite":     float(composite),
            "momentum":      float(np.clip(momentum, -1.0, 1.0)),
            "trend":         float(trend),
            "signal_energy": float(np.clip(signal_energy, 0.0, 1.73)),  # max sqrt(3)
        }

    def encode_horizon_feature_vector(self, ticker: str) -> List[float]:
        """Return horizon features as a plain list (matches HORIZON_FEATURE_DIM)."""
        d = self.encode_horizon_features(ticker)
        return [
            d["short_signal"],
            d["mid_signal"],
            d["long_signal"],
            d["composite"],
            d["momentum"],
            d["trend"],
            d["signal_energy"],
        ]

    def get_alignment_signal(
        self, ticker: str, action_weight: float, cash_weight: float
    ) -> float:
        """
        Compute the event-horizon alignment score for a trading action.

        Returns a scalar in [-1, +1]:
          > 0 means agent action aligns with current event signal
          < 0 means agent action contradicts current event signal
        """
        t = ticker.upper()
        if t not in self.symbols:
            return 0.0

        composite = self._composite_signal(t)

        # Translate portfolio weight into a directional signal:
        #   high asset weight → bullish, high cash weight → bearish
        net_asset = action_weight - cash_weight
        alignment = composite * net_asset

        # Confidence-gate: only apply if composite is meaningfully non-zero
        if abs(composite) < 0.05:
            return 0.0

        return float(np.clip(alignment, -1.0, 1.0))

    def reset(self):
        """Reset all signals and history (called at episode start)."""
        for ticker in self.symbols:
            for h in self.half_lives:
                self._signals[ticker][h] = 0.0
            self._history[ticker].clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _composite_signal(self, ticker: str) -> float:
        t = ticker.upper()
        total_weight = sum(self.blend_weights.values())
        composite = sum(
            self.blend_weights.get(h, 0.0) * self._signals[t][h]
            for h in self.half_lives
        ) / (total_weight + 1e-8)
        return float(np.clip(composite, -1.0, 1.0))
