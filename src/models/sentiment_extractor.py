"""
Sentiment-Only Baseline Extractor

Provides a lightweight lexicon-based sentiment signal that can replace
the full Qwen event extractor in baseline comparisons.  Unlike the
structured event extractor, this module has NO understanding of event
type, horizon, or entity – it returns only an aggregate sentiment score
per article, packaged to match the EventExtractor.FEATURE_DIM API so
that TradingEnv can run either module without code changes.

Design rationale:
  The comparison experiment tests whether *structured event understanding*
  (type / direction / horizon) adds value over raw positive/negative
  sentiment.  A fair baseline must:
    (a) use the same observation-space dimensionality (FEATURE_DIM = 10)
    (b) produce signals on the same date schedule
    (c) require no LLM inference — only a static lexicon
"""

import re
import math
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lexicon (deliberately minimal — extend for production use)
# ---------------------------------------------------------------------------
_POSITIVE_WORDS = {
    "beat", "beats", "record", "growth", "profit", "gain", "gains", "surge",
    "surged", "upgraded", "upgrade", "buy", "bullish", "raised", "raise",
    "strong", "outperform", "exceeded", "exceeds", "revenue", "expansion",
    "expanding", "partnership", "launch", "launches", "innovation", "dividend",
    "approved", "approval", "acquisition", "invest", "hiring", "hired",
    "boost", "boosts", "positive", "rally", "rallied", "momentum", "recovery",
}

_NEGATIVE_WORDS = {
    "miss", "misses", "missed", "loss", "losses", "decline", "declined",
    "lawsuit", "layoff", "layoffs", "cut", "cuts", "downgraded", "downgrade",
    "sell", "bearish", "weak", "underperform", "falling", "fell", "failed",
    "failure", "investigation", "fine", "fined", "bankruptcy", "debt",
    "freeze", "frozen", "halted", "halt", "recall", "crisis", "warning",
    "reduced", "negative", "crash", "crashed", "scandal", "fraud",
}

_NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "hardly"}

_INTENSIFIERS = {
    "very": 1.3, "extremely": 1.5, "significantly": 1.4, "sharply": 1.4,
    "slightly": 0.6, "modestly": 0.7, "marginally": 0.5,
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def _compute_raw_sentiment(tokens: List[str]) -> float:
    """
    Returns a score in [-1, +1].
    Negations flip the sign of the immediately following sentiment word.
    Intensifiers scale the next sentiment word's contribution.
    """
    score = 0.0
    count = 0
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]

        # look-ahead for negation
        negate = i > 0 and tokens[i - 1] in _NEGATION_WORDS

        # look-back for intensifier
        intensity = 1.0
        if i > 0:
            intensity = _INTENSIFIERS.get(tokens[i - 1], 1.0)

        if tok in _POSITIVE_WORDS:
            contribution = 1.0 * intensity
            score += -contribution if negate else contribution
            count += 1
        elif tok in _NEGATIVE_WORDS:
            contribution = -1.0 * intensity
            score += -contribution if negate else contribution
            count += 1
        i += 1

    if count == 0:
        return 0.0
    # Normalize by count but keep saturation via tanh
    return float(math.tanh(score / count * 2.0))


def _sentence_split(text: str) -> List[str]:
    return re.split(r"[.!?;]", text)


# ---------------------------------------------------------------------------
# Public API — mirrors EventExtractor interface
# ---------------------------------------------------------------------------

class SentimentExtractor:
    """
    Lexicon-based sentiment-only extractor.

    Matches the EventExtractor public interface so TradingEnv can use
    either module transparently.  The observation vector is padded to
    FEATURE_DIM = 10, with zeros for structured features that are
    absent in a pure-sentiment model.
    """

    FEATURE_DIM = 10  # same as EventExtractor.FEATURE_DIM

    def __init__(self, ticker_aliases: Optional[Dict[str, List[str]]] = None):
        """
        Parameters
        ----------
        ticker_aliases : optional dict mapping ticker → list of name variants
            e.g. {"AAPL": ["apple", "aapl"], "MSFT": ["microsoft", "msft"]}
        """
        self.ticker_aliases: Dict[str, List[str]] = {
            k.upper(): [v.lower() for v in vs]
            for k, vs in (ticker_aliases or {}).items()
        }

    # -- EventExtractor-compatible methods ---------------------------------

    def extract_events(self, news_text: str) -> List[Dict]:
        """
        Return a synthetic single-event list derived from sentiment score.
        The 'event_type' is always 'sentiment_signal' and horizon is 'short'.
        No ticker is assigned — encode_event_features handles the mapping.
        """
        tokens = _tokenize(news_text)
        raw_score = _compute_raw_sentiment(tokens)

        direction = "up" if raw_score > 0.05 else ("down" if raw_score < -0.05 else "uncertain")
        confidence = min(abs(raw_score), 1.0)

        # Try to detect tickers from aliases
        mentions: Dict[str, float] = self._detect_mentions(tokens, raw_score)

        events = []
        if mentions:
            for ticker, score in mentions.items():
                d = "up" if score > 0.05 else ("down" if score < -0.05 else "uncertain")
                c = min(abs(score), 1.0)
                events.append({
                    "event_type": "sentiment_signal",
                    "target": {"ticker": ticker, "company": ticker, "sector": "unknown"},
                    "direction": d,
                    "confidence": c,
                    "horizon": "short",
                    "magnitude": c * 0.5,
                    "sector_contagion": False,
                    "rationale": f"Lexicon sentiment score: {score:.3f}",
                    "_raw_score": score,
                })
        elif confidence > 0:
            # No ticker detected — create a market-wide signal
            events.append({
                "event_type": "sentiment_signal",
                "target": {"ticker": "", "company": "market", "sector": "unknown"},
                "direction": direction,
                "confidence": confidence,
                "horizon": "short",
                "magnitude": confidence * 0.5,
                "sector_contagion": True,
                "rationale": f"Lexicon sentiment score: {raw_score:.3f}",
                "_raw_score": raw_score,
            })
        return events

    def extract_events_batch(self, news_texts: List[str]) -> List[List[Dict]]:
        return [self.extract_events(t) for t in news_texts]

    def encode_event_features(self, events: List[Dict], ticker: str) -> Dict:
        """
        Return a 10-key dict matching EventExtractor.FEATURE_DIM = 10.

        Structured fields (magnitude, sector_contagion, urgency,
        event_diversity) are either 0 or derived from the sentiment score.
        """
        t = ticker.upper()
        relevant = [
            e for e in events
            if e.get("target", {}).get("ticker", "").upper() == t
            or (e.get("sector_contagion", False) and e.get("target", {}).get("ticker", "") == "")
        ]

        if not relevant:
            return {k: 0.0 for k in [
                "event_count", "weighted_direction", "avg_confidence",
                "short_term_signal", "mid_term_signal", "long_term_signal",
                "magnitude", "sector_contagion", "urgency", "event_diversity",
            ]}

        direction_map = {"up": 1.0, "down": -1.0, "uncertain": 0.0}
        n = len(relevant)
        confs = [e["confidence"] for e in relevant]
        dirs = [direction_map[e["direction"]] for e in relevant]
        mags = [e.get("magnitude", 0.5) for e in relevant]

        weighted_dir = sum(c * d for c, d in zip(confs, dirs)) / n
        avg_conf = sum(confs) / n
        avg_mag = sum(mags) / n

        # All sentiment events land on short horizon only
        short_signal = weighted_dir
        mid_signal = 0.0
        long_signal = 0.0

        return {
            "event_count":        float(min(n, 10)),
            "weighted_direction": float(weighted_dir),
            "avg_confidence":     float(avg_conf),
            "short_term_signal":  float(short_signal),
            "mid_term_signal":    float(mid_signal),
            "long_term_signal":   float(long_signal),
            "magnitude":          float(avg_mag),
            "sector_contagion":   float(sum(1 for e in relevant if e.get("sector_contagion")) / n),
            "urgency":            1.0,   # sentiment signals are always short-horizon
            "event_diversity":    0.0,   # no type diversity in sentiment model
        }

    def encode_event_feature_vector(self, events: List[Dict], ticker: str) -> List[float]:
        d = self.encode_event_features(events, ticker)
        return [
            d["event_count"], d["weighted_direction"], d["avg_confidence"],
            d["short_term_signal"], d["mid_term_signal"], d["long_term_signal"],
            d["magnitude"], d["sector_contagion"], d["urgency"], d["event_diversity"],
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_mentions(self, tokens: List[str], base_score: float) -> Dict[str, float]:
        """Return {TICKER: sentiment_score} for each mentioned ticker."""
        mentions: Dict[str, float] = {}
        for ticker, aliases in self.ticker_aliases.items():
            for alias in aliases:
                if alias in tokens:
                    mentions[ticker] = base_score
                    break
        return mentions
