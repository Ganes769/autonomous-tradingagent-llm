"""
Qwen LLM-based Event Extraction Module

Extracts structured events from news articles with:
- Event type (layoff, product launch, merger, lawsuit, earnings, etc.)
- Target (company, ticker, sector)
- Expected direction (up/down/uncertain)
- Confidence score
- Time horizon (short/mid/long-term)
- Magnitude estimation
- Sector contagion flag
"""

import json
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical event-type taxonomy with default directional priors and horizon
EVENT_TYPE_PRIORS: Dict[str, Dict] = {
    "layoff":           {"direction_prior": -0.4, "horizon_prior": "short",  "magnitude": 0.6},
    "product_launch":   {"direction_prior":  0.5, "horizon_prior": "mid",    "magnitude": 0.5},
    "merger":           {"direction_prior":  0.6, "horizon_prior": "mid",    "magnitude": 0.7},
    "acquisition":      {"direction_prior":  0.6, "horizon_prior": "mid",    "magnitude": 0.7},
    "lawsuit":          {"direction_prior": -0.5, "horizon_prior": "mid",    "magnitude": 0.6},
    "earnings_miss":    {"direction_prior": -0.8, "horizon_prior": "short",  "magnitude": 0.9},
    "earnings_beat":    {"direction_prior":  0.8, "horizon_prior": "short",  "magnitude": 0.9},
    "partnership":      {"direction_prior":  0.4, "horizon_prior": "mid",    "magnitude": 0.4},
    "hiring_freeze":    {"direction_prior": -0.3, "horizon_prior": "short",  "magnitude": 0.4},
    "regulatory":       {"direction_prior": -0.4, "horizon_prior": "long",   "magnitude": 0.5},
    "dividend_cut":     {"direction_prior": -0.7, "horizon_prior": "short",  "magnitude": 0.8},
    "dividend_raise":   {"direction_prior":  0.5, "horizon_prior": "short",  "magnitude": 0.5},
    "ceo_change":       {"direction_prior":  0.0, "horizon_prior": "mid",    "magnitude": 0.4},
    "guidance_raise":   {"direction_prior":  0.7, "horizon_prior": "short",  "magnitude": 0.8},
    "guidance_cut":     {"direction_prior": -0.7, "horizon_prior": "short",  "magnitude": 0.8},
    "stock_buyback":    {"direction_prior":  0.5, "horizon_prior": "short",  "magnitude": 0.5},
    "debt_issuance":    {"direction_prior": -0.2, "horizon_prior": "long",   "magnitude": 0.3},
    "supply_chain":     {"direction_prior": -0.3, "horizon_prior": "mid",    "magnitude": 0.5},
    "macro_rate_hike":  {"direction_prior": -0.5, "horizon_prior": "long",   "magnitude": 0.7},
    "macro_rate_cut":   {"direction_prior":  0.5, "horizon_prior": "long",   "magnitude": 0.7},
}

VALID_EVENT_TYPES = set(EVENT_TYPE_PRIORS.keys())
VALID_DIRECTIONS = {"up", "down", "uncertain"}
VALID_HORIZONS = {"short", "mid", "long"}


class EventExtractor:
    """
    Extracts market-moving events from news using Qwen LLM.

    Outputs a rich 10-dim feature vector per ticker per day:
      [event_count, weighted_direction, avg_confidence,
       short_signal, mid_signal, long_signal,
       magnitude, sector_contagion, urgency, event_diversity]
    """

    FEATURE_DIM = 10  # public constant for observation space sizing

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"

        logger.info(f"Loading Qwen model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device != "cpu" else None,
        )
        self.model.eval()

        self.max_tokens = max_tokens
        self.temperature = temperature

        self.system_prompt = (
            "You are a financial event extraction system. "
            "Extract ALL market-moving events from the news article.\n\n"
            "Output ONLY valid JSON with this structure:\n"
            "{\n"
            '  "events": [\n'
            "    {\n"
            '      "event_type": "layoff|product_launch|merger|acquisition|lawsuit|'
            "earnings_miss|earnings_beat|partnership|hiring_freeze|regulatory|"
            "dividend_cut|dividend_raise|ceo_change|guidance_raise|guidance_cut|"
            'stock_buyback|debt_issuance|supply_chain|macro_rate_hike|macro_rate_cut",\n'
            '      "target": {"ticker": "AAPL", "company": "Apple Inc", "sector": "Technology"},\n'
            '      "direction": "up|down|uncertain",\n'
            '      "confidence": 0.0-1.0,\n'
            '      "horizon": "short|mid|long",\n'
            '      "magnitude": 0.0-1.0,\n'
            '      "sector_contagion": true|false,\n'
            '      "rationale": "Brief explanation"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Guidelines:\n"
            "- short horizon: days (immediate market impact)\n"
            "- mid horizon: weeks (medium-term impact)\n"
            "- long horizon: months (long-term strategic impact)\n"
            "- confidence: certainty that the event will affect stock price\n"
            "- magnitude: expected size of the price move (0=tiny, 1=huge)\n"
            "- sector_contagion: true if the event likely affects peers/sector\n"
        )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_events(self, news_text: str) -> List[Dict]:
        """Extract structured events from a single news article."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Extract events from:\n\n{news_text}"},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"{self.system_prompt}\n\nUser: {news_text}\n\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return self._parse_response(response)

    def extract_events_batch(self, news_texts: List[str]) -> List[List[Dict]]:
        """Extract events from multiple news articles."""
        return [self.extract_events(text) for text in news_texts]

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_event_features(self, events: List[Dict], ticker: str) -> Dict:
        """
        Convert events to a 10-element numerical feature dict for the RL agent.

        Features:
          event_count          – raw count of relevant events
          weighted_direction   – confidence-weighted direction [-1, +1]
          avg_confidence       – mean extraction confidence [0, 1]
          short_term_signal    – short-horizon directional signal
          mid_term_signal      – mid-horizon directional signal
          long_term_signal     – long-horizon directional signal
          magnitude            – mean expected magnitude [0, 1]
          sector_contagion     – fraction of events with contagion flag [0, 1]
          urgency              – fraction of events with short horizon [0, 1]
          event_diversity      – ratio of unique event types [0, 1]
        """
        relevant = self._filter_events(events, ticker)

        if not relevant:
            return {k: 0.0 for k in [
                "event_count", "weighted_direction", "avg_confidence",
                "short_term_signal", "mid_term_signal", "long_term_signal",
                "magnitude", "sector_contagion", "urgency", "event_diversity",
            ]}

        direction_map = {"up": 1.0, "down": -1.0, "uncertain": 0.0}
        horizon_map = {"short": 0, "mid": 1, "long": 2}

        n = len(relevant)
        confidences = [e["confidence"] for e in relevant]
        directions = [direction_map[e["direction"]] for e in relevant]
        magnitudes = [e.get("magnitude", 0.5) for e in relevant]

        weighted_direction = sum(c * d for c, d in zip(confidences, directions)) / n
        avg_confidence = sum(confidences) / n
        avg_magnitude = sum(magnitudes) / n

        horizon_buckets: Dict[int, List[float]] = {0: [], 1: [], 2: []}
        for e in relevant:
            h = horizon_map[e["horizon"]]
            horizon_buckets[h].append(e["confidence"] * direction_map[e["direction"]])

        horizon_signals = [
            sum(horizon_buckets[i]) / len(horizon_buckets[i]) if horizon_buckets[i] else 0.0
            for i in range(3)
        ]

        sector_contagion = sum(1 for e in relevant if e.get("sector_contagion", False)) / n
        urgency = len(horizon_buckets[0]) / n
        event_diversity = len({e["event_type"] for e in relevant}) / max(n, 1)

        return {
            "event_count": float(min(n, 10)),          # cap at 10
            "weighted_direction": weighted_direction,
            "avg_confidence": avg_confidence,
            "short_term_signal": horizon_signals[0],
            "mid_term_signal": horizon_signals[1],
            "long_term_signal": horizon_signals[2],
            "magnitude": avg_magnitude,
            "sector_contagion": sector_contagion,
            "urgency": urgency,
            "event_diversity": event_diversity,
        }

    def encode_event_feature_vector(self, events: List[Dict], ticker: str) -> List[float]:
        """Return the feature dict as an ordered list (consistent with FEATURE_DIM)."""
        d = self.encode_event_features(events, ticker)
        return [
            d["event_count"],
            d["weighted_direction"],
            d["avg_confidence"],
            d["short_term_signal"],
            d["mid_term_signal"],
            d["long_term_signal"],
            d["magnitude"],
            d["sector_contagion"],
            d["urgency"],
            d["event_diversity"],
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_events(self, events: List[Dict], ticker: str) -> List[Dict]:
        """Return events relevant to a ticker (also include sector-contagion events)."""
        t = ticker.upper()
        result = []
        for e in events:
            target = e.get("target", {})
            if target.get("ticker", "").upper() == t:
                result.append(e)
            elif e.get("sector_contagion", False) and target.get("ticker", ""):
                # Downweight contagion events
                e_copy = dict(e)
                e_copy["confidence"] = e["confidence"] * 0.4
                result.append(e_copy)
        return result

    def _parse_response(self, response: str) -> List[Dict]:
        """Parse LLM JSON response into a list of validated events."""
        try:
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            event_data = json.loads(response)
            events = event_data.get("events", [])

            validated = []
            for ev in events:
                ev = self._enrich_with_priors(ev)
                if self._validate_event(ev):
                    validated.append(ev)
            return validated

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return []

    def _enrich_with_priors(self, event: Dict) -> Dict:
        """Fill missing optional fields using event-type priors."""
        etype = event.get("event_type", "")
        prior = EVENT_TYPE_PRIORS.get(etype, {})

        if "magnitude" not in event:
            event["magnitude"] = prior.get("magnitude", 0.5)
        if "sector_contagion" not in event:
            event["sector_contagion"] = False
        if "rationale" not in event:
            event["rationale"] = ""

        # Clamp numeric fields
        event["confidence"] = max(0.0, min(1.0, float(event.get("confidence", 0.5))))
        event["magnitude"] = max(0.0, min(1.0, float(event.get("magnitude", 0.5))))

        return event

    def _validate_event(self, event: Dict) -> bool:
        """Validate required event fields."""
        required = ["event_type", "target", "direction", "confidence", "horizon"]
        if not all(f in event for f in required):
            return False
        if not isinstance(event.get("target"), dict):
            return False
        if event["direction"] not in VALID_DIRECTIONS:
            return False
        if event["horizon"] not in VALID_HORIZONS:
            return False
        if not (0.0 <= event["confidence"] <= 1.0):
            return False
        return True
