"""
Qwen LLM-based Event Extraction Module

Extracts structured events from news articles with:
- Event type (layoff, product launch, merger, lawsuit, earnings, etc.)
- Target (company, ticker, sector)
- Expected direction (up/down/uncertain)
- Confidence score
- Time horizon (short/mid/long-term)
"""

import json
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventExtractor:
    """Extracts market-moving events from news using Qwen LLM."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_tokens: int = 512,
        temperature: float = 0.1
    ):
        """
        Initialize the event extractor.
        
        Args:
            model_name: HuggingFace model name for Qwen
            device: Device to run on (cuda, cpu, mps)
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        
        logger.info(f"Loading Qwen model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device != "cpu" else None
        )
        self.model.eval()
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # System prompt for event extraction
        self.system_prompt = """You are a financial event extraction system. Extract market-moving events from news articles.

Output ONLY valid JSON with this structure:
{
  "events": [
    {
      "event_type": "layoff|product_launch|merger|lawsuit|earnings_miss|earnings_beat|partnership|hiring_freeze|regulatory|acquisition",
      "target": {
        "ticker": "AAPL",
        "company": "Apple Inc",
        "sector": "Technology"
      },
      "direction": "up|down|uncertain",
      "confidence": 0.0-1.0,
      "horizon": "short|mid|long",
      "rationale": "Brief explanation"
    }
  ]
}

Guidelines:
- short horizon: days (immediate market impact)
- mid horizon: weeks (medium-term impact)
- long horizon: months (long-term strategic impact)
- confidence: 0.0-1.0 based on event clarity and market relevance
- direction: up (positive), down (negative), uncertain (ambiguous)
"""
    
    def extract_events(self, news_text: str) -> List[Dict]:
        """
        Extract events from a news article.
        
        Args:
            news_text: Raw news article text
            
        Returns:
            List of extracted events
        """
        prompt = f"{self.system_prompt}\n\nNews article:\n{news_text}\n\nExtract events:"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Extract market-moving events from this news:\n\n{news_text}"}
        ]
        
        # Format for Qwen
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat template
            text = f"{self.system_prompt}\n\nUser: {news_text}\n\nAssistant:"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            event_data = json.loads(response)
            events = event_data.get("events", [])
            
            # Validate and normalize events
            validated_events = []
            for event in events:
                if self._validate_event(event):
                    validated_events.append(event)
            
            return validated_events
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response was: {response}")
            return []
    
    def extract_events_batch(self, news_texts: List[str]) -> List[List[Dict]]:
        """Extract events from multiple news articles."""
        return [self.extract_events(text) for text in news_texts]
    
    def _validate_event(self, event: Dict) -> bool:
        """Validate event structure."""
        required_fields = ["event_type", "target", "direction", "confidence", "horizon"]
        
        if not all(field in event for field in required_fields):
            return False
        
        if "target" not in event or not isinstance(event["target"], dict):
            return False
        
        if event["direction"] not in ["up", "down", "uncertain"]:
            return False
        
        if event["horizon"] not in ["short", "mid", "long"]:
            return False
        
        if not (0.0 <= event["confidence"] <= 1.0):
            return False
        
        return True
    
    def encode_event_features(self, events: List[Dict], ticker: str) -> Dict:
        """
        Convert events to numerical features for the RL agent.
        
        Args:
            events: List of extracted events
            ticker: Stock ticker to filter events for
            
        Returns:
            Dictionary with encoded features
        """
        # Filter events for this ticker
        relevant_events = [
            e for e in events
            if e.get("target", {}).get("ticker", "").upper() == ticker.upper()
        ]
        
        if not relevant_events:
            # Return neutral features if no events
            return {
                "event_count": 0,
                "weighted_direction": 0.0,  # -1 (down) to +1 (up)
                "avg_confidence": 0.0,
                "short_term_signal": 0.0,
                "mid_term_signal": 0.0,
                "long_term_signal": 0.0
            }
        
        # Aggregate event signals
        direction_map = {"up": 1.0, "down": -1.0, "uncertain": 0.0}
        horizon_map = {"short": 0, "mid": 1, "long": 2}
        
        weighted_direction = sum(
            e["confidence"] * direction_map[e["direction"]]
            for e in relevant_events
        ) / len(relevant_events)
        
        avg_confidence = sum(e["confidence"] for e in relevant_events) / len(relevant_events)
        
        # Horizon-specific signals
        horizon_signals = {0: 0.0, 1: 0.0, 2: 0.0}
        for event in relevant_events:
            horizon_idx = horizon_map[event["horizon"]]
            signal = event["confidence"] * direction_map[event["direction"]]
            horizon_signals[horizon_idx] += signal
        
        # Normalize horizon signals
        horizon_counts = {0: 0, 1: 0, 2: 0}
        for event in relevant_events:
            horizon_counts[horizon_map[event["horizon"]]] += 1
        
        for idx in horizon_signals:
            if horizon_counts[idx] > 0:
                horizon_signals[idx] /= horizon_counts[idx]
        
        return {
            "event_count": len(relevant_events),
            "weighted_direction": weighted_direction,
            "avg_confidence": avg_confidence,
            "short_term_signal": horizon_signals[0],
            "mid_term_signal": horizon_signals[1],
            "long_term_signal": horizon_signals[2]
        }
