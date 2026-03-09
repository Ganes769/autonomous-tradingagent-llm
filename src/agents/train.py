"""
Training Script for PPO Trading Agent

Supports three modes:
  train   — train a new PPO agent (event-based or sentiment-only)
  eval    — evaluate a saved agent against risk-adjusted metrics
  compare — run both event-based and sentiment-only evaluations and print
            a side-by-side comparison table

The comparison mode is the core experiment that answers whether structured
event understanding outperforms raw sentiment signals.
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.trading_env import TradingEnv
from src.data.market_data import MarketDataFetcher
from src.models.event_extractor import EventExtractor
from src.models.sentiment_extractor import SentimentExtractor
from src.models.horizon_interpreter import HorizonInterpreter
from src.utils.metrics import TradingMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _build_market_fetcher(config: Dict) -> MarketDataFetcher:
    return MarketDataFetcher(
        symbols=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        frequency=config["data"]["frequency"],
    )


def _build_event_extractor(config: Dict) -> EventExtractor:
    ecfg = config["event_extraction"]
    return EventExtractor(
        model_name=ecfg["model_name"],
        device=ecfg["device"],
        max_tokens=ecfg["max_tokens"],
        temperature=ecfg["temperature"],
    )


def _build_sentiment_extractor(config: Dict) -> SentimentExtractor:
    """Build a sentiment-only baseline extractor with ticker aliases from config."""
    symbols = config["data"]["symbols"]
    # Simple default aliases: lower-case ticker + common name fragments
    _KNOWN_ALIASES = {
        "AAPL":  ["apple", "aapl"],
        "MSFT":  ["microsoft", "msft"],
        "GOOGL": ["google", "googl", "alphabet"],
        "AMZN":  ["amazon", "amzn"],
        "NVDA":  ["nvidia", "nvda"],
        "META":  ["meta", "facebook"],
        "TSLA":  ["tesla", "tsla"],
    }
    aliases = {s: _KNOWN_ALIASES.get(s, [s.lower()]) for s in symbols}
    return SentimentExtractor(ticker_aliases=aliases)


def create_env(
    config: Dict,
    news_data: Optional[Dict] = None,
    extractor_type: str = "event",      # "event" | "sentiment" | "dummy"
) -> TradingEnv:
    """
    Create a TradingEnv with the requested extractor type.

    extractor_type:
      "event"     — full Qwen-based EventExtractor
      "sentiment" — lexicon SentimentExtractor (no LLM required)
      "dummy"     — zero-feature extractor (market-only baseline)
    """
    market_fetcher = _build_market_fetcher(config)
    rcfg = config.get("reward", {})

    if extractor_type == "event":
        extractor = _build_event_extractor(config)
    elif extractor_type == "sentiment":
        extractor = _build_sentiment_extractor(config)
    else:
        extractor = _DummyExtractor()

    horizon = HorizonInterpreter(symbols=config["data"]["symbols"])

    env = TradingEnv(
        symbols=config["data"]["symbols"],
        market_data_fetcher=market_fetcher,
        event_extractor=extractor,
        initial_cash=config["trading"]["initial_cash"],
        transaction_cost=config["trading"]["transaction_cost"],
        max_position_size=config["trading"]["max_position_size"],
        lookback_window=config["trading"]["lookback_window"],
        news_data=news_data or {},
        profit_weight=rcfg.get("profit_weight", 1.0),
        event_alignment_weight=rcfg.get("event_alignment_weight", 0.3),
        risk_penalty_weight=rcfg.get("risk_penalty_weight", 0.1),
        transaction_cost_penalty=rcfg.get("transaction_cost_penalty", 0.5),
        horizon_interpreter=horizon,
    )
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_agent(
    config_path: str,
    news_data: Optional[Dict] = None,
    resume_from: Optional[str] = None,
    extractor_type: str = "event",
):
    config = load_config(config_path)

    logger.info(f"[train] extractor={extractor_type}")
    logger.info("Creating training environment...")
    train_env = Monitor(create_env(config, news_data, extractor_type))
    train_env = DummyVecEnv([lambda: train_env])

    logger.info("Creating evaluation environment...")
    eval_env = Monitor(create_env(config, news_data, extractor_type))
    eval_env = DummyVecEnv([lambda: eval_env])

    log_dir = config["training"]["log_dir"]
    ckpt_dir = config["training"]["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if resume_from:
        logger.info(f"Resuming from {resume_from}")
        agent = PPO.load(resume_from, env=train_env)
    else:
        ppocfg = config["ppo"]
        agent = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=float(ppocfg["learning_rate"]),
            n_steps=ppocfg["n_steps"],
            batch_size=ppocfg["batch_size"],
            n_epochs=ppocfg["n_epochs"],
            gamma=ppocfg["gamma"],
            gae_lambda=ppocfg["gae_lambda"],
            clip_range=ppocfg["clip_range"],
            ent_coef=ppocfg["ent_coef"],
            vf_coef=ppocfg["vf_coef"],
            max_grad_norm=ppocfg["max_grad_norm"],
            verbose=1,
            tensorboard_log=None,  # set to log_dir if tensorboard is installed
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=ckpt_dir,
        log_path=log_dir,
        eval_freq=config["training"]["eval_frequency"],
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=config["training"]["save_frequency"],
        save_path=ckpt_dir,
        name_prefix=f"ppo_trading_{extractor_type}",
    )

    logger.info(f"Training for {config['training']['total_timesteps']} steps...")
    agent.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=[eval_cb, ckpt_cb],
        progress_bar=True,
    )

    final_path = os.path.join(ckpt_dir, f"final_model_{extractor_type}")
    agent.save(final_path)
    logger.info(f"Training complete. Model saved → {final_path}")
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent_path: str,
    config_path: str,
    news_data: Optional[Dict] = None,
    n_episodes: int = 10,
    extractor_type: str = "event",
    label: Optional[str] = None,
) -> Dict:
    """
    Evaluate a saved PPO agent and return a results dict with metrics.
    """
    config = load_config(config_path)
    env = create_env(config, news_data, extractor_type)

    agent_file = Path(agent_path)
    if not agent_file.exists():
        agent_file_zip = Path(f"{agent_path}.zip")
        if not agent_file_zip.exists():
            raise FileNotFoundError(
                f"Agent not found: {agent_path}\n"
                "Train first: python main.py --mode train"
            )
        agent_path = str(agent_file_zip)

    agent = PPO.load(agent_path, env=env)
    metrics_calc = TradingMetrics(risk_free_rate=config["evaluation"]["risk_free_rate"])

    all_values: List[float] = []
    all_dates: List = []
    all_labels: List[str] = []
    episode_rewards: List[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_values = [info["portfolio_value"]]
        ep_dates = []
        ep_reward = 0.0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_values.append(info["portfolio_value"])
            ep_reward += reward
            if env.current_step < len(env.dates):
                ep_dates.append(env.dates[env.current_step - 1])

        all_values.extend(ep_values)
        all_dates.extend(ep_dates)
        all_labels.extend(env.action_labels)
        episode_rewards.append(ep_reward)

        logger.info(
            f"Episode {ep + 1}/{n_episodes}: "
            f"final=${info['portfolio_value']:.2f}, reward={ep_reward:.4f}"
        )

    metrics = metrics_calc.calculate_all_metrics(
        all_values, all_dates or None, all_labels
    )
    conv = TradingMetrics.compute_convergence_stats(episode_rewards)
    metrics.update(conv)

    metrics_calc.print_metrics(metrics, label=label or extractor_type)

    results = {
        "portfolio_values": all_values,
        "dates": [str(d) for d in all_dates],
        "metrics": metrics,
        "episodes": n_episodes,
        "label": label or extractor_type,
        "episode_rewards": episode_rewards,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    stem = Path(agent_path).stem
    out_file = results_dir / f"evaluation_results_{stem}_{extractor_type}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {out_file}")

    return results


# ---------------------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------------------

def compare_event_vs_sentiment(
    agent_path: str,
    config_path: str,
    news_data: Optional[Dict] = None,
    n_episodes: int = 10,
) -> Dict:
    """
    Run the same agent in both event-based and sentiment-only environments
    and print a side-by-side comparison.

    If the event-based model does NOT outperform sentiment on at least
    Sharpe and Sortino, the comparison table highlights this, and the
    function returns a flag `event_wins = False` to signal that the
    operator should consider disabling the LLM extractor.
    """
    logger.info("=" * 60)
    logger.info("COMPARISON: Event-Based vs Sentiment-Only")
    logger.info("=" * 60)

    event_results = evaluate_agent(
        agent_path, config_path, news_data, n_episodes,
        extractor_type="event", label="Event-Based (LLM)"
    )
    sentiment_results = evaluate_agent(
        agent_path, config_path, news_data, n_episodes,
        extractor_type="sentiment", label="Sentiment-Only (Lexicon)"
    )

    em = event_results["metrics"]
    sm = sentiment_results["metrics"]
    deltas = TradingMetrics.compare_metrics(em, sm)

    print("\n" + "=" * 70)
    print("COMPARISON TABLE  (positive delta = Event-Based is better)")
    print("=" * 70)
    print(f"{'Metric':<30} {'Event-Based':>14} {'Sentiment':>14} {'Delta':>10}")
    print("-" * 70)
    key_metrics = [
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("calmar_ratio", "Calmar Ratio"),
        ("total_return", "Total Return"),
        ("max_drawdown", "Max Drawdown"),
        ("volatility", "Volatility"),
        ("win_rate", "Win Rate"),
        ("profit_factor", "Profit Factor"),
    ]
    for key, label in key_metrics:
        ev = em.get(key, 0.0)
        sv = sm.get(key, 0.0)
        delta = deltas.get(f"{key}_delta", 0.0)
        fmt_pct = key in {"total_return", "max_drawdown", "volatility", "win_rate"}
        if fmt_pct:
            print(f"  {label:<28} {ev:>13.2%} {sv:>13.2%} {delta:>+10.2%}")
        else:
            print(f"  {label:<28} {ev:>13.3f} {sv:>13.3f} {delta:>+10.3f}")
    print("=" * 70)

    sharpe_wins = deltas.get("sharpe_ratio_delta", 0.0) > 0
    sortino_wins = deltas.get("sortino_ratio_delta", 0.0) > 0
    event_wins = sharpe_wins and sortino_wins

    if event_wins:
        print("\n✅  Event-Based model OUTPERFORMS sentiment baseline on Sharpe & Sortino.")
        print("   Structured event understanding adds value.\n")
    else:
        print("\n⚠️  Event-Based model does NOT clearly outperform sentiment baseline.")
        print("   Consider using SentimentExtractor for lower-cost inference,")
        print("   or collect more news data to improve LLM signal quality.\n")

    comparison = {
        "event_metrics": em,
        "sentiment_metrics": sm,
        "deltas": deltas,
        "event_wins": event_wins,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "comparison_event_vs_sentiment.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Comparison saved → results/comparison_event_vs_sentiment.json")

    return comparison


# ---------------------------------------------------------------------------
# Dummy extractor for market-only baseline
# ---------------------------------------------------------------------------

class _DummyExtractor:
    FEATURE_DIM = 10

    def extract_events(self, text):
        return []

    def encode_event_features(self, events, ticker):
        return {k: 0.0 for k in [
            "event_count", "weighted_direction", "avg_confidence",
            "short_term_signal", "mid_term_signal", "long_term_signal",
            "magnitude", "sector_contagion", "urgency", "event_diversity",
        ]}

    def encode_event_feature_vector(self, events, ticker):
        return [0.0] * 10


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Trading Agent — train / eval / compare")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--mode", choices=["train", "eval", "compare"], default="train"
    )
    parser.add_argument("--checkpoint", default=None, help="Agent path for eval/compare/resume")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--extractor", choices=["event", "sentiment", "dummy"], default="event",
        help="Extractor type for train/eval modes"
    )
    parser.add_argument("--news-data", default=None, help="Path to news JSONL file")

    args = parser.parse_args()

    news_data = None
    if args.news_data:
        from src.utils.load_news import load_news_from_jsonl
        news_data = load_news_from_jsonl(args.news_data)

    if args.mode == "train":
        train_agent(
            args.config,
            news_data=news_data,
            resume_from=args.checkpoint,
            extractor_type=args.extractor,
        )
    elif args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for eval mode")
        evaluate_agent(
            args.checkpoint,
            args.config,
            news_data=news_data,
            n_episodes=args.episodes,
            extractor_type=args.extractor,
        )
    elif args.mode == "compare":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for compare mode")
        compare_event_vs_sentiment(
            args.checkpoint,
            args.config,
            news_data=news_data,
            n_episodes=args.episodes,
        )
