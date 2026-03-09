"""
Analysis script: evaluate event-based vs sentiment-only vs market-only baselines.

Usage:
  python analyze_results.py --agent models/checkpoints/best_model --compare
  python analyze_results.py --agent models/checkpoints/best_model --news-data data/news.jsonl

Produces:
  results/results_event.json
  results/results_sentiment.json
  results/comparison_event_vs_sentiment.json
  results/comparison.png
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging

from src.utils.visualize_results import ResultsVisualizer, compare_with_baseline
from src.utils.metrics import TradingMetrics
from src.agents.train import (
    evaluate_agent,
    compare_event_vs_sentiment,
    load_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze event-based vs sentiment-only trading performance"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--agent", required=True, help="Path to trained agent checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--news-data", default=None, help="Path to news JSONL file")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare event-based vs sentiment-only side by side"
    )

    args = parser.parse_args()

    news_data = None
    if args.news_data:
        from src.utils.load_news import load_news_from_jsonl
        news_data = load_news_from_jsonl(args.news_data)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    visualizer = ResultsVisualizer()

    # ---- Evaluate with event extraction ----
    logger.info("Evaluating agent WITH event extraction...")
    results_event = evaluate_agent(
        args.agent, args.config, news_data, args.episodes,
        extractor_type="event", label="Event-Based (LLM)"
    )

    visualizer.print_detailed_metrics(results_event["metrics"], "Event-Based")
    visualizer.plot_portfolio_performance(
        results_event["portfolio_values"],
        results_event.get("dates"),
        title="Portfolio — Event-Based (LLM)",
        save_path=str(output_dir / "performance_event.png"),
    )
    visualizer.save_results(
        results_event,
        str(output_dir / "results_event.json"),
        include_portfolio_values=True,
    )

    if args.compare:
        # ---- Evaluate with sentiment-only ----
        logger.info("Evaluating agent WITHOUT LLM (sentiment-only baseline)...")
        results_sentiment = evaluate_agent(
            args.agent, args.config, news_data, args.episodes,
            extractor_type="sentiment", label="Sentiment-Only (Lexicon)"
        )

        visualizer.print_detailed_metrics(results_sentiment["metrics"], "Sentiment-Only")

        # Visualise comparison
        visualizer.plot_comparison(
            results_event,
            results_sentiment,
            save_path=str(output_dir / "comparison.png"),
        )
        visualizer.save_results(
            results_sentiment,
            str(output_dir / "results_sentiment.json"),
            include_portfolio_values=True,
        )

        # Textual comparison summary
        improvements = compare_with_baseline(results_event, results_sentiment)

        # Also run the structured head-to-head
        comparison = compare_event_vs_sentiment(
            args.agent, args.config, news_data, args.episodes
        )

        logger.info(f"\nAll results saved to {output_dir}/")
        logger.info(f"  Sharpe delta:  {improvements['sharpe_improvement']:+.3f}")
        logger.info(f"  Return delta:  {improvements['return_improvement']:+.2%}")
        logger.info(f"  DD reduction:  {improvements['drawdown_reduction']:+.2%}")

        if comparison["event_wins"]:
            logger.info("✅  Event-based extraction is beneficial — keep the LLM pipeline.")
        else:
            logger.info("⚠️  Sentiment baseline performs comparably — consider simplifying.")


if __name__ == "__main__":
    main()
