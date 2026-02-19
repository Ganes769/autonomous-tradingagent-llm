"""
Analysis script to evaluate LLM event extraction impact on trading performance.

Compares:
- Trading with event extraction vs without
- Event quality metrics
- Performance attribution
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List
import logging

from src.utils.visualize_results import ResultsVisualizer, compare_with_baseline
from src.utils.metrics import TradingMetrics
from src.agents.train import evaluate_agent, create_env, load_config
from src.models.event_extractor import EventExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_baseline_evaluation(config_path: str, agent_path: str, n_episodes: int = 10) -> Dict:
    """
    Run evaluation WITHOUT event extraction (baseline).
    
    Args:
        config_path: Path to config file
        agent_path: Path to trained agent
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with results
    """
    from stable_baselines3 import PPO
    
    config = load_config(config_path)
    
    logger.info("Running BASELINE evaluation (no events)...")
    
    # Create environment WITHOUT event extractor (or with dummy one)
    from src.data.market_data import MarketDataFetcher
    
    market_fetcher = MarketDataFetcher(
        symbols=config['data']['symbols'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        frequency=config['data']['frequency']
    )
    
    # Create dummy event extractor that returns no events
    class DummyEventExtractor:
        def extract_events(self, text):
            return []
        def encode_event_features(self, events, ticker):
            return {
                "event_count": 0,
                "weighted_direction": 0.0,
                "avg_confidence": 0.0,
                "short_term_signal": 0.0,
                "mid_term_signal": 0.0,
                "long_term_signal": 0.0
            }
    
    from src.agents.trading_env import TradingEnv
    
    env = TradingEnv(
        symbols=config['data']['symbols'],
        market_data_fetcher=market_fetcher,
        event_extractor=DummyEventExtractor(),
        initial_cash=config['trading']['initial_cash'],
        transaction_cost=config['trading']['transaction_cost'],
        max_position_size=config['trading']['max_position_size'],
        lookback_window=config['trading']['lookback_window'],
        news_data={}  # No news data
    )
    
    # Check if agent file exists
    from pathlib import Path
    agent_file = Path(agent_path)
    if not agent_file.exists():
        # Try with .zip extension
        agent_file_zip = Path(f"{agent_path}.zip")
        if not agent_file_zip.exists():
            raise FileNotFoundError(
                f"\n❌ Agent file not found: {agent_path}\n\n"
                f"Please train an agent first using:\n"
                f"  python main.py --mode train --config {config_path}\n\n"
                f"Or check available agents with:\n"
                f"  python check_agent.py\n"
            )
        agent_path = str(agent_file_zip)
    
    agent = PPO.load(agent_path, env=env)
    metrics_calculator = TradingMetrics(risk_free_rate=config['evaluation']['risk_free_rate'])
    
    all_portfolio_values = []
    all_dates = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_values = [info['portfolio_value']]
        episode_dates = []
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_values.append(info['portfolio_value'])
            if env.current_step < len(env.dates):
                episode_dates.append(env.dates[env.current_step - 1])
        
        all_portfolio_values.extend(episode_values)
        all_dates.extend(episode_dates)
    
    metrics = metrics_calculator.calculate_all_metrics(all_portfolio_values, all_dates if all_dates else None)
    
    return {
        'portfolio_values': all_portfolio_values,
        'dates': all_dates,
        'metrics': metrics,
        'label': 'Baseline (No Events)'
    }


def run_event_evaluation(config_path: str, agent_path: str, news_data: Dict = None, n_episodes: int = 10) -> Dict:
    """
    Run evaluation WITH event extraction.
    
    Args:
        config_path: Path to config file
        agent_path: Path to trained agent
        news_data: News data dictionary
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with results
    """
    from stable_baselines3 import PPO
    
    config = load_config(config_path)
    
    logger.info("Running evaluation WITH event extraction...")
    
    # Create environment WITH event extraction
    from src.data.market_data import MarketDataFetcher
    from src.models.event_extractor import EventExtractor
    from src.agents.trading_env import TradingEnv
    
    market_fetcher = MarketDataFetcher(
        symbols=config['data']['symbols'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        frequency=config['data']['frequency']
    )
    
    event_extractor = EventExtractor(
        model_name=config['event_extraction']['model_name'],
        device=config['event_extraction']['device'],
        max_tokens=config['event_extraction']['max_tokens'],
        temperature=config['event_extraction']['temperature']
    )
    
    env = TradingEnv(
        symbols=config['data']['symbols'],
        market_data_fetcher=market_fetcher,
        event_extractor=event_extractor,
        initial_cash=config['trading']['initial_cash'],
        transaction_cost=config['trading']['transaction_cost'],
        max_position_size=config['trading']['max_position_size'],
        lookback_window=config['trading']['lookback_window'],
        news_data=news_data or {}
    )
    
    # Check if agent file exists
    from pathlib import Path
    agent_file = Path(agent_path)
    if not agent_file.exists():
        # Try with .zip extension
        agent_file_zip = Path(f"{agent_path}.zip")
        if not agent_file_zip.exists():
            raise FileNotFoundError(
                f"\n❌ Agent file not found: {agent_path}\n\n"
                f"Please train an agent first using:\n"
                f"  python main.py --mode train --config {config_path}\n\n"
                f"Or check available agents with:\n"
                f"  python check_agent.py\n"
            )
        agent_path = str(agent_file_zip)
    
    agent = PPO.load(agent_path, env=env)
    metrics_calculator = TradingMetrics(risk_free_rate=config['evaluation']['risk_free_rate'])
    
    all_portfolio_values = []
    all_dates = []
    event_stats = {}
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_values = [info['portfolio_value']]
        episode_dates = []
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_values.append(info['portfolio_value'])
            if env.current_step < len(env.dates):
                episode_dates.append(env.dates[env.current_step - 1])
        
        all_portfolio_values.extend(episode_values)
        all_dates.extend(episode_dates)
    
    metrics = metrics_calculator.calculate_all_metrics(all_portfolio_values, all_dates if all_dates else None)
    
    return {
        'portfolio_values': all_portfolio_values,
        'dates': all_dates,
        'metrics': metrics,
        'event_stats': event_stats,
        'label': 'With Event Extraction'
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM event extraction impact")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--agent", type=str, required=True, help="Path to trained agent")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--news-data", type=str, default=None, help="Path to news data JSONL file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline")
    
    args = parser.parse_args()
    
    # Load news data if provided
    news_data = None
    if args.news_data:
        from src.utils.load_news import load_news_from_jsonl
        news_data = load_news_from_jsonl(args.news_data)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run evaluation with events
    logger.info("Evaluating agent WITH event extraction...")
    results_with_events = run_event_evaluation(
        args.config, args.agent, news_data, args.episodes
    )
    
    # Visualize results
    visualizer = ResultsVisualizer()
    visualizer.print_detailed_metrics(results_with_events['metrics'], "With Events")
    visualizer.plot_portfolio_performance(
        results_with_events['portfolio_values'],
        results_with_events['dates'],
        title="Portfolio Performance - With Event Extraction",
        save_path=str(output_dir / "performance_with_events.png")
    )
    
    # Save results
    visualizer.save_results(
        results_with_events,
        str(output_dir / "results_with_events.json"),
        include_portfolio_values=True
    )
    
    # Compare with baseline if requested
    if args.compare:
        logger.info("Evaluating agent WITHOUT event extraction (baseline)...")
        results_without_events = run_baseline_evaluation(
            args.config, args.agent, args.episodes
        )
        
        visualizer.print_detailed_metrics(results_without_events['metrics'], "Baseline (No Events)")
        
        # Compare
        improvements = compare_with_baseline(results_with_events, results_without_events)
        
        # Plot comparison
        visualizer.plot_comparison(
            results_with_events,
            results_without_events,
            save_path=str(output_dir / "comparison.png")
        )
        
        # Save comparison results
        comparison_data = {
            'with_events': results_with_events['metrics'],
            'without_events': results_without_events['metrics'],
            'improvements': improvements
        }
        
        with open(output_dir / "comparison_results.json", 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {output_dir}/")
        logger.info("Key findings:")
        logger.info(f"  - Sharpe improvement: {improvements['sharpe_improvement']:+.3f}")
        logger.info(f"  - Return improvement: {improvements['return_improvement']:+.2%}")
        logger.info(f"  - Drawdown reduction: {improvements['drawdown_reduction']:+.2%}")


if __name__ == "__main__":
    main()
