"""
Training Script for PPO Trading Agent

Trains the agent using Stable-Baselines3 PPO algorithm.
"""

import os
import yaml
import argparse
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import torch
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.trading_env import TradingEnv
from src.data.market_data import MarketDataFetcher
from src.models.event_extractor import EventExtractor
from src.utils.metrics import TradingMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_env(config: Dict, news_data: Optional[Dict] = None) -> TradingEnv:
    """Create trading environment."""
    # Initialize market data fetcher
    market_fetcher = MarketDataFetcher(
        symbols=config['data']['symbols'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        frequency=config['data']['frequency']
    )
    
    # Initialize event extractor
    event_extractor = EventExtractor(
        model_name=config['event_extraction']['model_name'],
        device=config['event_extraction']['device'],
        max_tokens=config['event_extraction']['max_tokens'],
        temperature=config['event_extraction']['temperature']
    )
    
    # Create environment
    env = TradingEnv(
        symbols=config['data']['symbols'],
        market_data_fetcher=market_fetcher,
        event_extractor=event_extractor,
        initial_cash=config['trading']['initial_cash'],
        transaction_cost=config['trading']['transaction_cost'],
        max_position_size=config['trading']['max_position_size'],
        lookback_window=config['trading']['lookback_window'],
        news_data=news_data
    )
    
    return env


def train_agent(
    config_path: str,
    news_data: Optional[Dict] = None,
    resume_from: Optional[str] = None
):
    """
    Train PPO agent.
    
    Args:
        config_path: Path to config YAML file
        news_data: Optional news data dictionary
        resume_from: Optional path to checkpoint to resume from
    """
    # Load configuration
    config = load_config(config_path)
    
    logger.info("Creating training environment...")
    train_env = create_env(config, news_data)
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])
    
    logger.info("Creating evaluation environment...")
    eval_env = create_env(config, news_data)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create directories
    log_dir = config['training']['log_dir']
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize or load PPO agent
    if resume_from:
        logger.info(f"Loading agent from {resume_from}")
        agent = PPO.load(resume_from, env=train_env)
    else:
        logger.info("Creating new PPO agent...")
        agent = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config['ppo']['learning_rate'],
            n_steps=config['ppo']['n_steps'],
            batch_size=config['ppo']['batch_size'],
            n_epochs=config['ppo']['n_epochs'],
            gamma=config['ppo']['gamma'],
            gae_lambda=config['ppo']['gae_lambda'],
            clip_range=config['ppo']['clip_range'],
            ent_coef=config['ppo']['ent_coef'],
            vf_coef=config['ppo']['vf_coef'],
            max_grad_norm=config['ppo']['max_grad_norm'],
            verbose=1,
            tensorboard_log=log_dir
        )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=config['training']['eval_frequency'],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_frequency'],
        save_path=checkpoint_dir,
        name_prefix='ppo_trading'
    )
    
    # Train agent
    logger.info(f"Starting training for {config['training']['total_timesteps']} steps...")
    agent.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model')
    agent.save(final_model_path)
    logger.info(f"Training complete! Model saved to {final_model_path}")
    
    return agent


def evaluate_agent(
    agent_path: str,
    config_path: str,
    news_data: Optional[Dict] = None,
    n_episodes: int = 10
):
    """
    Evaluate trained agent.
    
    Args:
        agent_path: Path to saved agent
        config_path: Path to config YAML file
        news_data: Optional news data dictionary
        n_episodes: Number of episodes to evaluate
    """
    from stable_baselines3 import PPO
    
    config = load_config(config_path)
    
    logger.info("Creating evaluation environment...")
    env = create_env(config, news_data)
    
    logger.info(f"Loading agent from {agent_path}")
    agent = PPO.load(agent_path, env=env)
    
    metrics_calculator = TradingMetrics(
        risk_free_rate=config['evaluation']['risk_free_rate']
    )
    
    all_portfolio_values = []
    all_dates = []
    
    logger.info(f"Running {n_episodes} evaluation episodes...")
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_values = [info['portfolio_value']]
        episode_dates = []
        
        step = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_values.append(info['portfolio_value'])
            if env.current_step < len(env.dates):
                episode_dates.append(env.dates[env.current_step - 1])
            
            step += 1
        
        all_portfolio_values.extend(episode_values)
        all_dates.extend(episode_dates)
        
        logger.info(f"Episode {episode + 1}: Final portfolio value = ${info['portfolio_value']:.2f}")
    
    # Calculate metrics
    metrics = metrics_calculator.calculate_all_metrics(
        all_portfolio_values,
        all_dates if all_dates else None
    )
    
    metrics_calculator.print_metrics(metrics)
    
    # Save detailed results
    import os
    from pathlib import Path
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_data = {
        'portfolio_values': all_portfolio_values,
        'dates': [str(d) for d in all_dates] if all_dates else None,
        'metrics': metrics,
        'episodes': n_episodes
    }
    
    import json
    results_file = results_dir / f"evaluation_results_{Path(agent_path).stem}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Trading Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Mode: train or eval"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for resume/eval"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_agent(args.config, resume_from=args.checkpoint)
    elif args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for evaluation mode")
        evaluate_agent(args.checkpoint, args.config, n_episodes=args.episodes)
