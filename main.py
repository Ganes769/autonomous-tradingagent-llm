"""
Main Entry Point for Event-Based Trading Agent

Usage:
    python main.py --mode train
    python main.py --mode eval --checkpoint models/checkpoints/best_model
"""

import argparse
from src.agents.train import train_agent, evaluate_agent


def main():
    parser = argparse.ArgumentParser(
        description="Event-Based Trading Agent with PPO"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
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
        help="Path to checkpoint (for resume/eval)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("Starting training...")
        train_agent(
            config_path=args.config,
            news_data=None,  # Can load news data here
            resume_from=args.checkpoint
        )
    elif args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for evaluation mode")
        print("Starting evaluation...")
        evaluate_agent(
            agent_path=args.checkpoint,
            config_path=args.config,
            news_data=None,
            n_episodes=args.episodes
        )


if __name__ == "__main__":
    main()
