"""
Example usage script for the Event-Based Trading Agent

This script demonstrates how to:
1. Load configuration
2. Initialize components
3. Train an agent
4. Evaluate performance
"""

import yaml
from src.agents.train import train_agent, evaluate_agent
from src.utils.load_news import create_sample_news_data, load_news_from_jsonl

def main():
    """Example usage of the trading agent."""
    
    # Configuration path
    config_path = "configs/config.yaml"
    
    # Option 1: Use sample news data
    print("Using sample news data...")
    news_data = create_sample_news_data()
    
    # Option 2: Load news from JSONL file (if you have one)
    # news_data = load_news_from_jsonl("data/news/news.jsonl")
    
    # Option 3: No news data (agent will still work, just no event signals)
    # news_data = None
    
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)
    
    # Train the agent
    agent = train_agent(
        config_path=config_path,
        news_data=news_data,
        resume_from=None  # Set to checkpoint path to resume training
    )
    
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    # Evaluate the trained agent
    metrics = evaluate_agent(
        agent_path="models/checkpoints/best_model",
        config_path=config_path,
        news_data=news_data,
        n_episodes=5
    )
    
    print("\nEvaluation complete!")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
