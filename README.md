# Event-Based Trading Agent with PPO

An autonomous trading agent that uses structured news events (not just sentiment) combined with price data, learning optimal trading actions using Proximal Policy Optimization (PPO) deep reinforcement learning.

## Overview

This system implements a complete pipeline:

1. **News → Event Extraction**: Raw news articles are parsed using Qwen LLM to extract structured events
2. **Event Analysis**: Each event is analyzed for short-, mid-, and long-term market impact
3. **PPO Agent**: A reinforcement learning agent observes market data + event features and selects actions (buy/sell/hold/portfolio weights)
4. **Reward Function**: Combines realized profit with event-alignment bonus to encourage correct reactions to events

## Key Features

- **Event-Based Signals**: Focuses on concrete events (layoffs, product launches, earnings, lawsuits) rather than sentiment
- **Multi-Time Horizon**: Events classified as short/mid/long-term to prevent wrong-timing reactions
- **PPO Reinforcement Learning**: Stable policy gradient algorithm for learning trading strategies
- **Event-Alignment Reward**: Agent gets bonus for reacting correctly to identified events
- **Comprehensive Metrics**: Sharpe ratio, Sortino ratio, Maximum Drawdown for robust evaluation

## Project Structure

```
event-llm/
├── configs/
│   └── config.yaml          # Configuration file
├── data/
│   ├── events/              # Event extraction training data
│   ├── market/              # Market data cache
│   └── processed/           # Processed datasets
├── models/
│   ├── checkpoints/         # Saved model checkpoints
│   └── logs/                # Training logs
├── src/
│   ├── agents/
│   │   ├── trading_env.py   # Gymnasium trading environment
│   │   └── train.py         # Training script
│   ├── data/
│   │   └── market_data.py   # Market data fetcher (Yahoo Finance, Stooq support)
│   ├── models/
│   │   └── event_extractor.py  # Qwen LLM event extraction
│   └── utils/
│       ├── metrics.py       # Evaluation metrics
│       └── load_news.py     # News data loading utilities
├── main.py                  # Main entry point
├── example_usage.py         # Example usage script
├── test_setup.py            # Setup verification script
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

1. **Clone the repository** (if applicable) or navigate to project directory

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download Qwen model** (first run will download automatically):
   - The model will be downloaded from HuggingFace on first use
   - For faster inference, consider using smaller models like `Qwen/Qwen2.5-1.5B-Instruct`

## Configuration

Edit `configs/config.yaml` to customize:

- **Data sources**: Stock symbols, date ranges, data frequency
- **Event extraction**: Qwen model name, device (cuda/cpu/mps)
- **PPO hyperparameters**: Learning rate, batch size, etc.
- **Trading parameters**: Initial cash, transaction costs, position limits
- **Reward weights**: Profit vs event-alignment vs risk penalties

## Usage

### Training

Train a new agent:
```bash
python main.py --mode train --config configs/config.yaml
```

Resume training from checkpoint:
```bash
python main.py --mode train --config configs/config.yaml --checkpoint models/checkpoints/ppo_trading_50000_steps
```

### Evaluation

Evaluate a trained agent:
```bash
python main.py --mode eval --checkpoint models/checkpoints/best_model --episodes 10
```

### Direct Training Script

You can also use the training script directly:
```bash
python -m src.agents.train --config configs/config.yaml --mode train
python -m src.agents.train --config configs/config.yaml --mode eval --checkpoint models/checkpoints/best_model
```

### Analyzing Results and LLM Impact

To see how your LLM event extraction improves trading performance:

**Compare with vs without events:**
```bash
python analyze_results.py \
    --agent models/checkpoints/best_model \
    --config configs/config.yaml \
    --episodes 10 \
    --compare \
    --output-dir results
```

**View saved results:**
```bash
python view_results.py --results results/evaluation_results_best_model.json
```

**View comparison:**
```bash
python view_results.py \
    --compare \
    --with-events results/results_with_events.json \
    --without-events results/comparison_results.json \
    --save-plots
```

See `QUICK_START_RESULTS.md` for detailed guide on interpreting results.

## Data Sources

The system supports multiple data sources:

1. **Yahoo Finance** (default): Free historical stock data via `yfinance` library
2. **Stooq**: Free historical market data from https://stooq.com/db/h/ (download manually and load into system)
3. **GDELT Project**: For news events (can be integrated) - https://www.gdeltproject.org/
4. **Kaggle**: Stock market datasets available on Kaggle

### Adding News Data

To add news data for event extraction, you can:

**Option 1: Use the utility function**
```python
from src.utils.load_news import load_news_from_jsonl, create_sample_news_data

# Load from JSONL file
news_data = load_news_from_jsonl("data/news/news.jsonl")

# Or use sample data for testing
news_data = create_sample_news_data()

train_agent(config_path, news_data=news_data)
```

**Option 2: Manual dictionary**
```python
news_data = {
    "2024-01-15": [
        "Apple announced record iPhone sales...",
        "Microsoft reports strong cloud growth..."
    ],
    # ... more dates
}

train_agent(config_path, news_data=news_data)
```

**JSONL Format** (for `load_news_from_jsonl`):
```json
{"date": "2024-01-15", "text": "Apple announced record iPhone sales..."}
{"date": "2024-01-16", "text": "Microsoft reports strong cloud growth..."}
```

## Evaluation Metrics

The system calculates:

1. **Sharpe Ratio**: Measures risk-adjusted returns (higher = better, more stable profits)
2. **Sortino Ratio**: Focuses on downside risk (higher = better, avoids crashes)
3. **Maximum Drawdown**: Worst peak-to-trough decline (lower = better, safer)
4. **Total Return**: Overall portfolio performance

These metrics are printed after evaluation and logged during training.

## How It Works

### Event Extraction

The Qwen LLM extracts structured events from news:

```json
{
  "events": [{
    "event_type": "layoff",
    "target": {"ticker": "MSFT", "company": "Microsoft", "sector": "Technology"},
    "direction": "down",
    "confidence": 0.83,
    "horizon": "short",
    "rationale": "Layoffs indicate near-term restructuring risk."
  }]
}
```

### Trading Environment

The environment provides observations combining:
- **Market features**: OHLCV data + technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Event features**: Event count, weighted direction, confidence, horizon-specific signals
- **Portfolio state**: Current holdings, cash position

### Reward Function

```
Total Reward = Trading Profit 
             - Transaction Costs 
             - Risk Penalty 
             + Event-Alignment Bonus
```

The event-alignment bonus encourages the agent to:
- Increase exposure when positive events occur
- Reduce exposure (hold cash) when negative events occur
- Be cautious when events are uncertain

## Performance Considerations

- **GPU**: Use CUDA for faster LLM inference (set `device: cuda` in config)
- **Mac M1/M2**: Use MPS backend (`device: mps`)
- **Smaller Models**: For faster training, use smaller Qwen models like `Qwen2.5-1.5B-Instruct`
- **Batch Processing**: Event extraction can be batched for efficiency

## Research Claims

This system demonstrates:

1. **Event-based signals improve over sentiment**: More objective, less noise
2. **Multi-horizon prevents wrong reactions**: Not all news matters immediately
3. **Reward shaping speeds learning**: Agent gets feedback even when profit is noisy
4. **Adaptability**: Agent can hold cash when news risk is high (proven by lower max drawdown)

## Future Enhancements

- [ ] Real-time news feed integration (GDELT, news APIs)
- [ ] Multi-asset portfolio optimization
- [ ] Risk management modules (stop-loss, position sizing)
- [ ] Backtesting framework with historical news
- [ ] Ensemble of multiple agents
- [ ] Online learning (continuous adaptation)

## License

[Add your license here]

## Citation

If you use this code in research, please cite:

```bibtex
@software{event_trading_agent,
  title={Event-Based Trading Agent with PPO},
  author={Your Name},
  year={2024}
}
```

## Contact

[Add contact information]
