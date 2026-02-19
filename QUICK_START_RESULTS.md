# Quick Start: Viewing Results and LLM Impact

This guide shows you how to see results and evaluate how your LLM event extraction improves trading performance.

## Step 1: Train Your Agent

First, train your agent (with event extraction):

```bash
python main.py --mode train --config configs/config.yaml
```

This will save checkpoints to `models/checkpoints/`.

## Step 2: Evaluate Performance

### Option A: Quick Evaluation (Single Run)

Evaluate your trained agent:

```bash
python main.py --mode eval --checkpoint models/checkpoints/best_model --episodes 10
```

This will:
- Print metrics (Sharpe, Sortino, Max Drawdown, etc.)
- Save results to `results/evaluation_results_best_model.json`

### Option B: Compare With vs Without Events

To see how much the LLM event extraction helps, compare performance:

```bash
python analyze_results.py \
    --agent models/checkpoints/best_model \
    --config configs/config.yaml \
    --episodes 10 \
    --compare \
    --output-dir results
```

This will:
1. Run evaluation WITH event extraction
2. Run evaluation WITHOUT event extraction (baseline)
3. Compare the two and show improvements
4. Generate comparison plots

## Step 3: View Results

### View Single Results File

```bash
python view_results.py --results results/evaluation_results_best_model.json
```

### View Comparison

```bash
python view_results.py \
    --compare \
    --with-events results/results_with_events.json \
    --without-events results/comparison_results.json
```

### Save Plots

Add `--save-plots` to save visualization images:

```bash
python view_results.py --results results/evaluation_results_best_model.json --save-plots
```

## Understanding the Results

### Key Metrics

1. **Sharpe Ratio**: Risk-adjusted returns
   - Higher = Better (more stable profits)
   - If Sharpe improves with events → LLM helps reduce volatility

2. **Sortino Ratio**: Downside risk focus
   - Higher = Better (avoids crashes)
   - If Sortino improves → LLM helps avoid bad drawdowns

3. **Max Drawdown**: Worst peak-to-trough decline
   - Lower = Better (safer)
   - If drawdown reduces → LLM helps agent hold cash during risky periods

4. **Total Return**: Overall profit
   - Higher = Better
   - If return improves → LLM helps identify profitable opportunities

### What Good Results Look Like

**With Event Extraction should show:**
- ✅ Higher Sharpe Ratio (more consistent profits)
- ✅ Higher Sortino Ratio (better downside protection)
- ✅ Lower Max Drawdown (safer during market stress)
- ✅ Better or similar Total Return

**Example of good improvement:**
```
EVENT EXTRACTION IMPACT ANALYSIS
============================================================
Sharpe Ratio Improvement:  +0.250  ← Good!
Sortino Ratio Improvement:  +0.180  ← Good!
Return Improvement:         +2.50%  ← Good!
Drawdown Reduction:         -3.20%  ← Excellent! (lower is better)
```

## Visualizations Generated

The analysis scripts create:

1. **Portfolio Performance Plot**: Shows portfolio value over time
2. **Comparison Plot**: Side-by-side comparison with baseline
3. **Metrics Comparison**: Bar charts showing metric improvements
4. **Drawdown Comparison**: Shows how much safer the agent is

## Files Generated

After running analysis, you'll find:

```
results/
├── evaluation_results_best_model.json    # Single evaluation results
├── results_with_events.json              # Results with LLM events
├── comparison_results.json                # Comparison data
├── performance_with_events.png           # Portfolio performance plot
└── comparison.png                        # Comparison visualization
```

## Tips for Better Results

1. **More Training**: Train for more timesteps (increase `total_timesteps` in config)
2. **Better News Data**: Use real news data instead of sample data
3. **Tune Reward Weights**: Adjust `event_alignment_weight` in config
4. **Try Different Models**: Use smaller/faster Qwen models for testing
5. **More Episodes**: Evaluate with more episodes for more reliable metrics

## Troubleshooting

**No improvement shown?**
- Check if news data is being loaded correctly
- Verify event extractor is working (check logs)
- Try increasing `event_alignment_weight` in config
- Train for longer (more timesteps)

**Can't see plots?**
- Make sure matplotlib backend is set correctly
- On Mac/Linux, may need: `export MPLBACKEND=TkAgg`
- Or use `--save-plots` to save images instead

**Baseline comparison fails?**
- Make sure agent was trained with events
- Check that news data is available for event evaluation

## Next Steps

1. **Experiment with different event types**: Modify event extraction prompts
2. **Add more news sources**: Integrate GDELT or news APIs
3. **Fine-tune reward function**: Adjust event alignment weights
4. **Backtest on historical data**: Test on different time periods
