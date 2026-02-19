# Quick Start Guide

## Step 1: Check if You Have a Trained Agent

```bash
python check_agent.py
```

This will show you if you have any trained agents available.

## Step 2: Train an Agent (If Needed)

If you don't have a trained agent yet, train one:

```bash
python main.py --mode train --config configs/config.yaml
```

**Note**: Training can take a while depending on:
- Number of timesteps (default: 1,000,000)
- Your hardware (CPU vs GPU)
- LLM inference speed

**For faster testing**, you can modify `configs/config.yaml`:
- Reduce `total_timesteps` to 100000 (for quick test)
- Use smaller Qwen model: `Qwen/Qwen2.5-1.5B-Instruct` (faster inference)

## Step 3: Evaluate Your Agent

Once training completes, evaluate:

```bash
# Find your checkpoint
python check_agent.py

# Evaluate (use the checkpoint path shown)
python main.py --mode eval --checkpoint models/checkpoints/best_model --episodes 10
```

## Step 4: Analyze LLM Impact

Compare performance with vs without events:

```bash
python analyze_results.py \
    --agent models/checkpoints/best_model \
    --config configs/config.yaml \
    --episodes 10 \
    --compare
```

## Common Issues

### "Agent file not found"
- Make sure you've trained an agent first (Step 2)
- Check the exact path with `python check_agent.py`
- Use the full path shown by check_agent.py

### Training is slow
- Use smaller Qwen model in config
- Reduce total_timesteps for testing
- Use CPU instead of GPU if GPU is slow

### Out of memory
- Use smaller batch sizes in config
- Use smaller Qwen model
- Reduce number of symbols

## Next Steps

- See `QUICK_START_RESULTS.md` for interpreting results
- See `README.md` for full documentation
