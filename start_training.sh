#!/bin/bash
# Quick start training script

echo "=========================================="
echo "Starting Training for Event-Based Trading Agent"
echo "=========================================="
echo ""
echo "Configuration: Quick Test (faster training)"
echo "  - Timesteps: 50,000 (quick test)"
echo "  - Model: Qwen2.5-1.5B-Instruct (faster)"
echo "  - Symbols: AAPL, MSFT, GOOGL (3 stocks)"
echo ""
echo "For full training, use:"
echo "  python main.py --mode train --config configs/config.yaml"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

python main.py --mode train --config configs/config_quick_test.yaml
