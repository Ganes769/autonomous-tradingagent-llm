"""
Quick test script to verify the setup is working.

Run this to check if all components can be imported and initialized.
"""

import sys
import traceback

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import stable_baselines3
        print(f"✓ Stable-Baselines3")
    except ImportError as e:
        print(f"✗ Stable-Baselines3: {e}")
        return False
    
    try:
        import gymnasium
        print(f"✓ Gymnasium")
    except ImportError as e:
        print(f"✗ Gymnasium: {e}")
        return False
    
    try:
        import yfinance
        print(f"✓ yfinance")
    except ImportError as e:
        print(f"✗ yfinance: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False
    
    try:
        from src.data.market_data import MarketDataFetcher
        print("✓ MarketDataFetcher")
    except Exception as e:
        print(f"✗ MarketDataFetcher: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.models.event_extractor import EventExtractor
        print("✓ EventExtractor")
    except Exception as e:
        print(f"✗ EventExtractor: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.agents.trading_env import TradingEnv
        print("✓ TradingEnv")
    except Exception as e:
        print(f"✗ TradingEnv: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.utils.metrics import TradingMetrics
        print("✓ TradingMetrics")
    except Exception as e:
        print(f"✗ TradingMetrics: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_market_data():
    """Test market data fetching."""
    print("\nTesting market data fetching...")
    
    try:
        from src.data.market_data import MarketDataFetcher
        
        fetcher = MarketDataFetcher(
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        data = fetcher.fetch_data("AAPL")
        if not data.empty:
            print(f"✓ Successfully fetched {len(data)} rows of data for AAPL")
            print(f"  Columns: {list(data.columns[:5])}...")
            return True
        else:
            print("✗ No data fetched")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics calculation...")
    
    try:
        from src.utils.metrics import TradingMetrics
        
        metrics = TradingMetrics()
        
        # Test with sample data
        portfolio_values = [100000, 101000, 102500, 101500, 103000]
        returns = metrics._calculate_returns(portfolio_values)
        
        sharpe = metrics.sharpe_ratio(returns)
        sortino = metrics.sortino_ratio(returns)
        max_dd = metrics.max_drawdown(portfolio_values)
        total_return = metrics.total_return(portfolio_values)
        
        print(f"✓ Metrics calculated:")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Sortino Ratio: {sortino:.3f}")
        print(f"  Max Drawdown: {max_dd:.2%}")
        print(f"  Total Return: {total_return:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("EVENT-BASED TRADING AGENT - SETUP TEST")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test market data (requires internet)
    print("\nNote: Market data test requires internet connection...")
    try:
        if not test_market_data():
            print("  (Skipping - may need internet connection)")
    except:
        print("  (Skipping - may need internet connection)")
    
    # Test metrics
    if not test_metrics():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Setup looks good.")
        print("\nNext steps:")
        print("1. Review configs/config.yaml")
        print("2. Run: python main.py --mode train")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*60)


if __name__ == "__main__":
    main()
