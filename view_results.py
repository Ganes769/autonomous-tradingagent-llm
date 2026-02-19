"""
Quick script to view and compare trading results.

Usage:
    python view_results.py --results results/evaluation_results_best_model.json
    python view_results.py --compare --with-events results/with_events.json --without-events results/baseline.json
"""

import argparse
import json
from pathlib import Path
from src.utils.visualize_results import ResultsVisualizer, compare_with_baseline

def main():
    parser = argparse.ArgumentParser(description="View trading agent results")
    parser.add_argument("--results", type=str, help="Path to results JSON file")
    parser.add_argument("--compare", action="store_true", help="Compare two results")
    parser.add_argument("--with-events", type=str, help="Results with events")
    parser.add_argument("--without-events", type=str, help="Results without events (baseline)")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer()
    
    if args.compare:
        # Compare two results
        if not args.with_events or not args.without_events:
            print("Error: --with-events and --without-events required for comparison")
            return
        
        with open(args.with_events, 'r') as f:
            results_with = json.load(f)
        with open(args.without_events, 'r') as f:
            results_without = json.load(f)
        
        # Convert date strings back to dates if needed
        if results_with.get('dates'):
            from datetime import datetime
            results_with['dates'] = [datetime.fromisoformat(d) if isinstance(d, str) else d 
                                    for d in results_with['dates']]
        if results_without.get('dates'):
            from datetime import datetime
            results_without['dates'] = [datetime.fromisoformat(d) if isinstance(d, str) else d 
                                       for d in results_without['dates']]
        
        visualizer.print_detailed_metrics(results_with.get('metrics', {}), "With Events")
        visualizer.print_detailed_metrics(results_without.get('metrics', {}), "Baseline")
        
        improvements = compare_with_baseline(results_with, results_without)
        
        save_path = "results/comparison.png" if args.save_plots else None
        visualizer.plot_comparison(
            results_with,
            results_without,
            save_path=save_path
        )
        
    elif args.results:
        # View single results file
        with open(args.results, 'r') as f:
            results = json.load(f)
        
        metrics = results.get('metrics', {})
        visualizer.print_detailed_metrics(metrics, results.get('label', 'Results'))
        
        portfolio_values = results.get('portfolio_values', [])
        dates = results.get('dates', [])
        
        if dates:
            from datetime import datetime
            dates = [datetime.fromisoformat(d) if isinstance(d, str) else d for d in dates]
        
        save_path = f"results/{Path(args.results).stem}_plot.png" if args.save_plots else None
        visualizer.plot_portfolio_performance(
            portfolio_values,
            dates if dates else None,
            title=results.get('label', 'Portfolio Performance'),
            save_path=save_path
        )
    else:
        print("Error: Either --results or --compare must be specified")
        print("\nUsage examples:")
        print("  python view_results.py --results results/evaluation_results_best_model.json")
        print("  python view_results.py --compare --with-events results/with_events.json --without-events results/baseline.json")

if __name__ == "__main__":
    main()
