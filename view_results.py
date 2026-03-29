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


def print_money_summary(results: dict) -> None:
    """Plain-language: capital in, profit/loss, ending balance split."""
    summaries = results.get("episode_summaries") or []
    if not summaries:
        # Fall back: first/last portfolio value only
        pv = results.get("portfolio_values") or []
        if len(pv) < 2:
            return
        initial = float(pv[0])
        final = float(pv[-1])
        pnl = final - initial
        pctp = 100.0 * pnl / initial if initial else 0.0
        print("\n" + "=" * 72)
        print("RESULTS — CAPITAL, PROFIT / LOSS, ENDING VALUE")
        print("=" * 72)
        print(f"  Starting capital:        ${initial:,.2f}")
        print(f"  Ending portfolio value:  ${final:,.2f}")
        sign = "+" if pnl >= 0 else ""
        print(f"  Profit or loss:          {sign}${pnl:,.2f}  ({sign}{pctp:.2f}%)")
        print("  (No per-asset balance in this file — re-run eval to save episode_summaries.)")
        print("=" * 72 + "\n")
        return

    label = results.get("label", "")
    n_ep = results.get("episodes", len(summaries))
    print("\n" + "=" * 72)
    print("RESULTS — HOW MUCH YOU PUT IN, PROFIT/LOSS, PORTFOLIO BALANCE")
    print("=" * 72)
    if label:
        print(f"  Label: {label}    |    Episodes evaluated: {n_ep}")
    for s in summaries:
        ep = s.get("episode", "?")
        ini = float(s.get("initial_cash", 0))
        fin = float(s.get("final_portfolio_value", 0))
        pnl = float(s.get("profit_dollars", fin - ini))
        pctp = 100.0 * float(s.get("profit_pct", (pnl / ini) if ini else 0))
        sign = "+" if pnl >= 0 else ""
        print(f"\n  Episode {ep}")
        print("  " + "-" * 66)
        print(f"  Starting capital (invested):     ${ini:,.2f}")
        print(f"  Portfolio value at end:          ${fin:,.2f}")
        print(f"  Profit or loss:                  {sign}${pnl:,.2f}   ({sign}{pctp:.2f}%)")
        print("  Balance at end (share of total portfolio):")
        if fin <= 1e-2:
            print("    — Portfolio is empty: $0 in cash and $0 in stocks.")
        else:
            alloc = s.get("allocation") or {}
            cw = float(s.get("cash_weight", 0))
            for sym, w in alloc.items():
                print(f"    {sym:<10}  {100.0 * float(w):6.2f}%  of portfolio")
            print(f"    {'Cash':<10}  {100.0 * cw:6.2f}%  of portfolio")
    print("\n" + "=" * 72 + "\n")


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
        results_path = Path(args.results)
        if not results_path.is_file():
            print(f"Error: file not found: {args.results}")
            print("(The name `evaluation_results_....json` in the docs is a placeholder — use a real file.)")
            rdir = Path("results")
            if rdir.is_dir():
                json_files = sorted(rdir.glob("*.json"))
                if json_files:
                    print("\nJSON files in results/:")
                    for p in json_files:
                        print(f"  python3 view_results.py --results {p}")
                else:
                    print("\nNo JSON files in results/ yet. Run an eval first, e.g.:")
                    print("  python3 -m src.agents.train --config configs/config.yaml --mode eval \\")
                    print("    --checkpoint models/checkpoints/final_model_dummy --extractor dummy --episodes 3")
            raise SystemExit(1)
        with open(results_path, "r") as f:
            results = json.load(f)

        print_money_summary(results)

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
