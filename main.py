# -*- coding: utf-8 -*-
"""
Jetski FPL Quantitative Engine CLI Application (main.py)
"""

import sys
import os
import argparse
import unittest
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import sync_fpl_api_data, load_player_history, fetch_clubelo_ratings, fetch_live_sharp_odds
from rate_engine import CanonicalRateEngine
from ml_rate_estimator import ComponentRateMLEstimator
from optimizer import MultiPeriodMILP
from chip_evaluator import ChipEvaluator
from backtester import WalkForwardBacktestHarness

def run_tests():
    """Runs all unittests in tests/ directory."""
    print("🧪 Running Unit Test Suite...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def main():
    parser = argparse.ArgumentParser(description="Jetski FPL Quantitative Engine")
    parser.add_argument("--sync", action="store_true", help="Sync live FPL API data")
    parser.add_argument("--optimize", action="store_true", help="Run 6-GW rolling MILP optimization")
    parser.add_argument("--use-ml-rates", action="store_true", help="Use LightGBM component rate estimator")
    parser.add_argument("--use-betting-odds", action="store_true", help="Use sharp betting odds for match engine")
    parser.add_argument("--chips", action="store_true", help="Evaluate chip reservation hurdle curves")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward simulation")
    parser.add_argument("--test", action="store_true", help="Run unittest suite")
    parser.add_argument("--start-gw", type=int, default=2, help="Start GW for backtest")
    parser.add_argument("--end-gw", type=int, default=10, help="End GW for backtest")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.test:
        success = run_tests()
        if not success:
            sys.exit(1)
        if len(sys.argv) == 1:
            print("\nDefault test run complete. Pass --help to see execution flags.")
        return

    if args.sync:
        print("📡 Syncing official FPL API data...")
        sync_fpl_api_data()

    history_df = load_player_history()
    if history_df.empty:
        print("⚠️ Warning: Historical player data empty. Run data ingestion first.")
        old_csv = "/Users/anshulkapoor/Documents/Coding-Python/fpl-scripts/fpl_all_player_data.csv"
        if os.path.exists(old_csv):
            print(f"   -> Copying historical dataset from {old_csv}...")
            target_csv = os.path.join(os.path.dirname(__file__), "data", "fpl_all_player_data.csv")
            df_old = pd.read_csv(old_csv)
            df_old.to_csv(target_csv, index=False)
            history_df = load_player_history()

    if args.optimize or args.chips:
        print("🧠 Building Canonical Rate Model & Horizon xP Matrix...")
        
        ml_rates_df = pd.DataFrame()
        if args.use_ml_rates:
            print("🤖 Training LightGBM Component Rate Estimator (npxG90, xA90, P(Start))...")
            ml_est = ComponentRateMLEstimator()
            if ml_est.train(history_df):
                ml_rates_df = ml_est.predict_rates(history_df)
                print(f"   -> Generated ML rate predictions for {len(ml_rates_df)} active players.")

        engine = CanonicalRateEngine(history_df, ml_rates_df=ml_rates_df)
        elo_dict = fetch_clubelo_ratings()
        
        sharp_odds_df = None
        if getattr(args, 'use_betting_odds', False):
            print("🎲 Ingesting Sharp Betting Odds (2-day TTL cache)...")
            sharp_odds_df = fetch_live_sharp_odds()


        start_gw = int(history_df["gameweek"].max()) + 1 if not history_df.empty and "gameweek" in history_df.columns else 1
        matrix = engine.generate_horizon_matrix(start_gw=start_gw, horizon_weeks=6, elo_dict=elo_dict, sharp_odds_df=sharp_odds_df)


        if matrix.empty:
            print("❌ Error: Matrix generation returned empty set.")
            return

        print(f"✅ Generated xP matrix for {len(matrix)} players across 6-GW horizon.")

        # Default initial squad sample (Top 15 available players by xP for demonstration)
        top_15 = matrix.sort_values("xP_horizon_sum", ascending=False)["player_id"].head(15).tolist()
        
        optimizer = MultiPeriodMILP(matrix)
        
        if args.optimize:
            print("\n📋 SOLVING 6-GW ROLLING MILP OPTIMIZATION...")
            plan = optimizer.solve_rolling_horizon(top_15, initial_bank=1.0, initial_fts=1)
            print(f"Status: {plan.get('status')}")
            print(f"Projected GW1 Points: {plan.get('projected_points_gw1')}")
            print(f"Bank Remaining: £{plan.get('bank')}m")
            print(f"Transfers Out: {plan.get('transfers_out')}")
            print(f"Transfers In: {plan.get('transfers_in')}")

        if args.chips:
            print("\n🃏 EVALUATING CHIP RESERVATION HURDLE CURVES...")
            chip_eval = optimizer.evaluate_chip_deltas(top_15, initial_bank=1.0, initial_fts=1)
            evaluator = ChipEvaluator()
            report = evaluator.evaluate(chip_eval["standard_plan"], chip_eval["wildcard_plan"], chip_eval["freehit_plan"])
            print(f"Summary Recommendation: {report['summary_recommendation']}")
            print(f"Wildcard Delta: +{chip_eval['delta_wildcard']} xP (Hurdle: +15.0)")
            print(f"Free Hit Delta: +{chip_eval['delta_freehit']} xP (Hurdle: +18.0)")

    if args.backtest:
        print(f"\n📈 RUNNING WALK-FORWARD BACKTEST (GW {args.start_gw} -> GW {args.end_gw})...")
        harness = WalkForwardBacktestHarness(start_gw=args.start_gw, end_gw=args.end_gw, history_df=history_df)
        res = harness.run_simulation(verbose=True)
        if not res.empty:
            print(f"\n🏆 Total Points Scored: {res['sota_engine_points'].sum()} pts across {len(res)} gameweeks.")

if __name__ == "__main__":
    main()
