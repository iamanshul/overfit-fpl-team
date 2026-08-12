# -*- coding: utf-8 -*-
"""
Unit Tests for Backtester (test_backtester.py)
"""

import unittest
import pandas as pd
from backtester import WalkForwardBacktestHarness

class TestBacktester(unittest.TestCase):

    def setUp(self):
        # Synthetic 5-player pool history for backtester test
        data = []
        positions = ["GKP", "GKP", "DEF", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "MID", "FWD", "FWD", "FWD"]
        teams = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Spurs"] * 3
        
        for p_id in range(1, 16):
            for gw in range(1, 4):
                data.append({
                    "element_id": p_id,
                    "player_id": p_id,
                    "name": f"Player_{p_id}",
                    "position": positions[p_id - 1],
                    "team": teams[p_id - 1],
                    "value": 50,
                    "gameweek": gw,
                    "minutes": 90,
                    "total_points": 5 + (p_id % 4),
                    "goals_scored": 1 if p_id > 10 else 0,
                    "assists": 0,
                    "clean_sheets": 1 if p_id <= 5 else 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "bonus": 1
                })
        self.df_history = pd.DataFrame(data)

    def test_backtest_simulation(self):
        harness = WalkForwardBacktestHarness(start_gw=2, end_gw=3, history_df=self.df_history)
        res = harness.run_simulation(verbose=False)
        self.assertFalse(res.empty)
        self.assertIn("sota_engine_points", res.columns)
        self.assertIn("free_transfers", res.columns)

if __name__ == "__main__":
    unittest.main()
