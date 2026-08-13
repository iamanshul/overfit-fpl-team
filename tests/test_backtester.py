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
        harness = WalkForwardBacktestHarness(start_gw=2, end_gw=3, history_df=self.df_history, simulate_chips=True)
        res = harness.run_simulation(verbose=False)
        self.assertFalse(res.empty)
        self.assertIn("sota_engine_points", res.columns)
        self.assertIn("free_transfers", res.columns)
        self.assertIn("auto_subs_used", res.columns)

    def test_formation_valid_autosub_resolution(self):
        harness = WalkForwardBacktestHarness(start_gw=2, end_gw=3, history_df=self.df_history)
        pos_map = {
            1: "GKP", 2: "DEF", 3: "DEF", 4: "DEF",
            5: "MID", 6: "MID", 7: "MID", 8: "MID", 9: "MID",
            10: "FWD", 11: "FWD",
            12: "GKP", 13: "MID", 14: "DEF", 15: "FWD" # Bench: [GKP, MID, DEF, FWD]
        }
        xi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # 3-5-2 Starting Lineup
        bench = [12, 13, 14, 15]

        # Defender 4 played 0 minutes
        actual_mins = {1: 90, 2: 90, 3: 90, 4: 0, 5: 90, 6: 90, 7: 90, 8: 90, 9: 90, 10: 90, 11: 90, 12: 0, 13: 90, 14: 90, 15: 90}
        actual_pts = {14: 6, 13: 8}

        sub_pts, subs_used, subs_list = harness._resolve_formation_valid_autosubs(xi, bench, actual_mins, actual_pts, pos_map)
        
        # Player 13 (MID) is 1st on outfield bench, but subbing him leaves 2 DEFs (illegal).
        # Therefore, Player 14 (DEF) MUST be subbed in instead!
        self.assertEqual(subs_used, 1)
        self.assertIn(14, subs_list)
        self.assertNotIn(13, subs_list)
        self.assertEqual(sub_pts, 6)

    def test_macro_season_chip_scheduler(self):
        from chip_evaluator import MacroSeasonChipScheduler
        roadmap = MacroSeasonChipScheduler.generate_macro_roadmap()
        self.assertEqual(len(roadmap), 38)
        self.assertIn("Strategic_Chip_Target", roadmap.columns)
        self.assertIn("WILDCARD 1", roadmap.iloc[5]["Strategic_Chip_Target"]) # GW6

if __name__ == "__main__":
    unittest.main()

