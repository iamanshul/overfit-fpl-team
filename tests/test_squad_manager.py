# -*- coding: utf-8 -*-
"""
Unit Tests for Squad Manager (test_squad_manager.py)
"""

import unittest
import numpy as np
import pandas as pd
from squad_manager import generate_player_rationale, get_active_squad_state

class TestSquadManager(unittest.TestCase):

    def test_generate_player_rationale_valid(self):
        row = {
            "name": "Haaland",
            "position": "FWD",
            "team": "Man City",
            "cost": 15.5,
            "r_goal": 0.8,
            "r_assist": 0.2,
            "xM": 90.0,
            "team_cs_rate": 0.4
        }
        res = generate_player_rationale(row, gw1_xp=5.5)
        self.assertIn("Haaland", res)
        self.assertIn("FWD", res)

    def test_generate_player_rationale_nan_handling(self):
        row = {
            "name": "Unknown",
            "position": "DEF",
            "team": "UNK",
            "cost": np.nan,
            "r_goal": np.nan,
            "r_assist": np.nan,
            "xM": np.nan,
            "team_cs_rate": np.nan
        }
        res = generate_player_rationale(row, gw1_xp=np.nan)
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 0)
        self.assertNotIn("NaN", res)

    def test_generate_player_rationale_empty_row(self):
        res = generate_player_rationale({}, gw1_xp=None)
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 0)

    def test_gw1_squad_budget_cap(self):
        from data_loader import load_player_history
        from squad_manager import build_gw1_start_of_season_squad
        df = load_player_history()
        squad_df, bank_rem = build_gw1_start_of_season_squad(df, budget=100.0)
        if not squad_df.empty:
            total_cost = squad_df["cost"].sum()
            self.assertLessEqual(round(total_cost, 2), 100.0, f"Squad total cost {total_cost} exceeds 100.0m limit!")
            self.assertGreaterEqual(round(bank_rem, 2), 0.0, f"Remaining bank {bank_rem} is negative!")

    def test_validate_squad_invariants(self):
        from squad_manager import validate_squad_invariants
        bad_df = pd.DataFrame({
            "cost": [10.0] * 15, # 150.0m total > 100.0m
            "team": ["Arsenal"] * 4 + ["Liverpool"] * 11,
            "role": ["Starter"] * 12 + ["Bench"] * 3, # 12 starters != 11
            "chance_of_playing": [100] * 15,
            "GW1_xP": [5.0] * 15
        })
        is_valid, errors = validate_squad_invariants(bad_df, budget=100.0)
        self.assertFalse(is_valid)
    def test_calculate_selling_price_profit_and_loss(self):
        from squad_manager import calculate_selling_price
        # 1. No price change: P_sell = P_buy
        self.assertEqual(calculate_selling_price(6.0, 6.0), 6.0)
        # 2. Price drop (loss realized in full): P_sell = P_curr
        self.assertEqual(calculate_selling_price(6.0, 5.8), 5.8)
        # 3. +0.1m gain: (1 // 2 = 0) -> P_sell = 6.0
        self.assertEqual(calculate_selling_price(6.0, 6.1), 6.0)
        # 4. +0.2m gain: (2 // 2 = 1) -> P_sell = 6.1
        self.assertEqual(calculate_selling_price(6.0, 6.2), 6.1)
        # 5. +0.5m gain: (5 // 2 = 2) -> P_sell = 6.2
        self.assertEqual(calculate_selling_price(6.0, 6.5), 6.2)
        # 6. +1.0m gain: (10 // 2 = 5) -> P_sell = 6.5
        self.assertEqual(calculate_selling_price(6.0, 7.0), 6.5)

    def test_squad_adversarial_critic_and_budget_audit(self):
        from squad_manager import SquadAdversarialCritic
        sample_squad = pd.DataFrame([
            {"player_id": 1, "name": "Raya", "position": "GKP", "team": "Arsenal", "cost": 6.0, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 4.5},
            {"player_id": 2, "name": "Forster", "position": "GKP", "team": "Spurs", "cost": 4.0, "role": "Bench", "xM": 0.0, "chance_of_playing": 100.0, "GW1_xP": 0.0},
            {"player_id": 3, "name": "Gabriel", "position": "DEF", "team": "Arsenal", "cost": 8.0, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 4.8},
            {"player_id": 4, "name": "Guéhi", "position": "DEF", "team": "Crystal Palace", "cost": 6.0, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 3.9},
            {"player_id": 5, "name": "Muñoz", "position": "DEF", "team": "Crystal Palace", "cost": 5.5, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 4.2},
            {"player_id": 6, "name": "Calafiori", "position": "DEF", "team": "Arsenal", "cost": 5.5, "role": "Starter", "xM": 85.0, "chance_of_playing": 100.0, "GW1_xP": 4.0},
            {"player_id": 7, "name": "Cash", "position": "DEF", "team": "Aston Villa", "cost": 4.5, "role": "Bench", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 3.8},
            {"player_id": 8, "name": "B.Fernandes", "position": "MID", "team": "Man Utd", "cost": 12.0, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 6.2},
            {"player_id": 9, "name": "Mbeumo", "position": "MID", "team": "Man Utd", "cost": 8.0, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 5.5},
            {"player_id": 10, "name": "Rogers", "position": "MID", "team": "Aston Villa", "cost": 7.5, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 4.8},
            {"player_id": 11, "name": "Rayan", "position": "MID", "team": "Bournemouth", "cost": 6.5, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 4.6},
            {"player_id": 12, "name": "Dewsbury-Hall", "position": "MID", "team": "Everton", "cost": 6.5, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 4.4},
            {"player_id": 13, "name": "Haaland", "position": "FWD", "team": "Man City", "cost": 15.5, "role": "👑 Captain", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 7.8},
            {"player_id": 14, "name": "Thiago", "position": "FWD", "team": "Brentford", "cost": 8.0, "role": "Starter", "xM": 90.0, "chance_of_playing": 100.0, "GW1_xP": 5.2},
            {"player_id": 15, "name": "Scarlett", "position": "FWD", "team": "Spurs", "cost": 4.0, "role": "Bench", "xM": 15.0, "chance_of_playing": 100.0, "GW1_xP": 1.0},
        ])
        report = SquadAdversarialCritic.critique_squad(sample_squad, budget=100.0)
        self.assertIn("overall_grade", report)
        self.assertIn("budget_score", report)
        self.assertIn("eo_score", report)
        self.assertEqual(report["budget_score"], 100) # Spent 100.0m exactly
        self.assertGreaterEqual(report["eo_score"], 80) # Contains Haaland + Bruno

if __name__ == "__main__":
    unittest.main()
