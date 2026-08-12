# -*- coding: utf-8 -*-
"""
Unit Tests for Optimizer (test_optimizer.py)
"""

import unittest
import pandas as pd
import numpy as np
from optimizer import MultiPeriodMILP

class TestOptimizer(unittest.TestCase):

    def setUp(self):
        # Create a valid pool of 20 players across positions
        data = []
        positions = ["GKP"] * 3 + ["DEF"] * 7 + ["MID"] * 7 + ["FWD"] * 3
        teams = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Spurs"] * 4
        
        for p_id in range(1, 21):
            row = {
                "player_id": p_id,
                "name": f"Player_{p_id}",
                "position": positions[p_id - 1],
                "team": teams[p_id - 1],
                "cost": 5.0 + (p_id % 5) * 1.0,
                "xP_1": round(2.0 + (p_id % 6) * 1.5, 2),
                "xP_2": round(2.0 + ((p_id + 1) % 6) * 1.5, 2),
                "xP_3": round(2.0 + ((p_id + 2) % 6) * 1.5, 2),
                "xP_4": round(2.0 + ((p_id + 3) % 6) * 1.5, 2),
            }
            data.append(row)

        self.xp_matrix = pd.DataFrame(data)
        # Select valid 15-man squad conforming to positional quotas (2 GKP, 5 DEF, 5 MID, 3 FWD)
        self.initial_squad = [1, 2, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20]
        self.initial_bank = 1.0

    def test_solve_rolling_horizon(self):
        optimizer = MultiPeriodMILP(self.xp_matrix)
        res = optimizer.solve_rolling_horizon(self.initial_squad, self.initial_bank, initial_fts=1)
        self.assertEqual(res["status"], "Optimal")
        self.assertEqual(len(res["squad_ids"]), 15)
        self.assertEqual(len(res["starting_xi_ids"]), 11)
        self.assertIsNotNone(res["captain_id"])
        self.assertIn("projected_points_gw1", res)

    def test_solve_with_purchase_prices_profit_tax(self):
        optimizer = MultiPeriodMILP(self.xp_matrix)
        # Purchase prices lower than current cost
        purchase_prices = {p: 4.0 for p in self.initial_squad}
        res = optimizer.solve_rolling_horizon(
            self.initial_squad, self.initial_bank, initial_fts=1, purchase_prices=purchase_prices
        )
        self.assertEqual(res["status"], "Optimal")
        self.assertEqual(len(res["squad_ids"]), 15)
        self.assertEqual(len(res["starting_xi_ids"]), 11)

if __name__ == "__main__":
    unittest.main()
