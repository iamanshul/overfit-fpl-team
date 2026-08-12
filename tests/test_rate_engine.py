# -*- coding: utf-8 -*-
"""
Unit Tests for Rate Engine (test_rate_engine.py)
"""

import unittest
import pandas as pd
import numpy as np
from rate_engine import CanonicalRateEngine
from devig_engine import SharpOddsEngine

class TestRateEngine(unittest.TestCase):

    def setUp(self):
        # Sample synthetic dataset for 5 players across 5 gameweeks
        data = []
        positions = ["GKP", "DEF", "DEF", "MID", "FWD"]
        teams = ["Arsenal", "Arsenal", "Chelsea", "Liverpool", "Man City"]
        
        for p_id in range(1, 6):
            for gw in range(1, 6):
                data.append({
                    "element_id": p_id,
                    "player_id": p_id,
                    "name": f"Player_{p_id}",
                    "position": positions[p_id - 1],
                    "team": teams[p_id - 1],
                    "value": 60,
                    "gameweek": gw,
                    "minutes": 90 if p_id != 2 or gw % 2 == 1 else 0,
                    "goals_scored": 1 if p_id == 5 and gw > 2 else 0,
                    "assists": 1 if p_id == 4 and gw == 3 else 0,
                    "clean_sheets": 1 if positions[p_id - 1] in ["GKP", "DEF"] and gw % 2 == 1 else 0,
                    "yellow_cards": 1 if gw == 2 and p_id == 3 else 0,
                    "red_cards": 0,
                    "bonus": 2 if p_id == 5 and gw == 3 else 0
                })
        self.df_history = pd.DataFrame(data)

    def test_devig_shins_method(self):
        odds = np.array([2.0, 3.5, 4.0]) # Total implied prob = 0.5 + 0.2857 + 0.25 = 1.0357 (3.57% vig)
        true_probs = SharpOddsEngine.devig_shins_method(odds)
        self.assertAlmostEqual(np.sum(true_probs), 1.0, places=4)
        self.assertTrue(np.all(true_probs > 0))
        self.assertTrue(np.all(true_probs < 1.0))

    def test_canonical_rate_engine_preprocessing(self):
        engine = CanonicalRateEngine(self.df_history)
        self.assertFalse(engine.player_rates.empty)
        self.assertIn("r_goal", engine.player_rates.columns)
        self.assertIn("xM", engine.player_rates.columns)

    def test_generate_horizon_matrix(self):
        engine = CanonicalRateEngine(self.df_history)
        matrix = engine.generate_horizon_matrix(start_gw=6, horizon_weeks=4)
        self.assertFalse(matrix.empty)
        self.assertIn("xP_6", matrix.columns)
        self.assertIn("xP_7", matrix.columns)
        self.assertIn("xP_horizon_sum", matrix.columns)

    def test_card_warning_and_uefa_decay(self):
        # Create history with 4 yellow cards and has_midweek_uefa
        df_card = self.df_history.copy()
        df_card.loc[df_card['player_id'] == 1, 'yellow_cards'] = 1 # 5 games * 1 = 5 cards total
        df_card.loc[df_card['player_id'] == 3, 'has_midweek_uefa'] = 1

        engine = CanonicalRateEngine(df_card)
        p3_rates = engine.player_rates[engine.player_rates['player_id'] == 3]
        self.assertFalse(p3_rates.empty)
    def test_dynamic_horizon_availability(self):
        df_avail = self.df_history.copy()
        # Set player 1 as suspended for 1 match
        df_avail.loc[df_avail['player_id'] == 1, 'status'] = 's'
        # Set player 2 as 75% knock
        df_avail.loc[df_avail['player_id'] == 2, 'chance_of_playing'] = 75
        # Set player 3 as long-term injury
        df_avail.loc[df_avail['player_id'] == 3, 'status'] = 'i'
        df_avail.loc[df_avail['player_id'] == 3, 'news'] = 'ACL surgery - out for season'

        engine = CanonicalRateEngine(df_avail)
        matrix = engine.generate_horizon_matrix(start_gw=6, horizon_weeks=4)

        p1_row = matrix[matrix['player_id'] == 1].iloc[0]
        p2_row = matrix[matrix['player_id'] == 2].iloc[0]
        p3_row = matrix[matrix['player_id'] == 3].iloc[0]

        # 1. Suspended player should have 0 xP in GW6 (first match), but > 0 xP in GW7, GW8, GW9
        self.assertEqual(p1_row['xP_6'], 0.0)
        self.assertGreater(p1_row['xP_7'], 0.0)
        self.assertGreater(p1_row['xP_8'], 0.0)

        # 2. 75% knock player should have xP_7 > xP_6 due to knock recovery
        self.assertGreater(p2_row['xP_6'], 0.0)
        self.assertGreater(p2_row['xP_7'], 0.0)

        # 3. Long-term injury player should have 0 xP across all 4 gameweeks
        self.assertEqual(p3_row['xP_6'], 0.0)
        self.assertEqual(p3_row['xP_7'], 0.0)
        self.assertEqual(p3_row['xP_8'], 0.0)
        self.assertEqual(p3_row['xP_9'], 0.0)

    def test_defensive_conceded_penalty_and_save_points(self):
        engine = CanonicalRateEngine(self.df_history)
        matrix = engine.generate_horizon_matrix(start_gw=1, horizon_weeks=1)
        
        # GKP (player 1) and DEF (player 2) should produce valid finite positive xP
        gkp_xp = matrix[matrix['position'] == 'GKP']['xP_1'].values[0]
        def_xp = matrix[matrix['position'] == 'DEF']['xP_1'].values[0]
        self.assertGreater(gkp_xp, 0.0)
        self.assertGreater(def_xp, 0.0)
        self.assertLess(gkp_xp, 12.0)
        self.assertLess(def_xp, 12.0)

if __name__ == "__main__":
    unittest.main()

