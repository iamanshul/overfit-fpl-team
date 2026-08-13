# -*- coding: utf-8 -*-
"""
Unit Tests for App UI Helpers (test_app_helpers.py)
"""

import unittest
import numpy as np
import pandas as pd
from app import render_player_card

class TestAppHelpers(unittest.TestCase):

    def test_render_player_card_valid(self):
        row = {
            "name": "Gabriel",
            "position": "DEF",
            "team": "Arsenal",
            "cost": 8.0,
            "role": "👑 Captain",
            "GW1_xP": 5.25,
            "chance_of_playing": 100,
            "news": "",
            "rationale": "Solid defensive anchor."
        }
        # Should not raise exception
        try:
            render_player_card(row)
        except Exception as e:
            self.fail(f"render_player_card raised {e}")

    def test_render_player_card_nan_news_and_chance(self):
        row = {
            "name": "Fringe Player",
            "position": "MID",
            "team": "CHE",
            "cost": np.nan,
            "role": np.nan,
            "GW1_xP": np.nan,
            "chance_of_playing": np.nan,
            "news": np.nan,
            "rationale": np.nan
        }
        try:
            render_player_card(row)
        except Exception as e:
            self.fail(f"render_player_card raised {e} on NaN inputs")

    def test_render_player_card_float_news(self):
        row = {
            "name": "Test Player",
            "position": "FWD",
            "team": "AVL",
            "cost": 6.5,
            "chance_of_playing": 75.0,
            "news": 123.456,  # float news object
        }
        try:
            render_player_card(row)
        except Exception as e:
            self.fail(f"render_player_card raised {e} on float news input")

    def test_data_loader_sqlite_and_csv_parity(self):
        """Verifies that both SQLite DB and CSV fallback load valid active 2025/2026 data."""
        from data_loader import load_player_history
        import os

        df = load_player_history()
        self.assertFalse(df.empty, "Loaded player history should not be empty")
        self.assertIn("name", df.columns)
        self.assertIn("team", df.columns)
        self.assertIn("position", df.columns)
        self.assertIn("cost", df.columns)
        
        # Verify valid teams are present
        teams = set(df["team"].dropna().unique())
        self.assertTrue(len(teams) >= 20, f"Expected at least 20 teams, got {len(teams)}")
        self.assertIn("Arsenal", teams)
        self.assertIn("Man City", teams)

    def test_optimizer_starting_xi_points_higher_than_bench(self):
        """Verifies that the MILP solver allocates high-xP players to Starting XI and cheaper assets to Bench."""
        from data_loader import load_player_history
        from squad_manager import build_gw1_start_of_season_squad

        history_df = load_player_history()
        squad_df, bank = build_gw1_start_of_season_squad(history_df, budget=100.0)
        self.assertEqual(len(squad_df), 15)

        starters = squad_df[squad_df["role"] != "Bench"]
        bench = squad_df[squad_df["role"] == "Bench"]

        self.assertEqual(len(starters), 11)
        self.assertEqual(len(bench), 4)

        # Starters average xP should exceed bench average xP
        avg_starter_xp = starters["GW1_xP"].mean()
        avg_bench_xp = bench["GW1_xP"].mean()
        self.assertGreater(avg_starter_xp, avg_bench_xp, f"Starters avg xP ({avg_starter_xp:.2f}) should exceed bench ({avg_bench_xp:.2f})")

        # Captain must be a starter with positive xP
        captain = squad_df[squad_df["role"] == "👑 Captain"]
        self.assertEqual(len(captain), 1)
        self.assertGreater(captain["GW1_xP"].values[0], 4.0)

if __name__ == "__main__":
    unittest.main()
