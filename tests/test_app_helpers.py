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

if __name__ == "__main__":
    unittest.main()
