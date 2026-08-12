# -*- coding: utf-8 -*-
"""
Unit Tests for Article Sentiment Engine (test_article_analyzer.py)
"""

import unittest
import pandas as pd
from article_analyzer import ArticleSentimentEngine

class TestArticleAnalyzer(unittest.TestCase):

    def setUp(self):
        self.matrix = pd.DataFrame([
            {
                "player_id": 1,
                "name": "B.Fernandes",
                "position": "MID",
                "team": "Man Utd",
                "cost": 8.5,
                "r_goal": 0.35,
                "r_assist": 0.25,
                "xM": 90.0,
                "xP_horizon_sum": 32.0,
                "xP_1": 5.5,
                "xP_2": 5.5
            },
            {
                "player_id": 2,
                "name": "De Bruyne",
                "position": "MID",
                "team": "Man City",
                "cost": 10.5,
                "r_goal": 0.20,
                "r_assist": 0.40,
                "xM": 70.0,
                "xP_horizon_sum": 30.0,
                "xP_1": 5.0,
                "xP_2": 5.0
            }
        ])

    def test_analyze_article_exact_name(self):
        engine = ArticleSentimentEngine(self.matrix)
        res = engine.analyze_article("Bruno Fernandes has been in sensational form for Man Utd.")
        self.assertFalse(res.empty)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["name"], "B.Fernandes")

    def test_analyze_article_alias_match(self):
        engine = ArticleSentimentEngine(self.matrix)
        res = engine.analyze_article("KDB is returning to full fitness ahead of the weekend.")
        self.assertFalse(res.empty)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["name"], "De Bruyne")

    def test_apply_user_overrides(self):
        engine = ArticleSentimentEngine(self.matrix)
        updated = engine.apply_user_overrides({1: 1.20})
        orig_xp1 = self.matrix.loc[self.matrix["player_id"] == 1, "xP_1"].values[0]
        new_xp1 = updated.loc[updated["player_id"] == 1, "xP_1"].values[0]
        self.assertAlmostEqual(new_xp1, orig_xp1 * 1.20, places=3)

    def test_analyze_article_injury_and_doubt(self):
        engine = ArticleSentimentEngine(self.matrix)
        # Test 1: Severe injury mention
        res_inj = engine.analyze_article("Bruno Fernandes has been ruled out with a severe hamstring tear.")
        self.assertFalse(res_inj.empty)
        self.assertEqual(res_inj.iloc[0]["suggested_multiplier"], 0.0)
        self.assertIn("RULED OUT", res_inj.iloc[0]["verdict"])

        # Test 2: Doubt mention
        res_doubt = engine.analyze_article("Kevin De Bruyne is a major doubt with muscle tightness.")
        self.assertFalse(res_doubt.empty)
        self.assertEqual(res_doubt.iloc[0]["suggested_multiplier"], 0.50)
        self.assertIn("DOUBTFUL", res_doubt.iloc[0]["verdict"])

if __name__ == "__main__":
    unittest.main()
