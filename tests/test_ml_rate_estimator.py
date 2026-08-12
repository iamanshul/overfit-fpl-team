# -*- coding: utf-8 -*-
"""
Unit Test Suite for ML Component Rate Estimator (tests/test_ml_rate_estimator.py)
"""

import unittest
import pandas as pd
import numpy as np
from ml_rate_estimator import ComponentRateMLEstimator

class TestComponentRateMLEstimator(unittest.TestCase):

    def setUp(self):
        # Build synthetic player history dataframe
        np.random.seed(42)
        n_rows = 120
        self.sample_df = pd.DataFrame({
            'player_id': np.repeat([1, 2, 3], 40),
            'gameweek': np.tile(np.arange(1, 41), 3),
            'minutes': np.random.choice([0, 60, 90], size=n_rows, p=[0.1, 0.2, 0.7]),
            'expected_goals': np.random.uniform(0.0, 0.8, size=n_rows),
            'expected_assists': np.random.uniform(0.0, 0.5, size=n_rows),
            'threat': np.random.uniform(5.0, 50.0, size=n_rows),
            'creativity': np.random.uniform(5.0, 40.0, size=n_rows),
            'bps': np.random.randint(5, 35, size=n_rows),
            'was_home': np.random.choice([0, 1], size=n_rows)
        })
        self.estimator = ComponentRateMLEstimator(window_size=3)

    def test_train_and_predict(self):
        trained = self.estimator.train(self.sample_df)
        self.assertTrue(trained)
        self.assertTrue(self.estimator.is_trained)

        res_df = self.estimator.predict_rates(self.sample_df)
        self.assertFalse(res_df.empty)
        self.assertIn('player_id', res_df.columns)
        self.assertIn('pred_npxg90', res_df.columns)
        self.assertIn('pred_xa90', res_df.columns)
        self.assertIn('pred_p_start', res_df.columns)
        self.assertEqual(len(res_df), 3)

if __name__ == '__main__':
    unittest.main()
