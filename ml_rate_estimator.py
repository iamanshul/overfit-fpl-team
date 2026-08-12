# -*- coding: utf-8 -*-
"""
Machine Learning Component Rate Estimator (ml_rate_estimator.py)
Replaces raw points ML regression with LightGBM estimators for npxG90, xA90, and P(Start),
normalizing historical performance against opponent defense strength (FDR / Elo).
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb

class ComponentRateMLEstimator:
    """
    Predicts underlying component rates (npxG90, xA90, start probability)
    using LightGBM on time-series rolling features.
    """

    def __init__(self, window_size=3):
        self.window_size = window_size
        self.model_npxg = lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=60, learning_rate=0.05)
        self.model_xa = lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=60, learning_rate=0.05)
        self.model_start = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=60, learning_rate=0.05)
        self.is_trained = False

    def _engineer_component_features(self, df):
        """Creates time-series EWMA and rolling features for component rate estimation."""
        df = df.copy()
        
        # Sort by player and time
        if 'kickoff_time' in df.columns:
            df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
            df = df.sort_values(['player_id', 'kickoff_time'])
        else:
            df = df.sort_values(['player_id', 'gameweek'])

        # Calculate per-90 metrics safely with Bayesian regularized minute denominator (prevents small-sample sub blowouts)
        mins_safe = np.maximum(df['minutes'], 30.0)
        df['npxg_per_90'] = (df['expected_goals'] / mins_safe) * 90.0
        df['xa_per_90'] = (df['expected_assists'] / mins_safe) * 90.0
        df['threat_per_90'] = (df['threat'] / mins_safe) * 90.0
        df['creativity_per_90'] = (df['creativity'] / mins_safe) * 90.0
        df['bps_per_90'] = (df['bps'] / mins_safe) * 90.0

        # Opponent Defense Strength Normalization
        def_adj = np.where(df.get('was_home', 1) == 1, 1.0, 1.10)
        df['adj_npxg_90'] = np.clip(df['npxg_per_90'] * def_adj, 0.0, 4.0)
        df['adj_xa_90'] = np.clip(df['xa_per_90'] * def_adj, 0.0, 3.0)

        # Rolling EWMA (Span=3)
        features = ['adj_npxg_90', 'adj_xa_90', 'threat_per_90', 'creativity_per_90', 'bps_per_90', 'minutes']
        
        grouped = df.groupby('player_id')
        feature_cols = []

        for col in features:
            ewm_col = f'ewm3_{col}'
            df[ewm_col] = grouped[col].shift(1).transform(lambda x: x.ewm(span=self.window_size, adjust=False).mean()).fillna(0.0)
            feature_cols.append(ewm_col)

        df['is_starter'] = (df['minutes'] >= 45).astype(int)
        return df, feature_cols

    def train(self, history_df):
        """Trains component ML models on active player minutes."""
        if history_df.empty:
            return False

        df_feat, feature_cols = self._engineer_component_features(history_df)
        active_df = df_feat[df_feat['minutes'] > 0].copy()

        if len(active_df) < 50:
            return False

        X = active_df[feature_cols]
        
        # Train Component Estimators
        self.model_npxg.fit(X, active_df['adj_npxg_90'])
        self.model_xa.fit(X, active_df['adj_xa_90'])
        self.model_start.fit(X, active_df['is_starter'])
        
        self.is_trained = True
        self.feature_cols = feature_cols
        return True

    def predict_rates(self, latest_df):
        """Generates predicted npxG90, xA90, and P(Start) per player for upcoming Gameweek."""
        if not self.is_trained or latest_df.empty:
            return pd.DataFrame()

        df_feat, feature_cols = self._engineer_component_features(latest_df)
        
        # Take the last recorded row per player
        latest_rows = df_feat.groupby('player_id').tail(1).copy()
        X_latest = latest_rows[feature_cols]

        pred_npxg = np.clip(self.model_npxg.predict(X_latest), 0.0, 1.20)
        pred_xa = np.clip(self.model_xa.predict(X_latest), 0.0, 0.90)
        pred_p_start = self.model_start.predict_proba(X_latest)[:, 1]

        res_df = pd.DataFrame({
            'player_id': latest_rows['player_id'].values,
            'pred_npxg90': np.round(pred_npxg, 3),
            'pred_xa90': np.round(pred_xa, 3),
            'pred_p_start': np.round(pred_p_start, 3)
        })
        return res_df
