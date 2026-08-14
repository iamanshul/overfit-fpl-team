# -*- coding: utf-8 -*-
"""
Sharp Odds De-Vigger Engine (devig_engine.py)
Implements Shin's algorithm to convert bookmaker odds into de-vigged market probabilities.
"""

import numpy as np

class SharpOddsEngine:
    """
    De-vigs betting market overround (vigorish) to extract true market-implied probabilities.
    """

    @staticmethod
    def devig_shins_method(decimal_odds: np.ndarray) -> np.ndarray:
        """
        Applies Shin's method to remove bookmaker margin and insider trading bias.
        decimal_odds: 1D array of decimal odds for exhaustive mutually exclusive events.
        Returns: 1D array of true probabilities summing to 1.0.
        """
        decimal_odds = np.asarray(decimal_odds, dtype=float)
        implied_probs = 1.0 / decimal_odds
        beta = np.sum(implied_probs)  # Bookmaker overround

        if abs(beta - 1.0) < 1e-5:
            return implied_probs

        n = len(decimal_odds)
        z_low = 0.0
        z_high = min(0.35, 1.95 / max(2.0, float(n)))
        z = 0.0

        for _ in range(30):
            z_mid = (z_low + z_high) / 2.0
            val = np.sum(np.sqrt(z_mid**2 + 4.0 * (1.0 - z_mid) * (implied_probs / beta)))
            if val > (2.0 - (n - 2.0) * z_mid):
                z_low = z_mid
            else:
                z_high = z_mid
            z = z_mid

        true_probs = (np.sqrt(z**2 + 4.0 * (1.0 - z) * (implied_probs / beta)) - z) / (2.0 * (1.0 - z))
        return true_probs / np.sum(true_probs)

    @staticmethod
    def devig_proportional(decimal_odds: np.ndarray) -> np.ndarray:
        """Simple proportional normalization fallback."""
        implied_probs = 1.0 / np.asarray(decimal_odds, dtype=float)
        return implied_probs / np.sum(implied_probs)


class BivariateDixonColes:
    """
    Computes exact joint match scoreline probabilities P(Home=x, Away=y)
    with Dixon-Coles low-score dependency coupling parameter rho.
    """

    def __init__(self, rho=-0.11):
        self.rho = rho

    def _tau(self, x: int, y: int, lambda_home: float, mu_away: float) -> float:
        if x == 0 and y == 0:
            return 1.0 - lambda_home * mu_away * self.rho
        elif x == 0 and y == 1:
            return 1.0 + lambda_home * self.rho
        elif x == 1 and y == 0:
            return 1.0 + mu_away * self.rho
        elif x == 1 and y == 1:
            return 1.0 - self.rho
        else:
            return 1.0

    def compute_match_probabilities(self, lambda_home: float, mu_away: float, max_goals: int = 8):
        """
        Generates full (max_goals+1 x max_goals+1) joint probability matrix.
        Returns: home_cs_prob, away_cs_prob, home_win_prob, draw_prob, away_win_prob, joint_matrix
        """
        import scipy.stats as stats
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                p_x = stats.poisson.pmf(x, lambda_home)
                p_y = stats.poisson.pmf(y, mu_away)
                adj = self._tau(x, y, lambda_home, mu_away)
                matrix[x, y] = max(0.0, p_x * p_y * adj)

        tot = np.sum(matrix)
        if tot > 0:
            matrix /= tot

        home_cs = float(np.sum(matrix[:, 0]))  # Away scores 0
        away_cs = float(np.sum(matrix[0, :]))  # Home scores 0
        home_win = float(np.sum(np.tril(matrix, -1)))
        draw = float(np.sum(np.diag(matrix)))
        away_win = float(np.sum(np.triu(matrix, 1)))

        return {
            "home_cs_prob": home_cs,
            "away_cs_prob": away_cs,
            "home_win_prob": home_win,
            "draw_prob": draw,
            "away_win_prob": away_win,
            "joint_matrix": matrix
        }

