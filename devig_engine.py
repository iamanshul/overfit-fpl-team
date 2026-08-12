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
        z_low, z_high = 0.0, 0.4
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
