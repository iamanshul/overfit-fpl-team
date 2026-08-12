# -*- coding: utf-8 -*-
"""
Strategic Chip Evaluator (chip_evaluator.py)
Evaluates Wildcard, Free Hit, Bench Boost, and Triple Captain trigger thresholds.
"""

from config import CHIP_RESERVATION_CURVES

class ChipEvaluator:
    """
    Evaluates whether point expectations meet chip reservation hurdle curves (Rho thresholds).
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or CHIP_RESERVATION_CURVES

    def evaluate(self, standard_plan, wildcard_plan, freehit_plan, bench_xp=0.0, captain_xp=0.0):
        """
        Calculates chip deltas and returns tactical recommendations.
        """
        pts_std = standard_plan.get("projected_points_gw1", 0.0)
        pts_wc = wildcard_plan.get("projected_points_gw1", 0.0)
        pts_fh = freehit_plan.get("projected_points_gw1", 0.0)

        delta_wc = round(pts_wc - pts_std, 2)
        delta_fh = round(pts_fh - pts_std, 2)

        decisions = {}

        decisions["wildcard"] = {
            "delta": delta_wc,
            "threshold": self.thresholds["wildcard"],
            "trigger": delta_wc >= self.thresholds["wildcard"]
        }

        decisions["freehit"] = {
            "delta": delta_fh,
            "threshold": self.thresholds["freehit"],
            "trigger": delta_fh >= self.thresholds["freehit"]
        }

        decisions["benchboost"] = {
            "bench_xp": bench_xp,
            "threshold": self.thresholds["benchboost"],
            "trigger": bench_xp >= self.thresholds["benchboost"]
        }

        decisions["triplecaptain"] = {
            "captain_extra_xp": captain_xp,
            "threshold": self.thresholds["triplecaptain"],
            "trigger": captain_xp >= self.thresholds["triplecaptain"]
        }

        rec = "HOLD ALL CHIPS"
        if decisions["freehit"]["trigger"]:
            rec = "🔥 PLAY FREE HIT"
        elif decisions["wildcard"]["trigger"]:
            rec = "🔥 PLAY WILDCARD"
        elif decisions["benchboost"]["trigger"]:
            rec = "🔥 PLAY BENCH BOOST"
        elif decisions["triplecaptain"]["trigger"]:
            rec = "🔥 PLAY TRIPLE CAPTAIN"

        return {
            "summary_recommendation": rec,
            "chip_decisions": decisions
        }
