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


class MacroSeasonChipScheduler:
    """
    38-Gameweek Global Strategic Chip Scheduler.
    Maps long-term chip timing across the full Premier League calendar:
    - Wildcard 1 (GW6-GW8): Early portfolio restructuring to bank early price gainers.
    - Free Hit (Target Major Blank e.g. GW29): Field 11 active starters during FA Cup clashes.
    - Wildcard 2 (GW30-GW33): Set up 15 Double Gameweek starters right before Bench Boost.
    - Bench Boost (Mega DGW e.g. DGW34/DGW37): Maximize 30 player fixture appearances (+30 to +45 net pts).
    - Triple Captain (Target Home DGW): Deploy on marquee talisman (Haaland/Salah/Palmer) with easiest double.
    """

    @staticmethod
    def generate_macro_roadmap(schedule_df=None):
        """
        Generates full 38-gameweek tactical roadmap table with optimal chip windows.
        """
        import pandas as pd
        roadmap = []
        for gw in range(1, 39):
            phase = "Early Season Anchor" if gw <= 5 else (
                "Autumn Value Building" if gw <= 14 else (
                "Festive Congestion" if gw <= 20 else (
                "Spring DGW / BGW Horizon" if gw <= 28 else (
                "Championship Run-In"
            ))))
            
            recommended_chip = "None (Standard 5-FT Rolling)"
            strategy_note = "Maximize baseline Starting XI EV and accumulate up to 5 Free Transfers."
            bgw_dgw_status = "Standard (10 Matches)"
            
            if gw in [6, 7, 8]:
                recommended_chip = "⚡ WILDCARD 1 (Prime Window)"
                strategy_note = "Restructure away from underperforming early punts and lock in accumulated squad price rises."
            elif gw == 29:
                recommended_chip = "🛡️ FREE HIT (Blank GW Shield)"
                strategy_note = "Deploy Free Hit to navigate massive FA Cup quarterfinal blank gameweek without taking transfer hits."
                bgw_dgw_status = "⚠️ Major Blank Gameweek (BGW)"
            elif gw in [31, 32, 33]:
                recommended_chip = "⚡ WILDCARD 2 (Pre-DGW Prep)"
                strategy_note = "Restructure full 15-man squad with 15 active Double Gameweek starters in preparation for Bench Boost."
            elif gw in [34, 37]:
                recommended_chip = "🚀 BENCH BOOST (Mega DGW)"
                strategy_note = "Activate with 15 double-gameweek starters to achieve 30 total fixture appearances across your squad."
                bgw_dgw_status = "🔥 Mega Double Gameweek (DGW)"
            elif gw in [25, 26, 35]:
                recommended_chip = "👑 TRIPLE CAPTAIN (Talisman DGW)"
                strategy_note = "Deploy on elite marquee captain (Haaland/Salah/Palmer) during a favorable home Double Gameweek matchup."
                bgw_dgw_status = "⭐ Double Gameweek (DGW)"

            roadmap.append({
                "Gameweek": f"GW {gw:02d}",
                "GW_Num": gw,
                "Season_Phase": phase,
                "Fixture_Status": bgw_dgw_status,
                "Strategic_Chip_Target": recommended_chip,
                "Tactical_Objective": strategy_note
            })
            
        return pd.DataFrame(roadmap)

