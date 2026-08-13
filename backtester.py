# -*- coding: utf-8 -*-
"""
Walk-Forward Backtest Harness (backtester.py)
Replays decisions gameweek-by-gameweek with strict temporal isolation (no lookahead),
auto-subs, captain fallback, and 5-FT stacking state dynamics.
"""

import os
import pandas as pd
import numpy as np
from data_loader import load_player_history, fetch_clubelo_ratings
from rate_engine import CanonicalRateEngine
from optimizer import MultiPeriodMILP
from config import MAX_FREE_TRANSFERS, TRANSFER_HIT_COST, DATA_DIR

class WalkForwardBacktestHarness:
    """Stateful walk-forward simulator across completed Premier League gameweeks with formation-valid auto-subs and chip execution."""

    def __init__(self, start_gw=2, end_gw=10, history_df=None, simulate_chips=True):
        self.start_gw = start_gw
        self.end_gw = end_gw
        self.full_history = history_df if history_df is not None and not history_df.empty else load_player_history()
        self.simulate_chips = simulate_chips
        self.available_chips = {
            "wildcard_1": True,  # Eligible GW1-19
            "wildcard_2": True,  # Eligible GW20-38
            "freehit": True,
            "benchboost": True,
            "triplecaptain": True
        }

    def _select_initial_squad(self):
        """Builds valid initial 15-man squad prior to start_gw using CanonicalRateEngine projections (no lookahead bias)."""
        from squad_manager import build_gw1_start_of_season_squad
        squad_df, bank_rem = build_gw1_start_of_season_squad(self.full_history, budget=100.0)
        if not squad_df.empty:
            squad_ids = squad_df["player_id"].tolist()
            return squad_ids, round(bank_rem, 2)

        # Fallback to positional rate sorting without total_points lookahead
        pos_quotas = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
        squad_ids = []
        for pos, quota in pos_quotas.items():
            sub = self.full_history[self.full_history["position"] == pos].drop_duplicates("player_id")
            picked = sub["player_id"].head(quota).tolist()
            squad_ids.extend(picked)

        return squad_ids, 0.5

    def _resolve_formation_valid_autosubs(self, xi_ids, bench_ids, actual_mins_map, actual_pts_map, pos_map):
        """
        Simulates official FPL auto-substitutions ensuring strict formation validity:
        - Min 1 GKP, 3 DEF, 2 MID, 1 FWD
        - Max 1 GKP, 5 DEF, 5 MID, 3 FWD
        - Goalkeeper subbed exclusively by bench Goalkeeper.
        - Outfield zero-minute players subbed in bench order by eligible outfield players.
        """
        playing_xi = [p for p in xi_ids if actual_mins_map.get(p, 0) > 0]
        zero_mins_xi = [p for p in xi_ids if actual_mins_map.get(p, 0) == 0]
        
        current_lineup = list(playing_xi)
        subs_brought_on = []
        sub_points = 0
        
        # 1. GKP Substitution
        gkp_zero = [p for p in zero_mins_xi if pos_map.get(p) == "GKP"]
        bench_gkp = [p for p in bench_ids if pos_map.get(p) == "GKP"]
        if gkp_zero and bench_gkp:
            bgkp = bench_gkp[0]
            if actual_mins_map.get(bgkp, 0) > 0:
                current_lineup.append(bgkp)
                subs_brought_on.append(bgkp)
                sub_points += actual_pts_map.get(bgkp, 0)

        # 2. Outfield Substitutions in Priority Order
        outfield_bench = [p for p in bench_ids if pos_map.get(p) != "GKP" and actual_mins_map.get(p, 0) > 0]
        outfield_zero = [p for p in zero_mins_xi if pos_map.get(p) != "GKP"]
        
        for _ in outfield_zero:
            if not outfield_bench:
                break
                
            chosen_sub = None
            for cand in outfield_bench:
                trial_lineup = current_lineup + [cand]
                pos_counts = {
                    "GKP": sum(1 for p in trial_lineup if pos_map.get(p) == "GKP"),
                    "DEF": sum(1 for p in trial_lineup if pos_map.get(p) == "DEF"),
                    "MID": sum(1 for p in trial_lineup if pos_map.get(p) == "MID"),
                    "FWD": sum(1 for p in trial_lineup if pos_map.get(p) == "FWD"),
                }
                remaining_slots = 11 - len(trial_lineup)
                
                # Check that trial lineup satisfies formation feasibility bounds
                if (pos_counts["DEF"] <= 5 and pos_counts["MID"] <= 5 and pos_counts["FWD"] <= 3 and
                    pos_counts["DEF"] + remaining_slots >= 3 and
                    pos_counts["MID"] + remaining_slots >= 2 and
                    pos_counts["FWD"] + remaining_slots >= 1):
                    chosen_sub = cand
                    break
                    
            if chosen_sub is not None:
                current_lineup.append(chosen_sub)
                subs_brought_on.append(chosen_sub)
                sub_points += actual_pts_map.get(chosen_sub, 0)
                outfield_bench.remove(chosen_sub)

        return sub_points, len(subs_brought_on), subs_brought_on

    def run_simulation(self, verbose=True):
        """Executes walk-forward simulation across gameweeks."""
        if self.full_history.empty:
            if verbose:
                print("⚠️ No historical data found for backtesting.")
            return pd.DataFrame()

        elo_dict = fetch_clubelo_ratings()
        squad, bank = self._select_initial_squad()
        fts = 1
        ledger = []

        pos_map = dict(zip(self.full_history["player_id"], self.full_history["position"]))

        if verbose:
            print(f"🚀 Starting Walk-Forward Simulation from GW {self.start_gw} to GW {self.end_gw}...")

        for gw in range(self.start_gw, self.end_gw + 1):
            history_slice = self.full_history[self.full_history["gameweek"] < gw].copy()
            if history_slice.empty:
                continue

            engine = CanonicalRateEngine(history_slice)
            matrix = engine.generate_horizon_matrix(start_gw=gw, horizon_weeks=4, elo_dict=elo_dict)

            # Strategic Chip Decision Logic
            active_chip = None
            if self.simulate_chips:
                if gw in range(6, 9) and self.available_chips["wildcard_1"]:
                    active_chip = "wildcard"
                    self.available_chips["wildcard_1"] = False
                elif gw in range(30, 34) and self.available_chips["wildcard_2"]:
                    active_chip = "wildcard"
                    self.available_chips["wildcard_2"] = False

            optimizer = MultiPeriodMILP(matrix)
            plan = optimizer.solve_rolling_horizon(squad, bank, initial_fts=fts, active_chip=active_chip)

            if plan.get("status") != "Optimal":
                xi = squad[:11]
                captain = xi[0] if xi else None
                bench = squad[11:]
                hits_cost = 0
                t_made = 0
            else:
                squad = plan["squad_ids"]
                xi = plan["starting_xi_ids"]
                captain = plan["captain_id"]
                bench = plan["bench_ids"]
                bank = plan["bank"]
                hits_cost = 0 if active_chip == "wildcard" else plan["hits_cost"]
                t_made = len(plan["transfers_in"])

            actual_gw = self.full_history[self.full_history["gameweek"] == gw]
            actual_pts_map = dict(zip(actual_gw["player_id"], actual_gw["total_points"]))
            actual_mins_map = dict(zip(actual_gw["player_id"], actual_gw["minutes"]))

            # Calculate Starting XI points
            xi_pts = 0
            for p in xi:
                mins = actual_mins_map.get(p, 0)
                if mins > 0:
                    xi_pts += actual_pts_map.get(p, 0)

            # Captain & Vice-Captain Resolution
            cap_mins = actual_mins_map.get(captain, 0)
            if cap_mins > 0:
                cap_pts = actual_pts_map.get(captain, 0)
            else:
                vc_id = plan.get("vice_captain_id")
                if vc_id and actual_mins_map.get(vc_id, 0) > 0:
                    cap_pts = actual_pts_map.get(vc_id, 0)
                else:
                    cap_pts = 0

            # Execute Formation-Valid Auto-Substitutions
            sub_pts, subs_used, subs_list = self._resolve_formation_valid_autosubs(
                xi, bench, actual_mins_map, actual_pts_map, pos_map
            )

            total_gw_points = xi_pts + cap_pts + sub_pts - hits_cost

            # FT accumulation logic (up to 5 FTs)
            if active_chip == "wildcard":
                fts = 1
            else:
                remaining_fts = max(0, fts - t_made)
                fts = min(MAX_FREE_TRANSFERS, remaining_fts + 1)

            chip_label = f" ({active_chip.upper()})" if active_chip else ""

            ledger.append({
                "gameweek": gw,
                "sota_engine_points": total_gw_points,
                "bank": bank,
                "free_transfers": fts,
                "transfers_made": t_made,
                "hits_cost": hits_cost,
                "captain_id": captain,
                "auto_subs_used": subs_used,
                "active_chip": active_chip or "None"
            })

            if verbose:
                print(f"GW {gw:02d}{chip_label} | Points: {total_gw_points:3d} (Hits: -{hits_cost}) | Transfers: {t_made} | Bank: £{bank:.1f}m | FTs: {fts}")

        res_df = pd.DataFrame(ledger)
        if not res_df.empty:
            res_df["cumulative_points"] = res_df["sota_engine_points"].cumsum()
            out_path = os.path.join(DATA_DIR, "backtest_equity_curve.csv")
            res_df.to_csv(out_path, index=False)
            if verbose:
                print(f"✅ Backtest complete! Log saved to {out_path}")

        return res_df

