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
    """Stateful walk-forward simulator across completed Premier League gameweeks."""

    def __init__(self, start_gw=2, end_gw=10, history_df=None):
        self.start_gw = start_gw
        self.end_gw = end_gw
        self.full_history = history_df if history_df is not None and not history_df.empty else load_player_history()

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

        if verbose:
            print(f"🚀 Starting Walk-Forward Simulation from GW {self.start_gw} to GW {self.end_gw}...")

        for gw in range(self.start_gw, self.end_gw + 1):
            history_slice = self.full_history[self.full_history["gameweek"] < gw].copy()
            if history_slice.empty:
                continue

            engine = CanonicalRateEngine(history_slice)
            matrix = engine.generate_horizon_matrix(start_gw=gw, horizon_weeks=4, elo_dict=elo_dict)

            optimizer = MultiPeriodMILP(matrix)
            plan = optimizer.solve_rolling_horizon(squad, bank, initial_fts=fts)

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
                hits_cost = plan["hits_cost"]
                t_made = len(plan["transfers_in"])

            actual_gw = self.full_history[self.full_history["gameweek"] == gw]
            actual_pts_map = dict(zip(actual_gw["player_id"], actual_gw["total_points"]))
            actual_mins_map = dict(zip(actual_gw["player_id"], actual_gw["minutes"]))

            xi_pts = 0
            played_xi = []
            zero_mins_xi = []

            for p in xi:
                mins = actual_mins_map.get(p, 0)
                pts = actual_pts_map.get(p, 0)
                if mins > 0:
                    xi_pts += pts
                    played_xi.append(p)
                else:
                    zero_mins_xi.append(p)

            cap_mins = actual_mins_map.get(captain, 0)
            if cap_mins > 0:
                cap_pts = actual_pts_map.get(captain, 0)
            else:
                vc_id = plan.get("vice_captain_id")
                if vc_id and actual_mins_map.get(vc_id, 0) > 0:
                    cap_pts = actual_pts_map.get(vc_id, 0)
                else:
                    cap_pts = 0

            sub_pts = 0
            subs_used = 0
            for zp in zero_mins_xi:
                if subs_used >= len(bench):
                    break
                sub_id = bench[subs_used]
                if actual_mins_map.get(sub_id, 0) > 0:
                    sub_pts += actual_pts_map.get(sub_id, 0)
                subs_used += 1

            total_gw_points = xi_pts + cap_pts + sub_pts - hits_cost

            # FT accumulation logic (up to 5 FTs)
            remaining_fts = max(0, fts - t_made)
            fts = min(MAX_FREE_TRANSFERS, remaining_fts + 1)

            ledger.append({
                "gameweek": gw,
                "sota_engine_points": total_gw_points,
                "bank": bank,
                "free_transfers": fts,
                "transfers_made": t_made,
                "hits_cost": hits_cost,
                "captain_id": captain,
                "auto_subs_used": subs_used
            })

            if verbose:
                print(f"GW {gw:02d} | Points: {total_gw_points:3d} (Hits: -{hits_cost}) | Transfers: {t_made} | Bank: £{bank:.1f}m | FTs: {fts}")

        res_df = pd.DataFrame(ledger)
        if not res_df.empty:
            res_df["cumulative_points"] = res_df["sota_engine_points"].cumsum()
            out_path = os.path.join(DATA_DIR, "backtest_equity_curve.csv")
            res_df.to_csv(out_path, index=False)
            if verbose:
                print(f"✅ Backtest complete! Log saved to {out_path}")

        return res_df
