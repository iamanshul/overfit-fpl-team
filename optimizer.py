# -*- coding: utf-8 -*-
"""
Multi-Period Mixed-Integer Linear Programming (MILP) Optimizer (optimizer.py)
Solves rolling horizon Model Predictive Control (MPC) squad selection, transfers,
FT accumulation (up to 5 FTs), hit penalties, and shadow chip activation solves.
"""

import os
import pulp
import pandas as pd
import numpy as np
from config import (
    ROLLING_HORIZON_WEEKS, HORIZON_DECAY_FACTOR, FREE_TRANSFER_OPTION_VALUE,
    TRANSFER_HIT_COST, BANK_SALVAGE_WEIGHT, MAX_FREE_TRANSFERS,
    SQUAD_SIZE, SQUAD_QUOTAS, XI_QUOTAS_MIN, XI_QUOTAS_MAX, MAX_PLAYERS_PER_TEAM,
    CHIP_RESERVATION_CURVES
)

class MultiPeriodMILP:
    """
    Rolling Horizon Model Predictive Control Solver using PuLP CBC.
    """

    def __init__(self, xp_matrix_df):
        self.df = xp_matrix_df.copy() if not xp_matrix_df.empty else pd.DataFrame()
        if not self.df.empty:
            self.df["player_id"] = self.df["player_id"].astype(int)
            self.df["cost"] = self.df["cost"].fillna(5.0).astype(float)
            self.players = self.df["player_id"].tolist()
            self.costs = {p: (5.0 if pd.isna(c) else float(c)) for p, c in zip(self.df["player_id"], self.df["cost"])}
            self.positions = dict(zip(self.df["player_id"], self.df["position"]))
            self.teams = dict(zip(self.df["player_id"], self.df["team"]))
            self.names = dict(zip(self.df["player_id"], self.df["name"]))
            self.gw_cols = [c for c in self.df.columns if c.startswith("xP_") and c != "xP_horizon_sum"]
            self.gws = [int(c.split("_")[1]) for c in self.gw_cols]
        else:
            self.players = []
            self.gws = []

    def solve_rolling_horizon(self, initial_squad, initial_bank, initial_fts=1, active_chip=None, formation=None, locked_player_ids=None, min_squad_cost=None, max_budget=None, risk_aversion=0.0, purchase_prices=None):
        """
        Solves optimal squad roadmap across rolling horizon gameweeks.
        min_squad_cost: Minimum total squad cost to force full budget utilization (e.g. 99.0m).
        max_budget: Hard upper bound on total squad wealth (e.g. 100.0m for start of season).
        risk_aversion: Weight lambda (0.0 to 0.5) penalizing same-team double defense variance.
        purchase_prices: Dict of {player_id: purchase_price} to enforce 50% profit tax on sells.
        """
        if not self.players or not self.gws:
            return {"status": "Infeasible / Empty Data"}

        # Calculate exact selling prices with 50% profit tax
        selling_costs = {}
        for p in self.players:
            if purchase_prices and p in purchase_prices:
                p_buy = float(purchase_prices[p])
                p_curr = float(self.costs[p])
                if p_curr <= p_buy:
                    selling_costs[p] = p_curr
                else:
                    p_buy_t = int(round(p_buy * 10))
                    p_curr_t = int(round(p_curr * 10))
                    profit_t = max(0, p_curr_t - p_buy_t)
                    selling_costs[p] = round((p_buy_t + profit_t // 2) / 10.0, 1)
            else:
                selling_costs[p] = self.costs[p]

        prob = pulp.LpProblem("FPL_Rolling_MPC", pulp.LpMaximize)

        x = {}       # Squad presence
        y = {}       # XI selection
        c = {}       # Captaincy
        v = {}       # Vice-Captaincy
        tr_in = {}   # Transfers In
        tr_out = {}  # Transfers Out
        hits = {}    # Penalty Hits
        fts_state = {} # Accumulated Free Transfers (1 to 5)

        for t_idx, gw in enumerate(self.gws):
            hits[gw] = pulp.LpVariable(f"hits_{gw}", lowBound=0, cat=pulp.LpInteger)
            fts_state[gw] = pulp.LpVariable(f"fts_{gw}", lowBound=1, upBound=MAX_FREE_TRANSFERS, cat=pulp.LpInteger)
            
            for p in self.players:
                x[(p, gw)] = pulp.LpVariable(f"x_{p}_{gw}", cat=pulp.LpBinary)
                y[(p, gw)] = pulp.LpVariable(f"y_{p}_{gw}", cat=pulp.LpBinary)
                c[(p, gw)] = pulp.LpVariable(f"c_{p}_{gw}", cat=pulp.LpBinary)
                v[(p, gw)] = pulp.LpVariable(f"v_{p}_{gw}", cat=pulp.LpBinary)
                tr_in[(p, gw)] = pulp.LpVariable(f"tr_in_{p}_{gw}", cat=pulp.LpBinary)
                tr_out[(p, gw)] = pulp.LpVariable(f"tr_out_{p}_{gw}", cat=pulp.LpBinary)

        # Objective Function: Discounted xP (Starters + Bench Auto-Sub EV) - Hit penalties + Bank Salvage + FT Value
        obj_terms = []
        for t_idx, gw in enumerate(self.gws):
            decay = HORIZON_DECAY_FACTOR ** t_idx
            xp_col = f"xP_{gw}"
            xp_map = dict(zip(self.df["player_id"], self.df[xp_col]))
            # Map Effective Ownership (selected_by_percent) for captaincy risk guardrails
            sel_pct_map = dict(zip(self.df["player_id"], self.df.get("selected_by_percent", pd.Series([0.0]*len(self.df)))))

            for p in self.players:
                # 1. Starting XI Contribution
                obj_terms.append(decay * y[(p, gw)] * xp_map[p])
                
                # 2. Formation-Valid Bench Auto-Sub Expectation:
                # GKP sub weight: 0.025, DEF sub weight: 0.16 (formation floor), MID/FWD sub weight: 0.12
                bench_weight = 0.025 if self.positions[p] == "GKP" else (0.16 if self.positions[p] == "DEF" else 0.12)
                bench_var = x[(p, gw)] - y[(p, gw)]
                obj_terms.append(decay * bench_weight * bench_var * xp_map[p])

                # 3. Captaincy xP Multiplier + Ceiling Variance Upside + EO Protection Guardrail
                npxg_map = dict(zip(self.df["player_id"], self.df.get("r_npxg", pd.Series([0.0]*len(self.df)))))
                cap_upside = float(npxg_map.get(p, 0.0)) * 0.20 # Prioritizes explosive goalscorers with high haul probability
                eo_boost = (sel_pct_map.get(p, 0.0) / 100.0) * 0.15 if sel_pct_map.get(p, 0.0) >= 50.0 else 0.0
                obj_terms.append(decay * c[(p, gw)] * (xp_map[p] * (1.0 + cap_upside) + eo_boost))
                obj_terms.append(decay * 0.05 * v[(p, gw)] * xp_map[p]) # Tie-breaker Vice-Captain preference

            if not (active_chip in ["wildcard", "freehit"] and gw == self.gws[0]):
                obj_terms.append(-TRANSFER_HIT_COST * hits[gw])

            # Reward holding free transfers in future gameweeks
            obj_terms.append(decay * FREE_TRANSFER_OPTION_VALUE * (fts_state[gw] - 1))

        # Portfolio Covariance Risk Penalty (Double/Triple Defense Stacking Penalty)
        gw1 = self.gws[0]
        eff_risk_aversion = risk_aversion if risk_aversion > 0.0 else 0.25 # Default 0.25 risk penalty
        defenders = [p for p in self.players if self.positions[p] in ["GKP", "DEF"]]
        for i in range(len(defenders)):
            for j in range(i + 1, len(defenders)):
                p1, p2 = defenders[i], defenders[j]
                if self.teams[p1] == self.teams[p2]:
                    w_pair = pulp.LpVariable(f"w_cov_{p1}_{p2}", cat=pulp.LpBinary)
                    prob += w_pair <= y[(p1, gw1)]
                    prob += w_pair <= y[(p2, gw1)]
                    prob += w_pair >= y[(p1, gw1)] + y[(p2, gw1)] - 1
                    obj_terms.append(-eff_risk_aversion * 0.45 * w_pair)

        # Bank Salvage & Total Wealth Dynamics
        initial_market_wealth = sum(self.costs.get(p, 5.0) for p in initial_squad) + initial_bank
        initial_liquid_wealth = sum(selling_costs.get(p, 5.0) for p in initial_squad) + initial_bank
        if max_budget is not None:
            initial_market_wealth = min(initial_market_wealth, float(max_budget))
            initial_liquid_wealth = min(initial_liquid_wealth, float(max_budget))

        bank_salvage = (initial_market_wealth - pulp.lpSum([x[(p, gw1)] * self.costs[p] for p in self.players])) * BANK_SALVAGE_WEIGHT
        obj_terms.append(bank_salvage)

        prob += pulp.lpSum(obj_terms)

        # Locked Players Constraint (e.g. Haaland / Salah Locks)
        if locked_player_ids:
            for locked_id in locked_player_ids:
                if locked_id in self.players:
                    for gw in self.gws:
                        prob += x[(locked_id, gw)] == 1
                        prob += y[(locked_id, gw)] == 1

        # Parse Custom Formation if provided (e.g., '3-4-3' -> DEF:3, MID:4, FWD:3)
        formation_targets = None
        if formation and formation != "Automatic":
            parts = formation.split("-")
            if len(parts) == 3:
                try:
                    formation_targets = {
                        "DEF": int(parts[0]),
                        "MID": int(parts[1]),
                        "FWD": int(parts[2])
                    }
                except ValueError:
                    formation_targets = None

        # Constraints across Horizon
        initial_squad_set = set(initial_squad)
        
        for t_idx, gw in enumerate(self.gws):
            is_gw1 = (t_idx == 0)
            prev_gw = self.gws[t_idx - 1] if t_idx > 0 else None

            # FT State Initialization & Transitions
            if is_gw1:
                prob += fts_state[gw] == initial_fts
            else:
                prev_transfers = pulp.lpSum([tr_in[(p, prev_gw)] for p in self.players])
                if active_chip in ["wildcard", "freehit"] and prev_gw == self.gws[0]:
                    prob += fts_state[gw] == 1
                else:
                    prob += fts_state[gw] == fts_state[prev_gw] - prev_transfers + hits[prev_gw] + 1
                    prob += fts_state[gw] <= MAX_FREE_TRANSFERS

            # Squad and XI Sizes
            prob += pulp.lpSum([x[(p, gw)] for p in self.players]) == SQUAD_SIZE
            prob += pulp.lpSum([y[(p, gw)] for p in self.players]) == 11
            prob += pulp.lpSum([c[(p, gw)] for p in self.players]) == 1
            prob += pulp.lpSum([v[(p, gw)] for p in self.players]) == 1

            for p in self.players:
                prob += y[(p, gw)] <= x[(p, gw)]
                prob += c[(p, gw)] <= y[(p, gw)]
                prob += v[(p, gw)] <= y[(p, gw)]
                prob += c[(p, gw)] + v[(p, gw)] <= 1

            # Squad Continuity & Transfer Balance
            transfers_in_sum = pulp.lpSum([tr_in[(p, gw)] for p in self.players])
            transfers_out_sum = pulp.lpSum([tr_out[(p, gw)] for p in self.players])
            prob += transfers_in_sum == transfers_out_sum

            for p in self.players:
                if is_gw1:
                    was_in_initial = 1 if p in initial_squad_set else 0
                    prob += x[(p, gw)] == was_in_initial + tr_in[(p, gw)] - tr_out[(p, gw)]
                    if was_in_initial:
                        prob += tr_in[(p, gw)] == 0
                    else:
                        prob += tr_out[(p, gw)] == 0
                else:
                    if active_chip == "freehit" and prev_gw == self.gws[0]:
                        was_in_initial = 1 if p in initial_squad_set else 0
                        prob += x[(p, gw)] == was_in_initial + tr_in[(p, gw)] - tr_out[(p, gw)]
                        prob += tr_in[(p, gw)] <= 1 - was_in_initial
                        prob += tr_out[(p, gw)] <= was_in_initial
                    else:
                        prob += x[(p, gw)] == x[(p, prev_gw)] + tr_in[(p, gw)] - tr_out[(p, gw)]
                        prob += tr_in[(p, gw)] <= 1 - x[(p, prev_gw)]
                        prob += tr_out[(p, gw)] <= x[(p, prev_gw)]

            # Hit Penalty Constraints
            if active_chip in ["wildcard", "freehit"] and is_gw1:
                prob += hits[gw] == 0
            else:
                prob += hits[gw] >= transfers_in_sum - fts_state[gw]

            # Position Quotas (15-man squad)
            for pos, quota in SQUAD_QUOTAS.items():
                prob += pulp.lpSum([x[(p, gw)] for p in self.players if self.positions[p] == pos]) == quota

            # Formation Quotas (Starting 11)
            prob += pulp.lpSum([y[(p, gw)] for p in self.players if self.positions[p] == "GKP"]) == 1
            
            if formation_targets:
                prob += pulp.lpSum([y[(p, gw)] for p in self.players if self.positions[p] == "DEF"]) == formation_targets["DEF"]
                prob += pulp.lpSum([y[(p, gw)] for p in self.players if self.positions[p] == "MID"]) == formation_targets["MID"]
                prob += pulp.lpSum([y[(p, gw)] for p in self.players if self.positions[p] == "FWD"]) == formation_targets["FWD"]
            else:
                for pos in ["DEF", "MID", "FWD"]:
                    min_q = XI_QUOTAS_MIN[pos]
                    max_q = XI_QUOTAS_MAX[pos]
                    xi_pos = pulp.lpSum([y[(p, gw)] for p in self.players if self.positions[p] == pos])
                    prob += xi_pos >= min_q
                    prob += xi_pos <= max_q

            # Club Quota (Max 3 per Premier League team)
            for team_name in set(self.teams.values()):
                prob += pulp.lpSum([x[(p, gw)] for p in self.players if self.teams[p] == team_name]) <= MAX_PLAYERS_PER_TEAM

            # Budget Constraints:
            # In GW1: Exact transfer cashflow constraint (Cash spent on buys <= Bank + Cash released from sells at P_sell)
            if is_gw1 and active_chip not in ["wildcard", "freehit"]:
                prob += (
                    pulp.lpSum([tr_in[(q, gw1)] * self.costs[q] for q in self.players])
                    <= initial_bank + pulp.lpSum([tr_out[(p, gw1)] * selling_costs.get(p, self.costs[p]) for p in self.players])
                )
            elif is_gw1 and active_chip in ["wildcard", "freehit"]:
                prob += pulp.lpSum([x[(p, gw)] * self.costs[p] for p in self.players]) <= initial_liquid_wealth
            else:
                prob += pulp.lpSum([x[(p, gw)] * self.costs[p] for p in self.players]) <= initial_market_wealth
            
            # Minimum Squad Cost Constraint (force maximizing budget)
            if min_squad_cost is not None and is_gw1:
                prob += pulp.lpSum([x[(p, gw)] * self.costs[p] for p in self.players]) >= min_squad_cost

        # Solve with multi-solver fallback for Linux and macOS environments
        solver = None
        if os.path.exists("/usr/bin/cbc"):
            try:
                solver = pulp.COIN_CMD(path="/usr/bin/cbc", msg=False, timeLimit=20)
            except Exception:
                solver = None

        if solver is None:
            try:
                solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=20)
            except Exception:
                solver = pulp.LpSolverDefault

        try:
            prob.solve(solver)
        except Exception:
            prob.solve()

        status_str = pulp.LpStatus[prob.status]
        if status_str != "Optimal":
            return {"status": status_str}

        # Extract GW1 Results
        gw1 = self.gws[0]
        opt_squad = [p for p in self.players if pulp.value(x[(p, gw1)]) == 1]
        opt_xi = [p for p in self.players if pulp.value(y[(p, gw1)]) == 1]
        opt_cap = [p for p in self.players if pulp.value(c[(p, gw1)]) == 1]
        opt_vc = [p for p in self.players if pulp.value(v[(p, gw1)]) == 1]
        captain_id = opt_cap[0] if opt_cap else (opt_xi[0] if opt_xi else None)
        vice_captain_id = opt_vc[0] if opt_vc else ([p for p in opt_xi if p != captain_id][0] if len(opt_xi) > 1 else None)

        transfers_out = [p for p in self.players if pulp.value(tr_out[(p, gw1)]) == 1]
        transfers_in = [p for p in self.players if pulp.value(tr_in[(p, gw1)]) == 1]

        if active_chip in ["wildcard", "freehit"]:
            squad_cost = sum(self.costs[p] for p in opt_squad)
            bank_remaining = round(initial_liquid_wealth - squad_cost, 2)
        else:
            cash_spent = sum(self.costs[p] for p in transfers_in)
            cash_gained = sum(selling_costs.get(p, self.costs[p]) for p in transfers_out)
            bank_remaining = round(initial_bank + cash_gained - cash_spent, 2)

        bench = [p for p in opt_squad if p not in opt_xi]
        bench_gkp = [p for p in bench if self.positions[p] == "GKP"]
        starting_defs = [p for p in opt_xi if self.positions[p] == "DEF"]
        
        # In FPL, a team must have min 3 DEFs on the pitch.
        # If starting exactly 3 DEFs, 1st outfield bench MUST be a DEF for legal auto-sub coverage!
        if len(starting_defs) == 3:
            bench_defs = sorted([p for p in bench if self.positions[p] == "DEF"], key=lambda pid: self.df.loc[self.df["player_id"] == pid, f"xP_{gw1}"].values[0] if pid in self.df["player_id"].values else 0, reverse=True)
            bench_others = sorted([p for p in bench if self.positions[p] not in ["GKP", "DEF"]], key=lambda pid: self.df.loc[self.df["player_id"] == pid, f"xP_{gw1}"].values[0] if pid in self.df["player_id"].values else 0, reverse=True)
            if bench_defs:
                ordered_outfield = [bench_defs[0]] + sorted(bench_defs[1:] + bench_others, key=lambda pid: self.df.loc[self.df["player_id"] == pid, f"xP_{gw1}"].values[0] if pid in self.df["player_id"].values else 0, reverse=True)
            else:
                ordered_outfield = bench_others
        else:
            ordered_outfield = sorted([p for p in bench if self.positions[p] != "GKP"], key=lambda pid: self.df.loc[self.df["player_id"] == pid, f"xP_{gw1}"].values[0] if pid in self.df["player_id"].values else 0, reverse=True)
            
        ordered_bench = bench_gkp + ordered_outfield

        projected_pts = round(
            sum(self.df.loc[self.df["player_id"] == p, f"xP_{gw1}"].values[0] for p in opt_xi if p in self.df["player_id"].values) +
            (self.df.loc[self.df["player_id"] == captain_id, f"xP_{gw1}"].values[0] if captain_id in self.df["player_id"].values else 0),
            2
        )

        # Calculate full multi-period horizon projected points
        horizon_pts_total = 0.0
        for t_gw in self.gws:
            t_xi = [p for p in self.players if pulp.value(y[(p, t_gw)]) == 1]
            t_cap = [p for p in self.players if pulp.value(c[(p, t_gw)]) == 1]
            t_cap_id = t_cap[0] if t_cap else (t_xi[0] if t_xi else None)
            t_hits = pulp.value(hits[t_gw]) if active_chip not in ["wildcard", "freehit"] or t_gw != self.gws[0] else 0.0
            
            t_pts = sum(self.df.loc[self.df["player_id"] == p, f"xP_{t_gw}"].values[0] for p in t_xi if p in self.df["player_id"].values)
            if t_cap_id in self.df["player_id"].values:
                t_pts += self.df.loc[self.df["player_id"] == t_cap_id, f"xP_{t_gw}"].values[0]
            t_pts -= (t_hits or 0.0) * TRANSFER_HIT_COST
            horizon_pts_total += t_pts

        return {
            "status": status_str,
            "gameweek": gw1,
            "squad_ids": opt_squad,
            "starting_xi_ids": opt_xi,
            "captain_id": captain_id,
            "vice_captain_id": vice_captain_id,
            "bench_ids": ordered_bench,
            "transfers_out": transfers_out,
            "transfers_in": transfers_in,
            "hits_cost": int(pulp.value(hits[gw1]) * TRANSFER_HIT_COST) if active_chip not in ["wildcard", "freehit"] else 0,
            "bank": bank_remaining,
            "projected_points_gw1": projected_pts,
            "projected_points_horizon": round(horizon_pts_total, 2)
        }

    def evaluate_chip_deltas(self, initial_squad, initial_bank, initial_fts=1, purchase_prices=None):
        """Runs concurrent shadow solves with dynamic gameweek hurdle curves."""
        gw1 = self.gws[0] if self.gws else 1
        std_plan = self.solve_rolling_horizon(initial_squad, initial_bank, initial_fts, active_chip=None, purchase_prices=purchase_prices)
        wc_plan = self.solve_rolling_horizon(initial_squad, initial_bank, initial_fts, active_chip="wildcard", purchase_prices=purchase_prices)
        fh_plan = self.solve_rolling_horizon(initial_squad, initial_bank, initial_fts, active_chip="freehit", purchase_prices=purchase_prices)

        pts_std_gw1 = std_plan.get("projected_points_gw1", 0.0)
        pts_fh_gw1 = fh_plan.get("projected_points_gw1", 0.0)

        pts_std_horizon = std_plan.get("projected_points_horizon", pts_std_gw1 * len(self.gws))
        pts_wc_horizon = wc_plan.get("projected_points_horizon", pts_std_gw1 * len(self.gws))

        # Wildcard benefits accumulate over the full 6-GW horizon; Free Hit benefits are purely 1-GW
        delta_wc = round(pts_wc_horizon - pts_std_horizon, 2)
        delta_fh = round(pts_fh_gw1 - pts_std_gw1, 2)

        # Dynamic time-decayed hurdle curve (prevents greedy early burning in early gameweeks)
        hurdle_wc = max(12.0, round(18.0 - 0.20 * gw1, 1))
        hurdle_fh = 16.0

        rec = "HOLD CHIPS"
        if delta_fh >= hurdle_fh:
            rec = "🔥 PLAY FREE HIT"
        elif delta_wc >= hurdle_wc:
            rec = "🔥 PLAY WILDCARD"

        return {
            "standard_plan": std_plan,
            "wildcard_plan": wc_plan,
            "freehit_plan": fh_plan,
            "delta_wildcard": delta_wc,
            "delta_freehit": delta_fh,
            "hurdle_wildcard": hurdle_wc,
            "hurdle_freehit": hurdle_fh,
            "recommendation": rec
        }
