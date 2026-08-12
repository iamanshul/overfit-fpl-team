# -*- coding: utf-8 -*-
"""
Head-to-Head Empirical Model Benchmark: Dixon-Coles Bivariate Poisson vs. Legacy Linear Elo (compare_models.py)
Fetches historical ClubElo ratings per gameweek date and evaluates prediction accuracy and walk-forward MILP solver points.
"""

import os
import sys
import datetime
import sqlite3
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_player_history, fetch_clubelo_ratings, get_db_connection
from rate_engine import CanonicalRateEngine
from optimizer import MultiPeriodMILP

def get_gw_kickoff_dates():
    """Gets kickoff date string (YYYY-MM-DD) per gameweek from SQLite fixtures table."""
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT event as gameweek, MIN(kickoff_time) as first_kickoff FROM fixtures WHERE finished = 1 GROUP BY event", conn)
        dates = {}
        for _, row in df.iterrows():
            gw = int(row['gameweek'])
            raw_ts = str(row['first_kickoff'])
            date_str = raw_ts.split('T')[0] if 'T' in raw_ts else raw_ts.split()[0]
            dates[gw] = date_str
        return dates
    finally:
        conn.close()

def run_head_to_head_benchmark(start_gw=2, end_gw=10):
    print("=" * 70)
    print(f"📊 RUNNING HEAD-TO-HEAD MODEL BENCHMARK (GW {start_gw} -> GW {end_gw})")
    print("   Comparing: [Model A] Legacy Linear Elo vs. [Model B] Dixon-Coles Poisson Engine")
    print("=" * 70)

    history_df = load_player_history()
    if history_df.empty:
        print("❌ Error: No historical player data found.")
        return

    gw_dates = get_gw_kickoff_dates()

    # Initial squad setup
    from squad_manager import build_gw1_start_of_season_squad
    squad_a, bank_a = build_gw1_start_of_season_squad(history_df, budget=100.0)
    squad_ids_a = squad_a["player_id"].tolist() if not squad_a.empty else []
    squad_ids_b = list(squad_ids_a)
    bank_b = bank_a
    fts_a, fts_b = 1, 1

    history_log = []

    for gw in range(start_gw, end_gw + 1):
        history_slice = history_df[history_df["gameweek"] < gw].copy()
        if history_slice.empty:
            continue

        # Get historical ClubElo ratings for this exact gameweek date
        gw_date = gw_dates.get(gw, (datetime.date.today() - datetime.timedelta(days=1)).isoformat())
        print(f"\n🗓️ Gameweek {gw:02d} (Kickoff Date: {gw_date}) | Fetching Historical ClubElo...")
        elo_dict = fetch_clubelo_ratings(target_date=gw_date)

        engine = CanonicalRateEngine(history_slice)

        # Generate matrices
        matrix_a = engine.generate_horizon_matrix(start_gw=gw, horizon_weeks=4, elo_dict=elo_dict, use_dixon_coles=False)
        matrix_b = engine.generate_horizon_matrix(start_gw=gw, horizon_weeks=4, elo_dict=elo_dict, use_dixon_coles=True)

        # Solve Model A (Linear Elo)
        opt_a = MultiPeriodMILP(matrix_a)
        plan_a = opt_a.solve_rolling_horizon(squad_ids_a, bank_a, initial_fts=fts_a)

        # Solve Model B (Dixon-Coles)
        opt_b = MultiPeriodMILP(matrix_b)
        plan_b = opt_b.solve_rolling_horizon(squad_ids_b, bank_b, initial_fts=fts_b)

        # Actual Points in Gameweek
        actual_gw = history_df[history_df["gameweek"] == gw]
        actual_pts_map = dict(zip(actual_gw["player_id"], actual_gw["total_points"]))
        actual_mins_map = dict(zip(actual_gw["player_id"], actual_gw["minutes"]))

        def evaluate_plan(plan, curr_squad):
            if plan.get("status") != "Optimal":
                xi = curr_squad[:11]
                cap = xi[0] if xi else None
                hits = 0
                t_made = 0
                bank_rem = 0.5
            else:
                xi = plan["starting_xi_ids"]
                cap = plan["captain_id"]
                hits = plan["hits_cost"]
                t_made = len(plan["transfers_in"])
                curr_squad = plan["squad_ids"]
                bank_rem = plan["bank"]

            pts = sum(actual_pts_map.get(p, 0) for p in xi if actual_mins_map.get(p, 0) > 0)
            if cap and actual_mins_map.get(cap, 0) > 0:
                pts += actual_pts_map.get(cap, 0) # 2x Captain
            pts -= hits
            return pts, curr_squad, bank_rem, t_made, hits

        pts_a, squad_ids_a, bank_a, t_a, h_a = evaluate_plan(plan_a, squad_ids_a)
        pts_b, squad_ids_b, bank_b, t_b, h_b = evaluate_plan(plan_b, squad_ids_b)

        fts_a = min(5, max(0, fts_a - t_a) + 1)
        fts_b = min(5, max(0, fts_b - t_b) + 1)

        # Clean Sheet Prediction Error Analysis (Defense CS MAE)
        actual_cs_map = dict(zip(actual_gw["player_id"], actual_gw["clean_sheets"]))
        def_players = matrix_a[matrix_a["position"].isin(["GKP", "DEF"])]["player_id"].tolist()

        err_a, err_b = [], []
        for pid in def_players:
            if pid in actual_cs_map:
                actual_cs = actual_cs_map[pid]
                row_a = matrix_a[matrix_a["player_id"] == pid].iloc[0]
                row_b = matrix_b[matrix_b["player_id"] == pid].iloc[0]

                # Estimated CS probability
                cs_prob_a = row_a.get(f"xP_{gw}", 4.0) / 4.0 # Proxy estimation
                cs_prob_b = row_b.get(f"xP_{gw}", 4.0) / 4.0

                err_a.append(abs(actual_cs - min(cs_prob_a, 1.0)))
                err_b.append(abs(actual_cs - min(cs_prob_b, 1.0)))

        mae_a = np.mean(err_a) if err_a else 0.0
        mae_b = np.mean(err_b) if err_b else 0.0

        print(f"   Model A (Linear Elo)   : {pts_a:3d} pts | Trans: {t_a} (Hit: -{h_a}) | CS MAE: {mae_a:.3f}")
        print(f"   Model B (Dixon-Coles)  : {pts_b:3d} pts | Trans: {t_b} (Hit: -{h_b}) | CS MAE: {mae_b:.3f}")

        history_log.append({
            "gameweek": gw,
            "date": gw_date,
            "pts_linear_elo": pts_a,
            "pts_dixon_coles": pts_b,
            "mae_cs_linear": mae_a,
            "mae_cs_dixon_coles": mae_b
        })

    benchmark_df = pd.DataFrame(history_log)
    if not benchmark_df.empty:
        total_a = benchmark_df["pts_linear_elo"].sum()
        total_b = benchmark_df["pts_dixon_coles"].sum()
        mean_mae_a = benchmark_df["mae_cs_linear"].mean()
        mean_mae_b = benchmark_df["mae_cs_dixon_coles"].mean()

        print("\n" + "=" * 70)
        print("🏆 FINAL BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        print(f"Model A (Linear Elo) Total Points        : {total_a} pts (Clean Sheet MAE: {mean_mae_a:.4f})")
        print(f"Model B (Dixon-Coles Poisson) Total Points : {total_b} pts (Clean Sheet MAE: {mean_mae_b:.4f})")
        print("=" * 70)
        
        diff = total_b - total_a
        if diff > 0:
            print(f"🔥 Dixon-Coles Poisson Engine OUTPERFORMED Linear Elo by +{diff} pts (+{diff/len(benchmark_df):.2f} pts/GW)!")
        elif diff < 0:
            print(f"⚠️ Linear Elo OUTPERFORMED Dixon-Coles by {abs(diff)} pts.")
        else:
            print("🤝 Both models tied on total points.")

    return benchmark_df

if __name__ == "__main__":
    run_head_to_head_benchmark(start_gw=2, end_gw=10)
