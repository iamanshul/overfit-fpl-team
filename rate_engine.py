# -*- coding: utf-8 -*-
"""
Canonical Rate Decomposition & Joint Monte Carlo BPS Engine (rate_engine.py)
Models expected minutes (xM), npxG, xA, clean sheets, joint BPS expectations,
2025/26 CBIT/CBIRT defensive bonus rules, and devigged sharp betting odds.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from config import ROLLING_HORIZON_WEEKS, HORIZON_DECAY_FACTOR
from devig_engine import SharpOddsEngine, BivariateDixonColes

class CanonicalRateEngine:
    """
    Decomposes FPL points into fundamental component rates:
    xP = xM/90 * [ r_app + r_goal * P_goal + r_assist * P_assist + r_CS * P_CS + E[Bonus] + E[Def_Bonus] - r_card ]
    Incorporates 2025/26 CBIT (+2 pts for 10+ DEF tackles/blocks/interceptions/clearances)
    and CBIRT (+2 pts for 12+ MID/FWD CBIT + recoveries) defensive bonus rules.
    """

    def __init__(self, history_df, ml_rates_df=None):
        self.history = history_df.copy() if not history_df.empty else pd.DataFrame()
        self.ml_rates = ml_rates_df.copy() if ml_rates_df is not None and not ml_rates_df.empty else pd.DataFrame()
        self.devig_eng = SharpOddsEngine()
        self.player_rates = pd.DataFrame()
        self._preprocess_rates()

    def _preprocess_rates(self):
        """Computes underlying rate metrics per player with minute-weighted Bayesian shrinkage and FPL caps."""
        df = self.history
        if df.empty:
            return

        if "name" not in df.columns and "player_name" in df.columns:
            df["name"] = df["player_name"]
        if "team" not in df.columns and "team_name" in df.columns:
            df["team"] = df["team_name"]
        if "gameweek" not in df.columns and "round" in df.columns:
            df["gameweek"] = df["round"]
        if "player_id" not in df.columns and "element_id" in df.columns:
            df["player_id"] = df["element_id"]

        df = df.sort_values(["player_id", "gameweek"])
        grouped = df.groupby("player_id")

        rates = pd.DataFrame()
        rates["name"] = grouped["name"].last() if "name" in df.columns else grouped["player_name"].last()
        rates["position"] = grouped["position"].last()
        rates["team"] = grouped["team"].last() if "team" in df.columns else grouped["team_name"].last()
        if "now_cost" in df.columns:
            rates["cost"] = grouped["now_cost"].last().astype(float) / 10.0
        elif "cost" in df.columns:
            raw_c = grouped["cost"].last().astype(float)
            rates["cost"] = np.where(raw_c > 20.0, raw_c / 10.0, raw_c)
        elif "value" in df.columns:
            rates["cost"] = grouped["value"].last().astype(float) / 10.0
        else:
            rates["cost"] = 5.0

        # Preserve metadata fields & classify availability categories
        if "chance_of_playing" in df.columns:
            rates["chance_of_playing"] = pd.to_numeric(grouped["chance_of_playing"].last(), errors="coerce").fillna(100)
        else:
            rates["chance_of_playing"] = 100.0
        rates["news"] = grouped["news"].last().fillna("") if "news" in df.columns else ""
        rates["status"] = grouped["status"].last().fillna("a") if "status" in df.columns else "a"
        rates["penalties_order"] = grouped["penalties_order"].last().fillna(0) if "penalties_order" in df.columns else 0
        rates["has_midweek_uefa"] = grouped["has_midweek_uefa"].last().fillna(0) if "has_midweek_uefa" in df.columns else 0
        rates["age"] = pd.to_numeric(grouped["age"].last(), errors="coerce").fillna(26.0) if "age" in df.columns else 26.0
        rates["height_cm"] = pd.to_numeric(grouped["height_cm"].last(), errors="coerce").fillna(182.0) if "height_cm" in df.columns else 182.0

        # Availability category classification
        news_text = rates["news"].str.lower()
        transfer_flag = news_text.str.contains("transfer|leaving|joined|loan|depart|out of favor|not in squad|sold", regex=True, na=False)
        rates["is_permanent_out"] = transfer_flag | rates["status"].isin(["u", "n"])
        rates["is_suspended"] = (rates["status"] == "s") | news_text.str.contains("suspended|banned", regex=True, na=False)
        rates["is_injured"] = (rates["status"] == "i") | (rates["chance_of_playing"] == 0)
        rates["is_long_term_injury"] = rates["is_injured"] & news_text.str.contains("season|months|acl|cruciate|achilles|surgery|long-term", regex=True, na=False)

        # Total minutes & sample size reliability factor
        rates["total_mins"] = grouped["minutes"].sum()
        rates["starts"] = (df["minutes"] >= 45).groupby(df["player_id"]).sum()
        rates["yellow_card_accumulation"] = grouped["yellow_cards"].sum() if "yellow_cards" in df.columns else 0

        # Weighted EWMA expected minutes (xM)
        def ewma(series, span=6):
            return series.ewm(span=span, min_periods=1).mean().iloc[-1]

        rates["start_prob"] = grouped["minutes"].apply(lambda s: ewma((s >= 45).astype(float), span=6))
        rates["recent_mins"] = grouped["minutes"].apply(lambda s: ewma(s, span=6))

        # Two-Part Hurdle Expected Minutes Model (P(Start)*E[M|Start] + P(Sub)*E[M|Sub])
        p_start_base = np.clip(rates["start_prob"] * (rates["chance_of_playing"] / 100.0), 0.0, 1.0)
        e_mins_start = np.where(p_start_base >= 0.5, np.maximum(rates["recent_mins"], 75.0), 70.0)
        p_sub_in = np.where(p_start_base < 0.85, (1.0 - p_start_base) * 0.55, 0.0)
        e_mins_sub = 18.0

        raw_xm = p_start_base * e_mins_start + p_sub_in * e_mins_sub
        
        # Apply Bayesian prior shrinkage for low-sample players instead of a hard penalty
        is_low_sample = (rates["total_mins"] < 360.0)
        prior_xm = np.where(rates["position"] == "GKP", 90.0, np.where(rates["cost"] >= 7.0, 75.0, 60.0))
        weight_obs = np.clip(rates["total_mins"] / 360.0, 0.0, 1.0)
        
        rates["xM"] = np.where(
            is_low_sample,
            weight_obs * raw_xm + (1.0 - weight_obs) * prior_xm,
            raw_xm
        )
        rates["xM"] = np.clip(rates["xM"], 0.0, 90.0)

        # Yellow Card Suspension Warning (4 cards = elevated 0.90x xM penalty)
        rates["on_suspension_warning"] = (rates["yellow_card_accumulation"] == 4)
        rates["xM"] = np.where(rates["on_suspension_warning"], rates["xM"] * 0.90, rates["xM"])

        # Midweek UEFA European Congestion & Age Fatigue Interaction
        # Players aged >= 30 suffer progressive recovery decay during <72h turnaround
        age_fatigue_mult = np.clip(1.0 - np.maximum(0.0, rates["age"] - 29.0) * 0.035, 0.70, 1.0)
        rates["xM"] = np.where(rates["has_midweek_uefa"] == 1, rates["xM"] * 0.82 * age_fatigue_mult, rates["xM"])

        # Sub-60 minute early substitution hazard
        sub_60_h = np.zeros(len(rates), dtype=float)
        sub_60_h = np.where(rates["yellow_card_accumulation"] >= 4, sub_60_h + 0.10, sub_60_h)
        sub_60_h = np.where(rates["chance_of_playing"] < 100, sub_60_h + 0.15, sub_60_h)
        sub_60_h = np.where((rates["has_midweek_uefa"] == 1) & (rates["age"] >= 30), sub_60_h + 0.12, sub_60_h)
        rates["sub_60_hazard"] = np.clip(sub_60_h, 0.0, 0.45)


        # Marquee Starter Baseline xM Floor: High-cost active players (£7.0m+ MID/FWD, £5.5m+ DEF/GKP)
        is_marquee_starter = (rates["chance_of_playing"] >= 75) & (rates["status"] == "a") & (
            (rates["position"].isin(["MID", "FWD"]) & (rates["cost"] >= 7.0)) |
            (rates["position"].isin(["GKP", "DEF"]) & (rates["cost"] >= 5.5))
        )
        rates["xM"] = np.where(is_marquee_starter, np.maximum(rates["xM"], 75.0), rates["xM"])

        # Minute-weighted event rates (Goals & Assists per 90)
        tot_mins_safe = np.maximum(rates["total_mins"], 90.0)
        tot_goals = grouped["goals_scored"].sum()
        tot_assists = grouped["assists"].sum()
        tot_cards = (grouped["yellow_cards"].sum() + 3 * grouped["red_cards"].sum()) if "yellow_cards" in df.columns else 0
        tot_bonus = grouped["bonus"].sum()

        raw_r_goal = (tot_goals / tot_mins_safe) * 90.0
        raw_r_assist = (tot_assists / tot_mins_safe) * 90.0
        raw_r_cards = (tot_cards / tot_mins_safe) * 90.0
        raw_r_bonus = (tot_bonus / tot_mins_safe) * 90.0

        # 2025/26 Defensive Action Rate Estimators (CBIT for DEFs, CBIRT for MIDs)
        tot_bps = grouped["bps"].sum() if "bps" in df.columns else grouped["minutes"].sum() * 0.2
        tot_influence = grouped["influence"].sum() if "influence" in df.columns else grouped["minutes"].sum() * 0.1
        
        # Estimate per-90 defensive activity index (CBIT & CBIRT)
        raw_r_cbit = np.where(
            rates["position"] == "DEF",
            np.clip((tot_bps / tot_mins_safe) * 0.45 + (tot_influence / tot_mins_safe) * 0.15 + 4.0, 4.0, 14.0),
            np.clip((tot_bps / tot_mins_safe) * 0.30 + (tot_influence / tot_mins_safe) * 0.10 + 2.0, 2.0, 10.0)
        )
        
        raw_r_cbirt = np.where(
            rates["position"] == "MID",
            np.clip((tot_bps / tot_mins_safe) * 0.50 + (tot_influence / tot_mins_safe) * 0.20 + 3.0, 3.0, 15.0),
            raw_r_cbit
        )

        # Positional Bayesian Prior Shrinkage (weight prior vs sample size)
        prior_weight = 360.0
        weight_obs = rates["total_mins"] / (rates["total_mins"] + prior_weight)

        pos_prior_goal = np.where(rates["position"] == "FWD", 0.40, np.where(rates["position"] == "MID", 0.18, 0.05))
        pos_prior_assist = np.where(rates["position"] == "FWD", 0.15, np.where(rates["position"] == "MID", 0.18, 0.08))

        rates["r_goal"] = np.clip(weight_obs * raw_r_goal + (1.0 - weight_obs) * pos_prior_goal, 0.0, 0.95)
        rates["r_assist"] = np.clip(weight_obs * raw_r_assist + (1.0 - weight_obs) * pos_prior_assist, 0.0, 0.70)

        # Set-Piece & Corner Dead-Ball Delivery Assist Boost
        is_set_piece_taker = (rates.get("corners_and_indirect_freekicks_order", 0) == 1) | (rates.get("direct_freekicks_order", 0) == 1)
        rates["r_assist"] = np.where(is_set_piece_taker, rates["r_assist"] + 0.06, rates["r_assist"])
        
        # Penalty Duty Decomposition: API penalties_order == 1 or known_pen_takers list
        known_pen_takers = ["Haaland", "Palmer", "Saka", "B.Fernandes", "Salah", "Isak", "Watkins", "Solanke", "Son", "Gyökeres", "Mateta", "Mbeumo", "Wood", "João Pedro", "Bruno G.", "Eze"]
        rates["is_pen_taker"] = (rates["penalties_order"] == 1) | rates["name"].isin(known_pen_takers) | (rates["r_goal"] > 0.42)
        rates["p_pen_90"] = np.where(rates["is_pen_taker"], 0.15, 0.0)
        rates["r_npxg"] = np.maximum(0.0, rates["r_goal"] - rates["p_pen_90"] * 0.79)

        # Merge ML Estimator Predictions (npxG90, xA90, and P(Start)) if provided
        if not self.ml_rates.empty:
            ml_map_npxg = dict(zip(self.ml_rates['player_id'], self.ml_rates.get('pred_npxg90', [])))
            ml_map_xa = dict(zip(self.ml_rates['player_id'], self.ml_rates.get('pred_xa90', [])))
            ml_map_start = dict(zip(self.ml_rates['player_id'], self.ml_rates.get('pred_p_start', [])))
            
            pids = rates.index
            ml_npxg_vals = pids.map(ml_map_npxg)
            ml_xa_vals = pids.map(ml_map_xa)
            ml_start_vals = pids.map(ml_map_start)
            
            rates["r_npxg"] = np.where(pd.notna(ml_npxg_vals), ml_npxg_vals, rates["r_npxg"])
            rates["r_assist"] = np.where(pd.notna(ml_xa_vals), ml_xa_vals, rates["r_assist"])
            
            # Wire LightGBM P(Start) into xM calculation
            if pd.notna(ml_start_vals).any():
                ml_p_start = np.where(pd.notna(ml_start_vals), ml_start_vals, rates["start_prob"])
                rates["xM"] = np.clip(rates["xM"] * (0.5 + 0.5 * ml_p_start), 0.0, 90.0)

        # Aerial Dominance & Set-Piece Threat for tall players (>= 188cm)
        is_tall = (rates["height_cm"] >= 188.0)
        rates["r_npxg"] = np.where(is_tall & rates["position"].isin(["DEF", "FWD"]), rates["r_npxg"] + 0.04, rates["r_npxg"])
        rates["r_cbit"] = np.where(is_tall & (rates["position"] == "DEF"), raw_r_cbit + 1.2, raw_r_cbit)

        # Price Rise Velocity & Wealth Momentum (Net Transfer Flow)
        transfers_net = df.groupby("player_id")["transfers_balance"].last() if "transfers_balance" in df.columns else pd.Series(0, index=rates.index)
        rates["price_momentum"] = np.clip(rates.index.map(transfers_net).fillna(0.0) / 100000.0 * 0.1, -0.2, 0.2)

        rates["r_bonus_base"] = np.clip(raw_r_bonus, 0.0, 1.5)
        rates["r_cards"] = np.clip(raw_r_cards, 0.0, 0.50)
        rates["r_cbirt"] = raw_r_cbirt


        team_cs = df[df["minutes"] >= 60].groupby("team")["clean_sheets"].mean() if "clean_sheets" in df.columns else pd.Series()
        rates["team_cs_rate"] = rates["team"].map(team_cs).fillna(0.25)

        self.player_rates = rates.reset_index()

    def simulate_joint_bps(self, team_df, fix_map=None, n_sims=300):
        """
        Match-by-Match Joint Monte Carlo Bonus Point System (BPS) simulator.
        Pairs opposing teams in each fixture and allocates 3, 2, 1 bonus points per match across all 22 players.
        """
        if team_df.empty:
            return pd.Series(dtype=float)

        all_bonus = pd.Series(0.0, index=team_df.index)
        processed_teams = set()

        for team_name, group in team_df.groupby("team"):
            if team_name in processed_teams:
                continue

            # Identify match opponent from fixture schedule
            opp_name = None
            if fix_map and team_name in fix_map and len(fix_map[team_name]) > 0:
                opp_name = fix_map[team_name][0].get('opp')

            if opp_name and opp_name in team_df["team"].values and opp_name not in processed_teams:
                match_group = team_df[team_df["team"].isin([team_name, opp_name])]
                processed_teams.add(team_name)
                processed_teams.add(opp_name)
            else:
                match_group = group
                processed_teams.add(team_name)

            pids = match_group["player_id"].values
            chances = match_group.get("chance_of_playing", pd.Series([100]*len(match_group))).fillna(100)
            avail_factor = np.clip(chances / 100.0, 0.0, 1.0).values
            effective_xM = match_group["xM"].values * avail_factor

            p_goals = (match_group["r_goal"].values * (effective_xM / 90.0))
            p_assists = (match_group["r_assist"].values * (effective_xM / 90.0))

            # Sample Poisson events
            sim_goals = np.random.poisson(np.tile(p_goals, (n_sims, 1)))
            sim_assists = np.random.poisson(np.tile(p_assists, (n_sims, 1)))

            # Micro-Action BPS Decomposition by position and defensive action volume
            base_bps = np.where(
                match_group["position"].values == "DEF",
                match_group["r_cbit"].values * 1.8 + (effective_xM / 90.0) * 12.0,
                np.where(
                    match_group["position"].values == "MID",
                    match_group["r_cbirt"].values * 1.2 + (effective_xM / 90.0) * 8.0,
                    (effective_xM / 90.0) * 5.0
                )
            )
            sim_bps = sim_goals * 24 + sim_assists * 9 + base_bps + np.random.normal(0, 3, size=(n_sims, len(pids)))
            
            zero_mask = (effective_xM <= 0)
            sim_bps[:, zero_mask] = -999.0

            bonus_awarded = np.zeros_like(sim_bps, dtype=float)
            k = min(3, len(pids))
            for i in range(n_sims):
                top_idx = np.argsort(sim_bps[i])[-k:]
                if k >= 1: bonus_awarded[i, top_idx[-1]] += 3.0
                if k >= 2: bonus_awarded[i, top_idx[-2]] += 2.0
                if k >= 3: bonus_awarded[i, top_idx[-3]] += 1.0

            bonus_awarded[:, zero_mask] = 0.0
            exp_bonus = np.mean(bonus_awarded, axis=0)
            all_bonus.loc[match_group.index] = exp_bonus

        return all_bonus


    def _get_fixtures_map(self, gw):
        """Retrieves dict mapping team_name -> list of fixture dicts for gameweek gw."""
        try:
            from data_loader import get_db_connection, DB_PATH
            import os
            if not os.path.exists(DB_PATH):
                return {}
            conn = get_db_connection()
            try:
                df = pd.read_sql("""
                    SELECT f.event, f.team_h, f.team_a, th.name as team_h_name, ta.name as team_a_name
                    FROM fixtures f
                    LEFT JOIN teams th ON f.team_h = th.id
                    LEFT JOIN teams ta ON f.team_a = ta.id
                    WHERE f.event = ?
                """, conn, params=(gw,))
                
                fix_map = {}
                for _, row in df.iterrows():
                    h_name = row['team_h_name']
                    a_name = row['team_a_name']
                    if h_name:
                        fix_map.setdefault(h_name, []).append({'opp': a_name, 'was_home': True})
                    if a_name:
                        fix_map.setdefault(a_name, []).append({'opp': h_name, 'was_home': False})
                return fix_map
            except Exception:
                return {}
            finally:
                conn.close()
        except Exception:
            return {}

    def generate_horizon_matrix(self, start_gw, horizon_weeks=ROLLING_HORIZON_WEEKS, elo_dict=None, sharp_odds_df=None, use_dixon_coles=True):
        """Generates expected points matrix (xP) across rolling horizon with schedule awareness & Dixon-Coles Poisson engine."""
        if self.player_rates.empty:
            return pd.DataFrame()

        matrix = self.player_rates.copy()
        
        for w in range(horizon_weeks):
            gw = start_gw + w
            decay = HORIZON_DECAY_FACTOR ** w
            fix_map = self._get_fixtures_map(gw)

            teams = matrix["team"].unique()
            att_mult_map = {}
            def_mult_map = {}
            is_blank_map = {}
            dgw_count_map = {}

            for t in teams:
                fixtures_for_team = fix_map.get(t, [])
                if not fixtures_for_team:
                    # Blank Gameweek (BGW)
                    att_mult_map[t] = 0.0
                    def_mult_map[t] = 0.0
                    is_blank_map[t] = True
                    dgw_count_map[t] = 0
                else:
                    is_blank_map[t] = False
                    dgw_count_map[t] = len(fixtures_for_team)
                    att_sum = 0.0
                    def_sum = 0.0

                    for fix in fixtures_for_team:
                        opp_t = fix['opp']
                        was_home = fix['was_home']
                        
                        if elo_dict and t in elo_dict and opp_t in elo_dict:
                            try:
                                t_elo = float(elo_dict[t]) + (60.0 if was_home else 0.0)
                                opp_elo = float(elo_dict[opp_t]) + (0.0 if was_home else 60.0)
                                net_delta = t_elo - opp_elo

                                if use_dixon_coles:
                                    att_m = np.clip(np.exp(net_delta / 450.0), 0.5, 2.0)
                                    def_m = np.clip(np.exp(-net_delta / 450.0), 0.4, 1.8)
                                else:
                                    lin_m = round(1.0 + net_delta / 1000.0, 2)
                                    att_m = lin_m
                                    def_m = np.clip(2.0 - lin_m, 0.4, 1.8)
                            except (ValueError, TypeError):
                                att_m, def_m = 1.0, 1.0
                        else:
                            seed_val = (hash(t) + gw) % 5
                            diff_map = {0: 1.20, 1: 1.10, 2: 1.00, 3: 0.90, 4: 0.80}
                            att_m = diff_map[seed_val]
                            def_m = 2.0 - att_m

                        att_sum += att_m
                        def_sum += def_m

                    att_mult_map[t] = att_sum
                    def_mult_map[t] = def_sum

            att_mult = matrix["team"].map(att_mult_map).fillna(1.0)
            def_vulnerability = matrix["team"].map(def_mult_map).fillna(1.0)
            is_blank = matrix["team"].map(is_blank_map).fillna(False)

            # Dynamic Gameweek-Specific Availability Factor A_{p, w}
            avail_factor = np.ones(len(matrix), dtype=float)
            
            # 1. Permanent Ineligibility (transferred out of league / not in squad): 0% across all horizon weeks
            if "is_permanent_out" in matrix.columns:
                avail_factor = np.where(matrix["is_permanent_out"].values, 0.0, avail_factor)
            
            # 2. Suspensions (status == 's'): 0% in GW1 (w=0), but 100% available in GW2+ (w >= 1)
            if "is_suspended" in matrix.columns:
                avail_factor = np.where(matrix["is_suspended"].values & (w == 0), 0.0, avail_factor)
                
            # 3. Long-term catastrophic injury (season-ending, ACL, surgery): 0% across all horizon weeks
            if "is_long_term_injury" in matrix.columns:
                avail_factor = np.where(matrix["is_long_term_injury"].values, 0.0, avail_factor)
                
            # 4. Standard Non-Long-Term Injury (status == 'i' or 0% chance): progressive recovery
            if "is_injured" in matrix.columns:
                injury_recovery_curve = [0.0, 0.25, 0.55, 0.80, 0.95, 1.00]
                rec_val = injury_recovery_curve[min(w, len(injury_recovery_curve)-1)]
                is_perm = matrix["is_permanent_out"].values if "is_permanent_out" in matrix.columns else False
                is_susp = matrix["is_suspended"].values if "is_suspended" in matrix.columns else False
                is_lt = matrix["is_long_term_injury"].values if "is_long_term_injury" in matrix.columns else False
                std_injury_mask = matrix["is_injured"].values & (~is_lt) & (~is_perm) & (~is_susp)
                avail_factor = np.where(std_injury_mask, rec_val, avail_factor)
                
            # 5. Short-term Doubtful / Knock (0 < initial chance < 100): exponential recovery trajectory
            c0 = np.clip(pd.to_numeric(matrix.get("chance_of_playing", 100), errors="coerce").fillna(100).values / 100.0, 0.0, 1.0)
            doubtful_mask = (c0 > 0.0) & (c0 < 1.0)
            if "is_permanent_out" in matrix.columns:
                doubtful_mask = doubtful_mask & (~matrix["is_permanent_out"].values)
            knock_recovery = 1.0 - (1.0 - c0) * np.exp(-1.2 * w)
            avail_factor = np.where(doubtful_mask, knock_recovery, avail_factor)
            
            xM = matrix["xM"].values * avail_factor
            p90_factor = xM / 90.0

            # Sigmoidal cumulative probability P(Mins >= 60) adjusted for sub-60 hazard cliff
            hazard_factor = 1.0 - matrix["sub_60_hazard"].values if "sub_60_hazard" in matrix.columns else 1.0
            p_60_mins = (1.0 / (1.0 + np.exp(-(xM - 60.0) / 4.0))) * hazard_factor
            p_60_mins = np.where(xM < 40.0, 0.0, np.clip(p_60_mins, 0.0, 1.0))

            # Component 1: Appearance Points (FPL rules: 1 pt for 1-59 mins, +1 pt for 60+ mins, multiplied by fixture count in DGW)
            p_1_min = np.where(xM < 1.0, 0.0, 1.0 / (1.0 + np.exp(-(xM - 15.0) / 5.0)))
            dgw_mult = matrix["team"].map(dgw_count_map).fillna(1.0).values
            xp_app = (p_1_min * 1.0 + p_60_mins * 1.0) * dgw_mult

            # Component 2: Attacking Returns (Dirichlet / Softmax Team Goal Share Normalization)
            pts_goal = np.where(matrix["position"] == "FWD", 4.0, np.where(matrix["position"] == "MID", 5.0, 6.0))
            
            # Baseline team goal expectation based on fixture difficulty / Dixon-Coles
            base_team_goals = np.clip(1.35 * att_mult, 0.0, 4.5 * np.maximum(dgw_mult, 1.0))

            # Softmax / Dirichlet Normalized Attacking Shares (guarantees exact sum to team capacity)
            tau = 0.25
            raw_npxg_safe = np.maximum(matrix["r_npxg"].values, 0.01)
            raw_xa_safe = np.maximum(matrix["r_assist"].values, 0.01)

            matrix["_temp_w_npxg"] = np.exp(raw_npxg_safe / tau) * p90_factor
            matrix["_temp_w_xa"] = np.exp(raw_xa_safe / tau) * p90_factor

            team_w_npxg_sum = matrix.groupby("team")["_temp_w_npxg"].transform("sum")
            team_w_xa_sum = matrix.groupby("team")["_temp_w_xa"].transform("sum")

            npxg_share = np.where(team_w_npxg_sum > 0, matrix["_temp_w_npxg"] / np.maximum(team_w_npxg_sum, 1e-6), 0.0)
            xa_share = np.where(team_w_xa_sum > 0, matrix["_temp_w_xa"] / np.maximum(team_w_xa_sum, 1e-6), 0.0)

            cond_npxg = base_team_goals * npxg_share
            pen_goals = matrix["p_pen_90"] * p90_factor * 0.79 * att_mult
            exp_goals = cond_npxg + pen_goals
            exp_assists = base_team_goals * 0.75 * xa_share

            xp_att = exp_goals * pts_goal + exp_assists * 3.0
            matrix.drop(columns=["_temp_w_npxg", "_temp_w_xa"], inplace=True)

            # Component 3: Clean Sheet Probability (Scaled per match in DGW)
            pts_cs = np.where(matrix["position"].isin(["GKP", "DEF"]), 4.0, np.where(matrix["position"] == "MID", 1.0, 0.0))
            
            if use_dixon_coles:
                # Bivariate Dixon-Coles Clean Sheet Probability P(CS) with coupling parameter rho = -0.11
                mu_conceded_per_match = 1.25 * (def_vulnerability.values / np.maximum(dgw_mult, 1.0))
                # Coupling factor tau(0,0) = 1 - lambda * mu * rho increases 0-0 probability slightly in low-scoring games
                biv_cs_prob = np.clip(np.exp(-mu_conceded_per_match) * (1.0 + 0.11 * mu_conceded_per_match), 0.05, 0.65)
                prob_cs_per_match = np.where(is_blank, 0.0, biv_cs_prob)
                prob_cs = prob_cs_per_match * dgw_mult
            else:
                # Legacy Linear Elo Clean Sheet probability
                prob_cs = np.where(is_blank, 0.0, matrix["team_cs_rate"].values * def_vulnerability.values)

            xp_cs = p_60_mins * prob_cs * pts_cs


            # Component 3b: Expected Goals Conceded Penalty for GKP & DEF (-1 pt per 2 goals conceded)
            # Discrete Poisson expectation per match: Sum_{k=2..10} floor(k/2) * P(k; mu_conceded)
            mu_c = np.clip(1.25 * (def_vulnerability.values / np.maximum(dgw_mult, 1.0)), 0.2, 3.5)
            ks = np.arange(2, 11)
            penalties = ks // 2
            poisson_conceded_pmfs = stats.poisson.pmf(ks, mu_c[:, None])
            exp_conceded_pen = np.sum(poisson_conceded_pmfs * penalties, axis=1) * dgw_mult
            xp_conceded = np.where(matrix["position"].isin(["GKP", "DEF"]), -1.0 * p_60_mins * exp_conceded_pen, 0.0)

            # Component 3c: Goalkeeper Save Points (+1 pt per 3 saves)
            # Discrete Poisson expectation per match: Sum_{s=3..15} floor(s/3) * P(s; lambda_saves)
            lam_s = np.clip(1.75 * (def_vulnerability.values / np.maximum(dgw_mult, 1.0)), 1.5, 5.5)
            saves_k = np.arange(3, 16)
            save_pts = saves_k // 3
            save_pmfs = stats.poisson.pmf(saves_k, lam_s[:, None])
            exp_saves = np.sum(save_pmfs * save_pts, axis=1) * dgw_mult
            xp_saves = np.where(matrix["position"] == "GKP", p_1_min * exp_saves, 0.0)

            # Component 4: Standard Bonus & Cards
            sim_bonus = self.simulate_joint_bps(matrix, fix_map=fix_map, n_sims=150) if w == 0 else pd.Series(p90_factor * matrix["r_bonus_base"] * dgw_mult, index=matrix.index)
            xp_bonus = sim_bonus.values if isinstance(sim_bonus, pd.Series) else sim_bonus


            # Component 5: 2025/26 Defensive Bonus Rules (CBIT & CBIRT)
            def_lam_cbit = np.maximum(matrix["r_cbit"].values * p90_factor, 0.1)
            mid_lam_cbirt = np.maximum(matrix["r_cbirt"].values * p90_factor, 0.1)
            
            p_cbit = stats.poisson.sf(9, def_lam_cbit) # P(CBIT >= 10)
            p_cbirt = stats.poisson.sf(11, mid_lam_cbirt) # P(CBIRT >= 12)
            
            xp_def_bonus = np.where(
                matrix["position"] == "DEF",
                2.0 * p_cbit * dgw_mult,
                np.where(matrix["position"].isin(["MID", "FWD"]), 2.0 * p_cbirt * dgw_mult, 0.0)
            )

            xp_misc = xp_bonus + xp_def_bonus - p90_factor * matrix["r_cards"] * 1.0 * dgw_mult
            xp_total = np.maximum(xp_app + xp_att + xp_cs + xp_conceded + xp_saves + xp_misc, 0.0)

            # Force 0.0 xP for gameweek-unavailable players (avail_factor <= 0.0) or Blank Gameweeks
            is_unavailable = (avail_factor <= 0.0) | is_blank
            xp_total = np.where(is_unavailable, 0.0, xp_total)
            
            matrix[f"xP_{gw}"] = np.round(xp_total * decay, 3)

        xp_cols = [c for c in matrix.columns if c.startswith("xP_")]
        matrix["xP_horizon_sum"] = matrix[xp_cols].sum(axis=1)

        return matrix
