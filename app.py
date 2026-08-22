# -*- coding: utf-8 -*-
"""
Jetski FPL Quantitative Web Interface (app.py)
Streamlit dashboard structured into 4 Master Decision Hubs:
1. 🏟️ Squad Optimizer & Planning (GW1 Optimal Builder, Active Squad, 6-GW Roadmap, Chip Hurdles)
2. 🔍 Player Scout & Direct Comparison (Head-to-Head Option Comparison, Multi-Filter Scout Table, Injury Radar)
3. 📊 Fixture & Market Intelligence (380-Match Schedule & ClubElo, De-Vigged Betting Odds)
4. 🧪 Strategy & Backtest Lab (Article NLP Fact-Checker, Walk-Forward Backtest, What-If Scenario Studio)
"""

import os
import sys
import textwrap
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Overfit FPL Quantitative Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    sync_fpl_api_data, get_last_updated_info, check_and_auto_update_data,
    load_player_history, fetch_clubelo_ratings, get_full_fpl_schedule,
    get_clubelo_visualization_df, get_player_availability_df, get_db_connection,
    fetch_live_sharp_odds, get_odds_quota_info, get_price_change_radar_df
)
from rate_engine import CanonicalRateEngine
from devig_engine import SharpOddsEngine
from squad_manager import (
    get_active_squad_state, save_active_squad_state, execute_squad_transfer,
    build_gw1_start_of_season_squad, compare_all_formations_gw1,
    generate_player_rationale, SquadAdversarialCritic, iterative_squad_optimization_loop
)
from article_analyzer import ArticleSentimentEngine
from chip_evaluator import ChipEvaluator, MacroSeasonChipScheduler
from backtester import WalkForwardBacktestHarness
from optimizer import MultiPeriodMILP

MY_MANAGER_ID = 2896432

# Custom CSS for Premium Visual Styling
st.markdown("""
<style>
    .status-fresh { color: #00ff87; font-weight: bold; }
    .status-stale { color: #ff4b4b; font-weight: bold; }
    .badge-cap { background-color: #ffd700; color: #000; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    .badge-starter { background-color: #2e7d32; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    .badge-bench { background-color: #616161; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    .badge-flagged { background-color: #d32f2f; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    
    .player-card-captain { background-color: #1e261e; border: 2px solid #ffd700; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .player-card-starter { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .player-card-bench { background-color: #0d1117; border: 1px dashed #30363d; border-radius: 8px; padding: 12px; margin-bottom: 10px; opacity: 0.85; }
    
    .rationale-text { font-size: 0.85em; color: #8b949e; margin-top: 6px; font-style: italic; }
    .verdict-supported { background-color: #0d2818; border-left: 4px solid #00ff87; padding: 10px; border-radius: 4px; margin-bottom: 8px; }
    .verdict-caution { background-color: #2d2006; border-left: 4px solid #ffd700; padding: 10px; border-radius: 4px; margin-bottom: 8px; }
    .verdict-skeptical { background-color: #2d0e0e; border-left: 4px solid #ff4b4b; padding: 10px; border-radius: 4px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Overfit FPL Control Center")
info = get_last_updated_info()
st.sidebar.subheader("🗞️ Data Warehouse Status")
if info["is_fresh"]:
    st.sidebar.markdown(f"Status: <span class='status-fresh'>FRESH (Updated < 2h)</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"Status: <span class='status-stale'>STALE / REFRESH NEEDED (Updated > 2h)</span>", unsafe_allow_html=True)

st.sidebar.info(f"**Last Sync:** {info['last_updated']}\n\n**Data Age:** {info['age_hours']} hours")

if st.sidebar.button("⚡ Force Refresh Data", type="primary"):
    with st.sidebar.status("Syncing FPL API & ClubElo data..."):
        success = sync_fpl_api_data()
        if success:
            st.sidebar.success("✅ Synced successfully!")
            st.rerun()
        else:
            st.sidebar.error("❌ Sync failed. Please check network.")

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Manager Profile")
st.sidebar.markdown(f"**Manager ID:** `{MY_MANAGER_ID}`")

# Main Load
history_df = load_player_history()

# Rate Matrix Computation
@st.cache_data(ttl=3600)
def compute_rate_matrix(df):
    engine = CanonicalRateEngine(df)
    elo_dict = fetch_clubelo_ratings()
    
    conn = get_db_connection()
    try:
        res = pd.read_sql("SELECT MIN(event) as next_gw FROM fixtures WHERE finished = 0", conn)
        next_gw = res.iloc[0]["next_gw"]
        start_gw = int(next_gw) if pd.notna(next_gw) and 1 <= int(next_gw) <= 38 else 1
    except Exception:
        start_gw = 1
    finally:
        conn.close()

    matrix = engine.generate_horizon_matrix(start_gw=start_gw, horizon_weeks=6, elo_dict=elo_dict)
    return matrix, start_gw

@st.cache_data(ttl=3600)
def fetch_upcoming_fixtures_map(start_gw=1, num_gws=3):
    conn = get_db_connection()
    try:
        df = pd.read_sql("""
            SELECT f.event, f.team_h, f.team_a, f.team_h_difficulty, f.team_a_difficulty,
                   th.short_name as team_h_short, ta.short_name as team_a_short,
                   th.name as team_h_name, ta.name as team_a_name
            FROM fixtures f
            LEFT JOIN teams th ON f.team_h = th.id
            LEFT JOIN teams ta ON f.team_a = ta.id
            WHERE f.event >= ? AND f.event < ?
            ORDER BY f.event, f.id
        """, conn, params=(start_gw, start_gw + num_gws))
        
        fdr_colors = {
            1: ("#00ff87", "#000000"),
            2: ("#05f177", "#000000"),
            3: ("#e7e7e7", "#000000"),
            4: ("#ff5e00", "#ffffff"),
            5: ("#80072d", "#ffffff"),
        }

        fix_map = {}
        for _, row in df.iterrows():
            gw = row["event"]
            h_name, a_name = row["team_h_name"], row["team_a_name"]
            h_short, a_short = row["team_h_short"], row["team_a_short"]
            h_diff, a_diff = int(row["team_h_difficulty"] or 3), int(row["team_a_difficulty"] or 3)

            if h_name:
                bg, fg = fdr_colors.get(h_diff, ("#e7e7e7", "#000000"))
                fix_map.setdefault(h_name, []).append({"gw": gw, "text": f"{a_short} (H)", "fdr": h_diff, "bg": bg, "fg": fg})
            if a_name:
                bg, fg = fdr_colors.get(a_diff, ("#e7e7e7", "#000000"))
                fix_map.setdefault(a_name, []).append({"gw": gw, "text": f"{h_short} (A)", "fdr": a_diff, "bg": bg, "fg": fg})

        return fix_map
    except Exception:
        return {}
    finally:
        conn.close()

matrix, start_gw = compute_rate_matrix(history_df)
upcoming_fix_map = fetch_upcoming_fixtures_map(start_gw=start_gw, num_gws=3)
st.session_state["upcoming_fix_map"] = upcoming_fix_map

if "manual_overrides" not in st.session_state:
    st.session_state["manual_overrides"] = {}

if st.session_state["manual_overrides"] and not matrix.empty:
    analyzer = ArticleSentimentEngine(matrix)
    matrix = analyzer.apply_user_overrides(st.session_state["manual_overrides"])

# Header
st.title("🏆 Overfit FPL Quantitative Decision Engine")
st.caption(f"Real-Time Manager Tracking (`ID: {MY_MANAGER_ID}`) | Data Last Synced: **{info['last_updated']}**")

# Top KPI Bar
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Data Sync Time", info["last_updated"].split()[0] if " " in info["last_updated"] else info["last_updated"])
with m2: st.metric("Data Age", f"{info['age_hours']} hrs", delta="- Fresh" if info["is_fresh"] else "+ Stale")
with m3: st.metric("Manager ID", f"{MY_MANAGER_ID}")
with m4: st.metric("Active Fixtures", "380 Matches")

st.markdown("---")

def render_player_card(row, fix_map=None):
    role = str(row.get("role", "Starter"))
    badge_cls = "badge-cap" if "👑 Captain" in role else ("badge-starter" if role != "Bench" else "badge-bench")
    card_cls = "player-card-captain" if "👑 Captain" in role else ("player-card-starter" if role != "Bench" else "player-card-bench")
    
    xp_val = row.get("GW1_xP")
    if xp_val is None or pd.isna(xp_val) or float(xp_val) <= 0.0:
        for c in ["xP_1", "xP_2", "xP_3", "xP_4", "xP_5", "xP_6", "xP_7", "xP_8"]:
            if c in row and pd.notna(row.get(c)) and float(row.get(c)) > 0.0:
                xp_val = row.get(c)
                break
    if xp_val is None or pd.isna(xp_val) or float(xp_val) <= 0.0:
        if "xP_horizon_sum" in row and pd.notna(row.get("xP_horizon_sum")) and float(row.get("xP_horizon_sum")) > 0.0:
            xp_val = float(row.get("xP_horizon_sum")) / 6.0
        else:
            pos = str(row.get("position", "MID"))
            xp_val = 3.8 if pos in ["GKP", "DEF"] else (4.6 if pos == "MID" else 5.2)
    xp = float(xp_val)
    
    cost_val = row.get("cost", 5.0)
    cost = 5.0 if pd.isna(cost_val) else float(cost_val)
    
    chance_val = row.get("chance_of_playing", 100)
    chance = 100.0 if pd.isna(chance_val) else float(chance_val)
    
    news_val = row.get("news", "")
    news_str = "" if pd.isna(news_val) or not news_val else str(news_val)
    
    flag_html = ""
    if chance < 100 or (news_str and news_str != "Fully Fit / Available"):
        flag_html = f'<span class="badge-flagged">⚠️ {int(chance)}% ({news_str[:25]}...)</span>'

    if fix_map is None and "upcoming_fix_map" in st.session_state:
        fix_map = st.session_state["upcoming_fix_map"]

    fix_html = ""
    if fix_map:
        team_name = row.get("team", "")
        upcoming = fix_map.get(team_name, [])[:3]
        if upcoming:
            pills = []
            for f in upcoming:
                pills.append(f'<span style="background-color:{f["bg"]}; color:{f["fg"]}; padding:2px 6px; border-radius:4px; font-size:0.75em; font-weight:bold; margin-right:3px;">{f["text"]}</span>')
            fix_html = f'<div style="margin-top:4px; display:flex; align-items:center;"><span style="font-size:0.78em; color:#8b949e; margin-right:5px; font-weight:600;">Next 3:</span>{"".join(pills)}</div>'

    rationale_raw = str(row.get('rationale', 'Selected based on baseline xP.')).strip()
    if rationale_raw.startswith('"') and rationale_raw.endswith('"') and len(rationale_raw) > 2:
        rationale_raw = rationale_raw[1:-1].strip()

    card_html = f"""<div class="{card_cls}">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div><strong>{row.get('name', 'Player')}</strong> {flag_html}</div>
<span class="{badge_cls}">{role}</span>
</div>
<div style="margin-top:4px; font-size:0.9em;">
<span style="color:#58a6ff;">{row.get('position', 'MID')}</span> | {row.get('team', 'UNK')} | <strong>£{cost:.1f}m</strong> | <strong>{xp:.2f} xP</strong>
</div>
{fix_html}
<div class="rationale-text">"{rationale_raw}"</div>
</div>"""
    st.markdown(card_html, unsafe_allow_html=True)

# ==============================================================================
# 🌟 FOUR MASTER DECISION HUBS
# ==============================================================================
hub1, hub2, hub3, hub4 = st.tabs([
    "🏟️ Squad Optimizer & Planning",
    "🔍 Player Scout & Direct Comparison",
    "📊 Fixture & Market Intelligence",
    "🧪 Strategy & Backtest Lab"
])

# ==============================================================================
# HUB 1: SQUAD OPTIMIZATION & PLANNING
# ==============================================================================
with hub1:
    st.caption("Complete squad decision center: Optimize GW1 team, manage active transfers, build 6-GW roadmaps, and evaluate strategic chip hurdles.")
    
    subtab_squad_gw1, subtab_squad_active, subtab_squad_roadmap, subtab_squad_chips = st.tabs([
        "🚀 Optimal GW1 Squad Builder",
        "👤 Active Squad & Transfers",
        "📋 6-GW Transfer Roadmap",
        "🃏 Chip Hurdle Evaluator"
    ])

    # --- 1.1 OPTIMAL GW1 SQUAD BUILDER ---
    with subtab_squad_gw1:
        st.header("🚀 Start-of-Season GW1 Optimal Squad Builder")
        st.caption("Builds an optimal 15-man squad under budget with custom formation and elite player locks (e.g. Erling Haaland).")

        col_b, col_f, col_l = st.columns(3)
        with col_b:
            budget_input = st.number_input("Starting Budget (£m)", min_value=90.0, max_value=105.0, value=100.0, step=0.5, key="gw1_budget")
        with col_f:
            formation_input = st.selectbox("Select Target Formation", ["Automatic", "3-4-3", "3-5-2", "4-4-2", "4-3-3", "4-5-1", "5-3-2"], key="gw1_formation")
        with col_l:
            if not matrix.empty:
                premium_players = matrix[matrix["cost"] >= 7.5].sort_values("xP_horizon_sum", ascending=False)
                lock_options = {str(k): str(v) for k, v in zip(premium_players["player_id"], premium_players["name"] + " (" + premium_players["team"] + " - £" + premium_players["cost"].astype(str) + "m)")}
                locked_pids_str = st.multiselect("Lock Elite Premium Players", options=list(lock_options.keys()), format_func=lambda x: lock_options[str(x)], key="gw1_locks")
                locked_pids = [int(p) for p in locked_pids_str]
            else:
                locked_pids = []

        st.markdown("---")
        st.subheader("📊 Compare Projected Points Across Formations for GW1")
        if st.button("⚡ Compare All Formations for GW1", key="btn_compare_fmt"):
            with st.spinner("Solving MILP across 3-4-3, 3-5-2, 4-4-2, 4-3-3, 4-5-1, 5-3-2..."):
                fmt_comp_df = compare_all_formations_gw1(history_df, budget=budget_input, locked_player_ids=locked_pids)
                st.session_state["fmt_comp_df"] = fmt_comp_df

        if "fmt_comp_df" in st.session_state and not st.session_state["fmt_comp_df"].empty:
            fmt_comp_df = st.session_state["fmt_comp_df"]
            top_fmt = fmt_comp_df.iloc[0]
            st.success(f"🏆 Best Formation for GW1: **{top_fmt['Formation']}** projecting **{top_fmt['GW1_Projected_xP']} pts** (6-GW Total: {top_fmt['6GW_Horizon_xP']} pts) with Captain **{top_fmt['Top_Captain']}**!")
            
            c_chart, c_tbl = st.columns([1, 1])
            with c_chart:
                st.bar_chart(fmt_comp_df.set_index("Formation")["GW1_Projected_xP"])
            with c_tbl:
                st.dataframe(fmt_comp_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            gen_single = st.button("🔮 Generate Optimal GW1 Squad (100% Budget)", type="primary", key="btn_gen_single")
        with c_btn2:
            gen_multi = st.button("🤖 Run Multi-Round Adversarial Optimization (3 Rounds)", type="secondary", key="btn_gen_multi")

        if gen_single or gen_multi:
            with st.spinner("Executing Mathematical Solver & Adversarial Critique Engine..."):
                try:
                    if gen_multi:
                        squad_gw1, bank_rem, iter_history = iterative_squad_optimization_loop(
                            history_df,
                            budget=budget_input,
                            formation=formation_input,
                            locked_player_ids=locked_pids,
                            max_rounds=3
                        )
                        st.session_state["iter_history"] = iter_history
                    else:
                        squad_gw1, bank_rem = build_gw1_start_of_season_squad(
                            history_df,
                            budget=budget_input,
                            formation=formation_input,
                            locked_player_ids=locked_pids,
                            max_unspent_bank=0.0
                        )
                        st.session_state.pop("iter_history", None)

                    if not squad_gw1.empty:
                        st.session_state["squad_gw1"] = squad_gw1
                        st.session_state["bank_rem"] = bank_rem
                        st.success("✅ GW1 Optimal Squad successfully generated and audited!")
                    else:
                        st.warning("⚠️ Solver returned empty squad. Please adjust budget or player locks.")
                except Exception as e:
                    st.error(f"❌ Error during optimization: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if ("squad_gw1" not in st.session_state or st.session_state["squad_gw1"].empty) and not history_df.empty:
            squad_gw1, bank_rem = build_gw1_start_of_season_squad(history_df, budget=budget_input, max_unspent_bank=0.0)
            if not squad_gw1.empty:
                st.session_state["squad_gw1"] = squad_gw1
                st.session_state["bank_rem"] = bank_rem

        if "squad_gw1" in st.session_state and not st.session_state["squad_gw1"].empty:
            squad_gw1 = st.session_state["squad_gw1"]
            bank_rem = st.session_state["bank_rem"]

            starters = squad_gw1[squad_gw1["role"] != "Bench"]
            bench = squad_gw1[squad_gw1["role"] == "Bench"]
            cap_row = squad_gw1[squad_gw1["role"] == "👑 Captain"]
            cap_bonus_gw1 = cap_row.iloc[0]["GW1_xP"] if not cap_row.empty and "GW1_xP" in cap_row.columns else (cap_row.iloc[0]["xP_1"] if not cap_row.empty and "xP_1" in cap_row.columns else 0.0)
            cap_bonus_horiz = cap_row.iloc[0]["xP_horizon_sum"] if not cap_row.empty and "xP_horizon_sum" in cap_row.columns else 0.0

            gw1_pts = starters["GW1_xP"].sum() + cap_bonus_gw1 if "GW1_xP" in starters.columns else starters.get("xP_1", pd.Series([4.0]*len(starters))).sum() + cap_bonus_gw1
            horiz_pts = starters.get("xP_horizon_sum", pd.Series([20.0]*len(starters))).sum() + cap_bonus_horiz
            spent_cost = squad_gw1["cost"].sum()

            b1, b2, b3, b4 = st.columns(4)
            with b1: st.metric("Starting XI + Captain GW1 xP", f"{gw1_pts:.2f} pts")
            with b2: st.metric("Starting XI + Captain 6-GW Horizon xP", f"{horiz_pts:.2f} pts")
            with b3: st.metric("Total Squad Cost", f"£{spent_cost:.1f}m / £{budget_input:.1f}m")
            with b4: st.metric("Bank Balance", f"£{bank_rem:.1f}m", delta="100% Capitalized" if bank_rem <= 0.1 else "Unspent Cash")

            if "iter_history" in st.session_state and st.session_state["iter_history"]:
                st.markdown("---")
                st.subheader("🔁 Multi-Round Adversarial Convergence History")
                history_rows = []
                for h in st.session_state["iter_history"]:
                    history_rows.append({
                        "Round": f"Round {h['round']}",
                        "Phase Description": h["label"],
                        "Squad Cost": f"£{h['cost']:.1f}m",
                        "Bank": f"£{h['bank']:.1f}m",
                        "GW1 Projected xP": f"{h['projected_gw1_xp']:.2f} pts",
                        "Adversarial Grade": h["critic_report"]["overall_grade"],
                        "Audit Status": h["critic_report"]["budget_status"]
                    })
                st.dataframe(pd.DataFrame(history_rows), hide_index=True, use_container_width=True)

            critic_report = SquadAdversarialCritic.critique_squad(squad_gw1, budget=budget_input, matrix_df=matrix)
            
            st.markdown("---")
            st.subheader(f"🛡️ Adversarial Stress-Test Audit Scorecard (Grade: {critic_report['overall_grade']})")
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1: st.metric("💰 Budget Capitalization", f"{critic_report['budget_score']} / 100")
            with sc2: st.metric("👑 EO Anchor Shield", f"{critic_report['eo_score']} / 100")
            with sc3: st.metric("🪑 Bench Security", f"{critic_report['bench_score']} / 100")

            if critic_report["strengths"]:
                for s in critic_report["strengths"]:
                    st.markdown(f"> {s}")

            if critic_report["alerts"]:
                for a in critic_report["alerts"]:
                    st.warning(a)

            st.markdown("---")
            st.subheader("📋 Final Optimized 15-Man Squad Layout")
            
            col_st, col_bn = st.columns([2, 1])
            with col_st:
                st.markdown(f"### ⚽ Starting 11 Starters ({len(starters)} Players)")
                for pos in ["GKP", "DEF", "MID", "FWD"]:
                    pos_p = starters[starters["position"] == pos]
                    if not pos_p.empty:
                        st.markdown(f"**{pos}s:**")
                        for _, p_row in pos_p.iterrows():
                            render_player_card(p_row)

            with col_bn:
                st.markdown(f"### 🪑 Bench Enablers ({len(bench)} Players)")
                for _, p_row in bench.iterrows():
                    render_player_card(p_row)

            if critic_report.get("player_alternatives"):
                st.markdown("---")
                st.subheader("🔍 Transparent Decision Breakdown: 'Why Player X over Market Rivals?'")
                st.caption("Removes black-box uncertainty by showing the top 2 alternative candidates considered for each starter at similar price points.")

                with st.expander("🔎 View Statistical Alternatives & Trade-Offs for Every Starter", expanded=False):
                    for p_name, details in critic_report["player_alternatives"].items():
                        st.markdown(f"#### 👤 **{p_name}** ({details['position']} - £{details['cost']:.1f}m | **{details['xp']:.2f} xP**)")
                        st.caption(f"Selection Rationale: {details['rationale']}")
                        if details["competitors"]:
                            comp_str = " | ".join([f"**{c['name']}**: {c['xp']:.2f} xP ({c['delta_xp']})" for c in details["competitors"]])
                            st.markdown(f"⚔️ **Closest Competitors:** {comp_str}")
                        else:
                            st.markdown("⚔️ *Uncontested Tier Leader (No close competitors in price range)*")
                        st.markdown("---")

            if st.button("💾 Save as My Active Squad", key="btn_save_active"):
                save_active_squad_state(squad_gw1, bank=bank_rem, fts=1)
                st.success("✅ Saved optimal squad to active state!")
                st.rerun()

    # --- 1.2 ACTIVE SQUAD & TRANSFERS ---
    with subtab_squad_active:
        st.header(f"👤 Active Manager Squad (Manager ID: {MY_MANAGER_ID})")
        
        sync_col1, sync_col2 = st.columns([1, 1])
        with sync_col1:
            if st.button("🔄 Sync My Squad Live from FPL API", key="btn_sync_live_fpl", type="secondary"):
                with st.spinner("Fetching live squad from fantasy.premierleague.com..."):
                    live_squad = load_manager_2667805_squad(manager_id=MY_MANAGER_ID, gw=1)
                    if not live_squad.empty:
                        save_active_squad_state(live_squad, bank=0.0, fts=1)
                        st.success(f"✅ Successfully synced {len(live_squad)} live picks from official FPL API!")
                        st.rerun()
        with sync_col2:
            if st.button("⚡ Solve Optimal 1-Transfer Move for GW2", key="btn_solve_gw2_move", type="primary"):
                active_squad_curr, curr_bank, curr_fts = get_active_squad_state()
                if not active_squad_curr.empty and not matrix.empty:
                    opt = MultiPeriodMILP(matrix)
                    base_ids = active_squad_curr["player_id"].tolist()
                    plan = opt.solve_rolling_horizon(base_ids, initial_bank=curr_bank, initial_fts=curr_fts)
                    st.session_state["gw2_optimal_plan"] = plan

        if "gw2_optimal_plan" in st.session_state and st.session_state["gw2_optimal_plan"]:
            plan = st.session_state["gw2_optimal_plan"]
            if plan.get("status") == "Optimal":
                st.success("🏆 **Optimal Gameweek 2 Recommended Move (1 Free Transfer)**:")
                t_in = plan.get("transfers_in", [])
                t_out = plan.get("transfers_out", [])
                if t_out and t_in:
                    p_out = matrix[matrix["player_id"] == t_out[0]].iloc[0]
                    p_in = matrix[matrix["player_id"] == t_in[0]].iloc[0]
                    st.markdown(f"### 🔴 **SELL:** `{p_out['name']}` ({p_out['team']}, {p_out['position']} - £{p_out['cost']:.1f}m) ➔ 🟢 **BUY:** `{p_in['name']}` ({p_in['team']}, {p_in['position']} - £{p_in['cost']:.1f}m)")
                else:
                    st.info("ℹ️ **Recommendation:** Roll Free Transfer to accumulate 2 FTs for Gameweek 3.")
                
                cap_name = matrix.loc[matrix["player_id"] == plan["captain_id"], "name"].values[0] if "captain_id" in plan and not matrix[matrix["player_id"] == plan["captain_id"]].empty else "N/A"
                vc_name = matrix.loc[matrix["player_id"] == plan.get("vice_captain_id"), "name"].values[0] if plan.get("vice_captain_id") and not matrix[matrix["player_id"] == plan.get("vice_captain_id")].empty else "N/A"
                st.markdown(f"👑 **Captain:** `{cap_name}` | 🥈 **Vice-Captain:** `{vc_name}` | 📈 **Projected GW2 Starting Points:** `{plan['projected_points_gw1']:.2f} pts`")
                st.markdown("---")

        active_squad, bank, fts = get_active_squad_state()
        
        if not active_squad.empty:
            if not matrix.empty and "player_id" in active_squad.columns:
                for col in ["name", "position", "team", "cost"]:
                    if col not in active_squad.columns and col in matrix.columns:
                        active_squad = active_squad.merge(matrix[["player_id", col]], on="player_id", how="left")
                
                opt_cols = [c for c in ["chance_of_playing", "news", "status"] if c in matrix.columns]
                req_cols = ["player_id", f"xP_{start_gw}", "r_goal", "r_assist", "xM", "team_cs_rate"] + opt_cols
                req_cols = [c for c in req_cols if c in matrix.columns]
            merged_squad = active_squad.merge(
                matrix[req_cols],
                on="player_id",
                how="left"
            ).rename(columns={f"xP_{start_gw}": "GW1_xP"})
        else:
            merged_squad = active_squad.copy()
            
        if "GW1_xP" not in merged_squad.columns or merged_squad["GW1_xP"].isna().any():
            merged_squad["GW1_xP"] = merged_squad.get("GW1_xP", pd.Series([4.0]*len(merged_squad))).fillna(4.0)

        if "position" not in merged_squad.columns: merged_squad["position"] = "MID"
        if "name" not in merged_squad.columns: merged_squad["name"] = "Player"
        if "team" not in merged_squad.columns: merged_squad["team"] = "Premier League"
        if "cost" not in merged_squad.columns: merged_squad["cost"] = 5.0

        merged_squad["rationale"] = merged_squad.apply(lambda r: generate_player_rationale(r, r.get("GW1_xP", 4.0)), axis=1)

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Active Squad Size", f"{len(merged_squad)} / 15 Players")
        with k2: st.metric("Bank Balance", f"£{bank:.1f}m")
        with k3: st.metric("Available Free Transfers", f"{fts} FTs")
        with k4: st.metric("GW1 Total Projected xP", f"{merged_squad['GW1_xP'].fillna(0).sum():.2f} pts")


        st.markdown("---")
        st.subheader("📋 Active 15-Man Squad Cards & Selection Rationales")
        
        def clean_role(r):
            r_str = str(r)
            if "Captain" in r_str and "Vice" not in r_str: return "👑 Captain"
            elif "Vice" in r_str: return "🥈 Vice Captain"
            elif r_str == "Bench": return "Bench"
            else: return "Starter"

        merged_squad["role"] = merged_squad["role"].apply(clean_role)

        starters_act = merged_squad[merged_squad["role"] != "Bench"]
        bench_act = merged_squad[merged_squad["role"] == "Bench"]
        
        c_starters, c_bench = st.columns([2, 1])
        with c_starters:
            st.markdown(f"### ⚽ Starting 11 Starters ({len(starters_act)} Players)")
            for pos in ["GKP", "DEF", "MID", "FWD"]:
                pos_players = starters_act[starters_act["position"] == pos]
                if not pos_players.empty:
                    st.markdown(f"**{pos}s:**")
                    for _, p_row in pos_players.iterrows():
                        render_player_card(p_row)

        with c_bench:
            st.markdown(f"### 🪑 Bench Enablers ({len(bench_act)} Players)")
            for _, p_row in bench_act.iterrows():
                render_player_card(p_row)


        st.markdown("---")
        st.subheader("🔄 Interactive Transfer Execution Tool")
        c_sell, c_buy = st.columns(2)
        with c_sell:
            sell_options = {str(k): str(v) for k, v in zip(merged_squad["player_id"], merged_squad["name"] + " (" + merged_squad["position"] + " - £" + merged_squad["cost"].astype(str) + "m)")}
            sell_pid_str = st.selectbox("Sell Player (Out)", options=list(sell_options.keys()), format_func=lambda x: sell_options[str(x)], key="act_sell_p")
            sell_pid = int(sell_pid_str) if sell_pid_str else None
        with c_buy:
            if not matrix.empty:
                owned_ids = merged_squad["player_id"].tolist()
                market_df = matrix[~matrix["player_id"].isin(owned_ids)].sort_values("xP_horizon_sum", ascending=False)
                buy_options = {str(k): str(v) for k, v in zip(market_df["player_id"], market_df["name"] + " (" + market_df["position"] + ", " + market_df["team"] + " - £" + market_df["cost"].astype(str) + "m)")}
                buy_pid_str = st.selectbox("Buy Player (In)", options=list(buy_options.keys()), format_func=lambda x: buy_options[str(x)], key="act_buy_p")
                buy_pid = int(buy_pid_str) if buy_pid_str else None
            else:
                buy_pid = None

        if st.button("Confirm Transfer & Update Squad State", type="primary", key="btn_confirm_tr"):
            if sell_pid and buy_pid:
                buy_row = matrix[matrix["player_id"] == buy_pid].iloc[0]
                success, msg, new_squad, new_bank, new_fts, hit = execute_squad_transfer(merged_squad, sell_pid, buy_row, bank, fts)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)


    # --- 1.3 6-GW ROADMAP ---
    with subtab_squad_roadmap:
        st.header(f"📋 6-Gameweek Squad Roadmap (Target GW {start_gw})")
        st.caption("Solves rolling-horizon Model Predictive Control (MPC) to optimize transfer timing, FT accumulation, and captaincy.")

        if st.button("🚀 Calculate 6-GW Transfer Roadmap", type="primary", key="btn_calc_roadmap"):
            with st.spinner("Solving Rolling Horizon Model Predictive Control (MPC)..."):
                if not matrix.empty:
                    optimizer = MultiPeriodMILP(matrix)
                    active_squad_curr, curr_bank, curr_fts = get_active_squad_state()
                    if not active_squad_curr.empty:
                        base_squad_ids = active_squad_curr["player_id"].tolist()
                        purchase_prices = dict(zip(active_squad_curr["player_id"], active_squad_curr.get("purchase_price", active_squad_curr["cost"])))
                    else:
                        base_squad_ids = matrix.sort_values("xP_horizon_sum", ascending=False)["player_id"].head(15).tolist()
                        purchase_prices = None

                    plan = optimizer.solve_rolling_horizon(base_squad_ids, initial_bank=curr_bank, initial_fts=curr_fts, purchase_prices=purchase_prices)
                    st.session_state["roadmap_plan"] = plan

        if "roadmap_plan" in st.session_state and st.session_state["roadmap_plan"]:
            plan = st.session_state["roadmap_plan"]
            if plan.get("status") == "Optimal":
                k1, k2, k3, k4 = st.columns(4)
                with k1: st.metric("Projected Points", f"{plan['projected_points_gw1']:.2f} pts")
                with k2: st.metric("Remaining Bank", f"£{plan['bank']:.1f}m")
                with k3: st.metric("Hit Penalty Cost", f"-{plan['hits_cost']} pts")
                with k4: st.metric("Transfers Executed", len(plan["transfers_in"]))

                cap_name = matrix.loc[matrix["player_id"] == plan["captain_id"], "name"].values[0] if "captain_id" in plan and not matrix[matrix["player_id"] == plan["captain_id"]].empty else "N/A"
                vc_name = matrix.loc[matrix["player_id"] == plan.get("vice_captain_id"), "name"].values[0] if plan.get("vice_captain_id") and not matrix[matrix["player_id"] == plan.get("vice_captain_id")].empty else "N/A"
                
                st.info(f"👑 **Optimal Captain:** {cap_name} | 🥈 **Vice-Captain:** {vc_name}")
                
                roadmap_df = matrix[matrix["player_id"].isin(plan["starting_xi_ids"])][["name", "position", "team", "cost", f"xP_{start_gw}"]].copy()
                roadmap_df["Role"] = np.where(roadmap_df["name"] == cap_name, "👑 Captain", np.where(roadmap_df["name"] == vc_name, "🥈 Vice Captain", "Starter"))
                st.dataframe(roadmap_df, hide_index=True, use_container_width=True)

    # --- 1.4 CHIP EVALUATOR ---
    with subtab_squad_chips:
        st.header("🃏 Chip Reservation Hurdle Curve Evaluator")
        st.caption("Evaluates dynamic time-decayed hurdle curves (Rho thresholds) across Wildcard, Free Hit, Bench Boost, and Triple Captain.")

        if st.button("⚡ Calculate Chip Hurdle Curves", type="primary", key="btn_calc_chips"):
            with st.spinner("Evaluating shadow solves across standard, wildcard, and free-hit horizons..."):
                if not matrix.empty:
                    optimizer = MultiPeriodMILP(matrix)
                    active_squad_curr, curr_bank, curr_fts = get_active_squad_state()
                    if not active_squad_curr.empty:
                        base_squad_ids = active_squad_curr["player_id"].tolist()
                        purchase_prices = dict(zip(active_squad_curr["player_id"], active_squad_curr.get("purchase_price", active_squad_curr["cost"])))
                    else:
                        base_squad_ids = matrix.sort_values("xP_horizon_sum", ascending=False)["player_id"].head(15).tolist()
                        purchase_prices = None

                    chip_data = optimizer.evaluate_chip_deltas(base_squad_ids, initial_bank=curr_bank, initial_fts=curr_fts, purchase_prices=purchase_prices)
                    evaluator = ChipEvaluator()
                    report = evaluator.evaluate(chip_data["standard_plan"], chip_data["wildcard_plan"], chip_data["freehit_plan"])
                    st.session_state["chip_data"] = chip_data
                    st.session_state["chip_report"] = report

        if "chip_data" in st.session_state and "chip_report" in st.session_state:
            chip_data = st.session_state["chip_data"]
            report = st.session_state["chip_report"]
            
            st.subheader(f"Tactical Recommendation: **{report['summary_recommendation']}**")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Wildcard Delta", f"+{chip_data['delta_wildcard']:.2f} xP", delta=f"Hurdle: +{chip_data['hurdle_wildcard']:.1f}")
            with c2: st.metric("Free Hit Delta", f"+{chip_data['delta_freehit']:.2f} xP", delta=f"Hurdle: +{chip_data['hurdle_freehit']:.1f}")
            with c3: st.metric("Bench Boost Hurdle", "+12.0 xP")
            with c4: st.metric("Triple Captain Hurdle", "+10.0 xP")
            
            st.markdown("---")
            for chip_name, dec in report["chip_decisions"].items():
                trig = dec.get("trigger", False)
                status_str = "🔥 TRIGGER / PLAY CHIP" if trig else "✋ HOLD CHIP"
                thresh = dec.get("threshold", 0.0)
                st.markdown(f"**{chip_name.upper()}:** {status_str} (Hurdle Threshold: +{thresh:.1f} xP)")

        st.markdown("---")
        st.subheader("🗺️ 38-Gameweek Macro-Season Strategic Chip Roadmap & Timeline")
        st.caption("Global dynamic calendar mapping optimal multi-period execution windows for Wildcard 1, Free Hit, Wildcard 2, Bench Boost, and Triple Captain.")

        p1, p2, p3, p4 = st.columns(4)
        with p1: st.info("⚡ **Wildcard 1**\n\n**Window:** GW 6–8\n\n**Objective:** Restructure early punts & lock value.")
        with p2: st.warning("🛡️ **Free Hit**\n\n**Target:** BGW 29\n\n**Objective:** Field 11 starters during FA Cup clashes.")
        with p3: st.info("⚡ **Wildcard 2**\n\n**Window:** GW 31–33\n\n**Objective:** Build 15 DGW starters before Bench Boost.")
        with p4: st.success("🚀 **Bench Boost**\n\n**Target:** DGW 34 / 37\n\n**Objective:** Maximize 30 player appearances.")

        macro_roadmap_df = MacroSeasonChipScheduler.generate_macro_roadmap()
        
        # Interactive Gameweek Scrubbing Inspector
        st.markdown("##### 🔍 Inspect Specific Gameweek Strategy")
        selected_gw_scrub = st.slider("Select Gameweek to Inspect", min_value=1, max_value=38, value=int(start_gw), key="gw_scrub_slider")
        gw_row = macro_roadmap_df[macro_roadmap_df["GW_Num"] == selected_gw_scrub].iloc[0]
        
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 8px; border-left: 4px solid #00ff87; margin-bottom: 12px;">
            <div style="font-size: 1.1em; font-weight: bold; color: #00ff87;">Gameweek {selected_gw_scrub:02d} — {gw_row['Season_Phase']}</div>
            <div style="margin-top: 4px;"><strong>Fixture Profile:</strong> {gw_row['Fixture_Status']} | <strong>Chip Recommendation:</strong> {gw_row['Strategic_Chip_Target']}</div>
            <div style="margin-top: 4px; color: #cbd5e1;"><strong>Tactical Directive:</strong> {gw_row['Tactical_Objective']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(macro_roadmap_df.drop(columns=["GW_Num"]), hide_index=True, use_container_width=True)



# ==============================================================================
# HUB 2: PLAYER SCOUT & DIRECT COMPARISON STUDIO
# ==============================================================================
with hub2:
    st.caption("Deep-dive player analysis: Compare alternative options head-to-head, filter and scout 600+ Premier League assets, and monitor live injury & suspension reports.")
    
    subtab_scout_h2h, subtab_scout_table, subtab_scout_injuries, subtab_scout_price = st.tabs([
        "⚖️ Head-to-Head Option Comparison",
        "🔎 Interactive Player Scout Table",
        "🏥 Availability & Injury Radar",
        "📈 Daily Price Change & Wealth Radar"
    ])

    # Prepare matrix with display helpers
    mat_scout = matrix.copy()
    if not mat_scout.empty:
        mat_scout["cost"] = mat_scout["cost"].fillna(5.0).astype(float)
        mat_scout["GW1_xP"] = mat_scout[f"xP_{start_gw}"].fillna(4.0).astype(float)
        mat_scout["Horizon_xP"] = mat_scout["xP_horizon_sum"].fillna(20.0).astype(float)
        mat_scout["Value_Ratio"] = (mat_scout["Horizon_xP"] / mat_scout["cost"].clip(lower=1.0)).round(2)
        mat_scout["CS_Pct"] = (mat_scout["team_cs_rate"] * 100).round(1).astype(str) + "%"
        mat_scout["display_label"] = mat_scout.apply(
            lambda r: f"{r['name']} ({r['position']}, {r['team']} - £{r['cost']:.1f}m | {r['GW1_xP']:.2f} xP)",
            axis=1
        )

    # --- 2.1 HEAD-TO-HEAD COMPARISON ---
    with subtab_scout_h2h:
        st.subheader("⚖️ Direct Player vs Player Score & Budget Impact")
        st.caption("Select any two players to analyze the exact trade-off in projected points, budget savings, underlying stats, and upcoming fixtures.")

        if not mat_scout.empty:
            all_labels = mat_scout["display_label"].tolist()
            def_idx_a = next((i for i, s in enumerate(all_labels) if "Gabriel" in s and "Arsenal" in s), 0)
            def_idx_b = next((i for i, s in enumerate(all_labels) if ("Muñoz" in s or "Munoz" in s) and "Palace" in s), min(1, len(all_labels)-1))

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                p_label_a = st.selectbox("🅰️ Select Player A (e.g. Recommended Option)", all_labels, index=def_idx_a, key="h2h_p1")
            with col_p2:
                p_label_b = st.selectbox("🅱️ Select Player B (e.g. Alternative Option)", all_labels, index=def_idx_b, key="h2h_p2")

            row_a = mat_scout[mat_scout["display_label"] == p_label_a].iloc[0]
            row_b = mat_scout[mat_scout["display_label"] == p_label_b].iloc[0]

            cost_a, cost_b = float(row_a["cost"]), float(row_b["cost"])
            xp1_a, xp1_b = float(row_a["GW1_xP"]), float(row_b["GW1_xP"])
            xph_a, xph_b = float(row_a["Horizon_xP"]), float(row_b["Horizon_xP"])
            val_a, val_b = float(row_a["Value_Ratio"]), float(row_b["Value_Ratio"])

            d_cost = cost_b - cost_a
            d_xp1 = xp1_b - xp1_a
            d_xph = xph_b - xph_a
            d_val = val_b - val_a

            st.markdown("#### 📊 Score & Budget Delta (Player B vs Player A)")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                cost_str = f"£{d_cost:+.1f}m"
                cost_desc = f"Saves £{abs(d_cost):.1f}m" if d_cost < 0 else (f"Costs extra £{d_cost:.1f}m" if d_cost > 0 else "Same Cost")
                st.metric("Budget Impact", cost_str, delta=cost_desc, delta_color="inverse")
            with k2:
                st.metric(f"GW{start_gw} Score Impact", f"{d_xp1:+.2f} pts", delta=f"{xp1_b:.2f} vs {xp1_a:.2f} xP")
            with k3:
                st.metric("6-GW Total Score Impact", f"{d_xph:+.2f} pts", delta=f"{xph_b:.2f} vs {xph_a:.2f} xP")
            with k4:
                st.metric("Value Efficiency (xP/£m)", f"{d_val:+.2f} pts/£", delta=f"{val_b:.2f} vs {val_a:.2f}")

            st.markdown("---")
            card_col1, card_col2 = st.columns(2)
            with card_col1:
                st.markdown(f"### 🅰️ **{row_a['name']}** ({row_a['team']})")
                render_player_card(row_a)
            with card_col2:
                st.markdown(f"### 🅱️ **{row_b['name']}** ({row_b['team']})")
                render_player_card(row_b)

            st.markdown("---")
            st.subheader("📋 Detailed Statistical Breakdown")
            comp_table_data = [
                {"Metric": "Position", row_a["name"]: row_a["position"], row_b["name"]: row_b["position"], "Advantage": "Same" if row_a["position"] == row_b["position"] else f"{row_b['position']} vs {row_a['position']}"},
                {"Metric": "Club Team", row_a["name"]: row_a["team"], row_b["name"]: row_b["team"], "Advantage": "—"},
                {"Metric": "Cost (£m)", row_a["name"]: f"£{cost_a:.1f}m", row_b["name"]: f"£{cost_b:.1f}m", "Advantage": f"🅱️ Cheaper by £{abs(d_cost):.1f}m" if d_cost < 0 else (f"🅰️ Cheaper by £{d_cost:.1f}m" if d_cost > 0 else "Equal")},
                {"Metric": f"GW{start_gw} Expected Points (xP)", row_a["name"]: f"{xp1_a:.2f} pts", row_b["name"]: f"{xp1_b:.2f} pts", "Advantage": f"🅱️ +{d_xp1:.2f} pts" if d_xp1 > 0 else f"🅰️ +{abs(d_xp1):.2f} pts"},
                {"Metric": "6-GW Total Expected Points", row_a["name"]: f"{xph_a:.2f} pts", row_b["name"]: f"{xph_b:.2f} pts", "Advantage": f"🅱️ +{d_xph:.2f} pts" if d_xph > 0 else f"🅰️ +{abs(d_xph):.2f} pts"},
                {"Metric": "Value for Money (xP / £m)", row_a["name"]: f"{val_a:.2f}", row_b["name"]: f"{val_b:.2f}", "Advantage": f"🅱️ +{d_val:.2f} pts/£" if d_val > 0 else f"🅰️ +{abs(d_val):.2f} pts/£"},
                {"Metric": "Clean Sheet Rate", row_a["name"]: f"{float(row_a.get('team_cs_rate', 0.3))*100:.1f}%", row_b["name"]: f"{float(row_b.get('team_cs_rate', 0.3))*100:.1f}%", "Advantage": "🅱️ Higher" if float(row_b.get("team_cs_rate", 0.3)) > float(row_a.get("team_cs_rate", 0.3)) else "🅰️ Higher"},
                {"Metric": "Goal Threat (r_goal / 90)", row_a["name"]: f"{float(row_a.get('r_goal', 0.0)):.3f}", row_b["name"]: f"{float(row_b.get('r_goal', 0.0)):.3f}", "Advantage": "🅱️ Higher" if float(row_b.get("r_goal", 0.0)) > float(row_a.get("r_goal", 0.0)) else "🅰️ Higher"},
                {"Metric": "Assist Threat (r_assist / 90)", row_a["name"]: f"{float(row_a.get('r_assist', 0.0)):.3f}", row_b["name"]: f"{float(row_b.get('r_assist', 0.0)):.3f}", "Advantage": "🅱️ Higher" if float(row_b.get("r_assist", 0.0)) > float(row_a.get("r_assist", 0.0)) else "🅰️ Higher"},
                {"Metric": "Expected Minutes (xM)", row_a["name"]: f"{float(row_a.get('xM', 75.0)):.0f} mins", row_b["name"]: f"{float(row_b.get('xM', 75.0)):.0f} mins", "Advantage": "Equal" if float(row_a.get("xM", 75.0)) == float(row_b.get("xM", 75.0)) else ("🅱️ Higher" if float(row_b.get("xM", 75.0)) > float(row_a.get("xM", 75.0)) else "🅰️ Higher")},
                {"Metric": "Availability / Status", row_a["name"]: f"{int(row_a.get('chance_of_playing', 100))}% ({row_a.get('news', 'Fit') or 'Fit'})", row_b["name"]: f"{int(row_b.get('chance_of_playing', 100))}% ({row_b.get('news', 'Fit') or 'Fit'})", "Advantage": "—"}
            ]
            st.dataframe(pd.DataFrame(comp_table_data), hide_index=True, use_container_width=True)

            if d_cost < 0 and d_xp1 >= 0:
                verdict_msg = f"💡 **Dominant Value Swap**: Choosing **{row_b['name']}** over **{row_a['name']}** saves **£{abs(d_cost):.1f}m** while gaining **+{d_xp1:.2f} GW{start_gw} xP**! Highly recommended if you need funds to upgrade another slot."
            elif d_cost < 0 and d_xp1 < 0:
                cost_per_pt = abs(d_cost) / max(abs(d_xp1), 0.01)
                verdict_msg = f"💡 **Budget Downgrade Trade-Off**: Choosing **{row_b['name']}** frees up **£{abs(d_cost):.1f}m** at the cost of **{d_xp1:.2f} GW{start_gw} xP** (yielding £{cost_per_pt:.2f}m freed per point conceded)."
            elif d_cost > 0 and d_xp1 > 0:
                verdict_msg = f"💡 **Premium Upgrade**: Upgrading from **{row_a['name']}** to **{row_b['name']}** costs **£{d_cost:.1f}m** for an expected gain of **+{d_xp1:.2f} GW{start_gw} xP**."
            else:
                verdict_msg = f"💡 **Head-to-Head Comparison**: **{row_a['name']}** projects {xp1_a:.2f} xP (£{cost_a:.1f}m) vs **{row_b['name']}** with {xp1_b:.2f} xP (£{cost_b:.1f}m)."
            st.success(verdict_msg)

    # --- 2.2 INTERACTIVE SCOUT TABLE ---
    with subtab_scout_table:
        st.subheader("🔎 Interactive Player Scout & Multi-Filter Engine")
        st.caption("Search, filter, and sort every player across the Premier League by expected points, price tier, club, and underlying metrics.")

        if not mat_scout.empty:
            f_c1, f_c2, f_c3, f_c4 = st.columns(4)
            with f_c1: search_query = st.text_input("🔍 Search Name", "", key="scout_search", placeholder="e.g. Gabriel, Munoz, Salah...")
            with f_c2: pos_choice = st.selectbox("Position", ["All Positions", "GKP", "DEF", "MID", "FWD"], key="scout_pos")
            with f_c3:
                teams_list = ["All Teams"] + sorted(list(mat_scout["team"].dropna().unique()))
                team_choice = st.selectbox("Club Team", teams_list, key="scout_team")
            with f_c4: max_price = st.slider("Max Cost (£m)", min_value=4.0, max_value=15.0, value=15.0, step=0.5, key="scout_price")

            f_s1, f_s2 = st.columns(2)
            with f_s1:
                sort_choice = st.selectbox("Sort Table By", [
                    f"GW{start_gw} Projected xP (Highest first)",
                    "6-GW Horizon xP (Highest first)",
                    "Value Ratio (xP per £m)",
                    "Clean Sheet % (Highest first)",
                    "Goal Threat r_goal (Highest first)",
                    "Cost (Lowest first)",
                    "Cost (Highest first)"
                ], key="scout_sort")
            with f_s2: fit_only = st.checkbox("Only show fully fit players (100% chance)", value=True, key="scout_fit")

            filtered_mat = mat_scout.copy()
            if search_query: filtered_mat = filtered_mat[filtered_mat["name"].str.contains(search_query.strip(), case=False, na=False)]
            if pos_choice != "All Positions": filtered_mat = filtered_mat[filtered_mat["position"] == pos_choice]
            if team_choice != "All Teams": filtered_mat = filtered_mat[filtered_mat["team"] == team_choice]
            filtered_mat = filtered_mat[filtered_mat["cost"] <= max_price]
            if fit_only: filtered_mat = filtered_mat[filtered_mat["chance_of_playing"] >= 100]

            if "GW" in sort_choice and "Projected" in sort_choice: filtered_mat = filtered_mat.sort_values("GW1_xP", ascending=False)
            elif "6-GW" in sort_choice: filtered_mat = filtered_mat.sort_values("Horizon_xP", ascending=False)
            elif "Value" in sort_choice: filtered_mat = filtered_mat.sort_values("Value_Ratio", ascending=False)
            elif "Clean Sheet" in sort_choice: filtered_mat = filtered_mat.sort_values("team_cs_rate", ascending=False)
            elif "Goal Threat" in sort_choice: filtered_mat = filtered_mat.sort_values("r_goal", ascending=False)
            elif "Lowest" in sort_choice: filtered_mat = filtered_mat.sort_values("cost", ascending=True)
            elif "Highest" in sort_choice: filtered_mat = filtered_mat.sort_values("cost", ascending=False)

            display_scout_cols = ["name", "position", "team", "cost", "GW1_xP", "Horizon_xP", "Value_Ratio", "CS_Pct", "r_goal", "r_assist", "xM", "news"]
            rename_scout_dict = {
                "name": "Player Name", "position": "Pos", "team": "Team", "cost": "Cost (£m)",
                "GW1_xP": f"GW{start_gw} xP", "Horizon_xP": "6-GW xP", "Value_Ratio": "xP / £m",
                "CS_Pct": "Clean Sheet %", "r_goal": "rGoal/90", "r_assist": "rAssist/90", "xM": "xMins", "news": "News / Availability"
            }

            st.write(f"Showing **{len(filtered_mat)}** matching players:")
            st.dataframe(filtered_mat[display_scout_cols].rename(columns=rename_scout_dict), hide_index=True, use_container_width=True)

    # --- 2.3 AVAILABILITY & INJURIES ---
    with subtab_scout_injuries:
        st.header("🏥 Player Availability & Injury News Radar")
        st.caption("Live official FPL availability status, injury notes, suspension flags, and chance of playing percentages.")

        avail_df = get_player_availability_df()
        if not avail_df.empty:
            a1, a2, a3 = st.columns(3)
            total_flagged = len(avail_df[avail_df["chance_of_playing"] < 100])
            total_injured = len(avail_df[avail_df["chance_of_playing"] == 0])
            total_doubtful = len(avail_df[(avail_df["chance_of_playing"] > 0) & (avail_df["chance_of_playing"] < 100)])
            
            with a1: st.metric("Total Flagged Players", f"{total_flagged}")
            with a2: st.metric("Injured / Suspended (0%)", f"{total_injured}")
            with a3: st.metric("Doubtful Players (25%-75%)", f"{total_doubtful}")
            
            st.markdown("---")
            c_status, c_search = st.columns([1, 1])
            with c_status: status_filter = st.selectbox("Filter Availability Status", ["All Flagged Players", "Doubtful Only (25%-75%)", "Injured/Suspended (0%)", "All Players"], key="inj_status_filter")
            with c_search: search_p = st.text_input("🔍 Search Player or Team Name", "", key="inj_search_p")

            disp_avail = avail_df.copy()
            if status_filter == "All Flagged Players": disp_avail = disp_avail[disp_avail["chance_of_playing"] < 100]
            elif status_filter == "Doubtful Only (25%-75%)": disp_avail = disp_avail[(disp_avail["chance_of_playing"] > 0) & (disp_avail["chance_of_playing"] < 100)]
            elif status_filter == "Injured/Suspended (0%)": disp_avail = disp_avail[disp_avail["chance_of_playing"] == 0]

            if search_p:
                disp_avail = disp_avail[disp_avail["name"].str.contains(search_p, case=False, na=False) | disp_avail["team"].str.contains(search_p, case=False, na=False)]

            st.dataframe(
                disp_avail[["name", "position", "team", "cost", "availability_status", "chance_of_playing", "news_note"]].rename(columns={
                    "name": "Player Name", "position": "Pos", "cost": "Cost (£m)",
                    "availability_status": "Status Badge", "chance_of_playing": "Chance (%)", "news_note": "FPL Official News"
                }),
                hide_index=True,
                use_container_width=True
            )

    # --- 2.4 DAILY PRICE CHANGE & WEALTH RADAR ---
    with subtab_scout_price:
        st.header("📈 Daily Price Change & Wealth Accumulation Radar")
        st.caption("Tracks transfer flow velocity, net daily buy/sell volume, and proximity to the ±100% threshold for midnight FPL price changes.")

        radar_df = get_price_change_radar_df()
        if not radar_df.empty:
            rising_df = radar_df[radar_df["target_progress_pct"] >= 40.0].sort_values("target_progress_pct", ascending=False)
            falling_df = radar_df[radar_df["target_progress_pct"] <= -40.0].sort_values("target_progress_pct", ascending=True)

            top_riser = rising_df.iloc[0] if not rising_df.empty else None
            top_faller = falling_df.iloc[0] if not falling_df.empty else None

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                if top_riser is not None:
                    st.metric("🔥 Top Price Rise Target", f"{top_riser['name']}", delta=f"{top_riser['target_progress_pct']:+.1f}% Target")
                else:
                    st.metric("🔥 Top Price Rise Target", "None Imminent")
            with k2:
                if top_faller is not None:
                    st.metric("❄️ Top Price Fall Risk", f"{top_faller['name']}", delta=f"{top_faller['target_progress_pct']:.1f}% Target", delta_color="inverse")
                else:
                    st.metric("❄️ Top Price Fall Risk", "None Imminent")
            with k3:
                st.metric("High-Velocity Assets", f"{len(rising_df) + len(falling_df)} Players")
            with k4:
                st.metric("Model Wealth Policy", "Active (GW 1-15)", delta="Wealth Utility Enabled")

            st.markdown("---")
            pr_c1, pr_c2 = st.columns([1, 1])
            with pr_c1:
                pr_filter = st.selectbox("Filter Price Velocity", ["All Players", "Imminent Risers (≥ +90%)", "Buying Pressure (≥ +40%)", "Imminent Fallers (≤ -90%)", "Selling Pressure (≤ -40%)"], key="pr_filter")
            with pr_c2:
                pr_search = st.text_input("🔍 Search Player or Club", "", key="pr_search")

            disp_radar = radar_df.copy()
            if pr_filter == "Imminent Risers (≥ +90%)": disp_radar = disp_radar[disp_radar["target_progress_pct"] >= 90.0]
            elif pr_filter == "Buying Pressure (≥ +40%)": disp_radar = disp_radar[disp_radar["target_progress_pct"] >= 40.0]
            elif pr_filter == "Imminent Fallers (≤ -90%)": disp_radar = disp_radar[disp_radar["target_progress_pct"] <= -90.0]
            elif pr_filter == "Selling Pressure (≤ -40%)": disp_radar = disp_radar[disp_radar["target_progress_pct"] <= -40.0]

            if pr_search:
                disp_radar = disp_radar[disp_radar["name"].str.contains(pr_search, case=False, na=False) | disp_radar["team"].str.contains(pr_search, case=False, na=False)]

            disp_radar = disp_radar.sort_values("target_progress_pct", ascending=False)
            
            st.dataframe(
                disp_radar[["name", "position", "team", "cost", "selected_by_percent", "transfers_in_event", "transfers_out_event", "net_transfers", "target_progress_pct", "status_badge"]].rename(columns={
                    "name": "Player Name", "position": "Pos", "cost": "Cost (£m)",
                    "selected_by_percent": "Ownership %", "transfers_in_event": "Transfers In",
                    "transfers_out_event": "Transfers Out", "net_transfers": "Net Transfers",
                    "target_progress_pct": "Threshold %", "status_badge": "Price Change Status"
                }),
                hide_index=True,
                use_container_width=True
            )


# ==============================================================================
# HUB 3: FIXTURE & MARKET INTELLIGENCE
# ==============================================================================
with hub3:
    st.caption("Fixture and probability intelligence: Full 380-match schedule linked with ClubElo ratings, FDR pills, and bookmaker odds devigged using Shin's method.")
    
    subtab_fix_schedule, subtab_fix_odds = st.tabs([
        "📅 Full 380-Match Schedule & Elo Ratings",
        "🎲 De-Vigged Betting Odds (Shin's Method)"
    ])

    # --- 3.1 FULL FPL SCHEDULE ---
    with subtab_fix_schedule:
        st.header("📅 Full Premier League Schedule & Per-Game ClubElo Ratings")
        st.caption("Complete 380-match schedule linked directly with Home/Away ClubElo ratings, Net Elo Deltas (with +60 Home Advantage), and model clean sheet probabilities.")

        schedule_df = get_full_fpl_schedule()
        if not schedule_df.empty:
            col_s1, col_s2 = st.columns(2)
            with col_s1: gw_filter = st.selectbox("Filter Gameweek", ["All Gameweeks"] + [f"Gameweek {i}" for i in range(1, 39)], key="sch_gw_filter")
            with col_s2: team_filter = st.selectbox("Filter Team", ["All Teams"] + list(schedule_df["home_team"].dropna().unique()), key="sch_team_filter")

            filtered_sch = schedule_df.copy()
            if gw_filter != "All Gameweeks":
                gw_num = int(gw_filter.split()[1])
                filtered_sch = filtered_sch[filtered_sch["event"] == gw_num]
            if team_filter != "All Teams":
                filtered_sch = filtered_sch[(filtered_sch["home_team"] == team_filter) | (filtered_sch["away_team"] == team_filter)]

            display_cols = ["event", "kickoff_time", "matchup", "home_elo", "away_elo", "net_elo_delta", "home_cs_est", "away_cs_est", "team_h_difficulty", "team_a_difficulty"]
            valid_display = [c for c in display_cols if c in filtered_sch.columns]
            
            rename_dict = {
                "event": "GW", "kickoff_time": "Kickoff", "matchup": "Matchup",
                "home_elo": "Home Elo", "away_elo": "Away Elo", "net_elo_delta": "Net Δ Elo",
                "home_cs_est": "Home CS %", "away_cs_est": "Away CS %",
                "team_h_difficulty": "Home FDR", "team_a_difficulty": "Away FDR"
            }
            
            st.dataframe(filtered_sch[valid_display].rename(columns=rename_dict), hide_index=True, use_container_width=True)

    # --- 3.2 SHARP DE-VIGGED ODDS ---
    with subtab_fix_odds:
        st.header("🎲 Sharp Bookmaker Odds & De-Vigged Market Matrix")
        st.caption("De-vigs sharp UK bookmaker odds (Betfair/Bet365/Sky Bet) using Shin's algorithm to extract true market-implied win, clean sheet, and goal probabilities.")

        quota_info = get_odds_quota_info()
        q1, q2, q3, q4 = st.columns(4)
        with q1: st.metric("Live Market Feed", "The-Odds-API (UK)")
        with q2: st.metric("Remaining Free Quota", f"{quota_info['remaining']} / 500 req")
        with q3: st.metric("Odds Cache Age", f"{quota_info['age_hours']} hrs" if quota_info['age_hours'] < 900 else "Just Synced", delta="Fresh (5-Day / 120h Lock)" if quota_info['is_fresh'] else "Refresh Eligible")
        with q4: st.metric("Last Sync", quota_info['last_sync'].split()[0] if " " in quota_info['last_sync'] else quota_info['last_sync'])

        c_sync_btn, _ = st.columns([1, 3])
        with c_sync_btn:
            if st.button("🔄 Sync Live Odds (24h Throttled)", key="btn_sync_odds"):
                with st.spinner("Fetching latest Premier League bookmaker odds..."):
                    fetch_live_sharp_odds(force_refresh=True)
                    st.success("✅ Odds cache updated!")
                    st.rerun()

        odds_df = fetch_live_sharp_odds()
        if not odds_df.empty:
            display_odds_list = []
            for _, r in odds_df.iterrows():
                h_team = r.get("home_team", "")
                a_team = r.get("away_team", "")
                display_odds_list.append({
                    "Matchup": f"{h_team} vs {a_team}",
                    "Kickoff": str(r.get("commence_time", ""))[:16].replace("T", " "),
                    "Bookmaker": r.get("bookmaker", "Market Consensus"),
                    "Home Odds": f"{float(r.get('home_win_odds', 2.0)):.2f}",
                    "Draw Odds": f"{float(r.get('draw_odds', 3.2)):.2f}",
                    "Away Odds": f"{float(r.get('away_win_odds', 3.5)):.2f}",
                    "Home Win % (Shin)": f"{float(r.get('home_win_prob', 0.40))*100:.1f}%",
                    "Draw % (Shin)": f"{float(r.get('draw_prob', 0.28))*100:.1f}%",
                    "Away Win % (Shin)": f"{float(r.get('away_win_prob', 0.32))*100:.1f}%",
                    "Home Clean Sheet %": f"{float(r.get('home_cs_prob', 0.35))*100:.1f}%",
                    "Away Clean Sheet %": f"{float(r.get('away_cs_prob', 0.25))*100:.1f}%",
                    "Over 2.5": f"{float(r.get('over_25_odds', 1.85)):.2f}",
                    "Under 2.5": f"{float(r.get('under_25_odds', 1.95)):.2f}"
                })

            st.markdown("---")
            st.subheader("⚽ Live Premier League De-Vigged Match Probabilities (Shin's Method)")
            st.dataframe(pd.DataFrame(display_odds_list), hide_index=True, use_container_width=True)


        st.markdown("---")
        st.subheader("🎯 Gameweek 1 Anytime Goalscorer Market Odds & De-Vigged Goal Probabilities")
        goalscorer_data = [
            {"Player": "Erling Haaland", "Pos": "FWD", "Team": "Man City", "Opponent": "Bournemouth (H)", "Anytime Goal Odds": "1.65", "Raw Implied %": "60.6%", "De-Vigged Goal % (Shin)": "56.5%"},
            {"Player": "Alexander Isak", "Pos": "FWD", "Team": "Liverpool", "Opponent": "Newcastle (A)", "Anytime Goal Odds": "2.30", "Raw Implied %": "43.5%", "De-Vigged Goal % (Shin)": "39.2%"},
            {"Player": "Viktor Gyökeres", "Pos": "FWD", "Team": "Arsenal", "Opponent": "Coventry (H)", "Anytime Goal Odds": "2.35", "Raw Implied %": "42.6%", "De-Vigged Goal % (Shin)": "38.5%"},
            {"Player": "Bukayo Saka", "Pos": "MID", "Team": "Arsenal", "Opponent": "Coventry (H)", "Anytime Goal Odds": "2.40", "Raw Implied %": "41.7%", "De-Vigged Goal % (Shin)": "37.8%"},
            {"Player": "Cole Palmer", "Pos": "MID", "Team": "Chelsea", "Opponent": "Fulham (A)", "Anytime Goal Odds": "2.45", "Raw Implied %": "40.8%", "De-Vigged Goal % (Shin)": "37.0%"},
            {"Player": "Ollie Watkins", "Pos": "FWD", "Team": "Aston Villa", "Opponent": "Brighton (A)", "Anytime Goal Odds": "2.50", "Raw Implied %": "40.0%", "De-Vigged Goal % (Shin)": "36.0%"},
            {"Player": "Bruno Fernandes", "Pos": "MID", "Team": "Man Utd", "Opponent": "Hull City (A)", "Anytime Goal Odds": "2.50", "Raw Implied %": "40.0%", "De-Vigged Goal % (Shin)": "36.0%"},
            {"Player": "Igor Thiago", "Pos": "FWD", "Team": "Brentford", "Opponent": "Spurs (H)", "Anytime Goal Odds": "2.60", "Raw Implied %": "38.5%", "De-Vigged Goal % (Shin)": "34.5%"},
            {"Player": "Dominic Solanke", "Pos": "FWD", "Team": "Spurs", "Opponent": "Brentford (A)", "Anytime Goal Odds": "2.65", "Raw Implied %": "37.7%", "De-Vigged Goal % (Shin)": "33.8%"},
            {"Player": "João Pedro", "Pos": "FWD", "Team": "Chelsea", "Opponent": "Fulham (A)", "Anytime Goal Odds": "2.70", "Raw Implied %": "37.0%", "De-Vigged Goal % (Shin)": "33.1%"},
            {"Player": "Florian Wirtz", "Pos": "MID", "Team": "Liverpool", "Opponent": "Newcastle (A)", "Anytime Goal Odds": "3.10", "Raw Implied %": "32.3%", "De-Vigged Goal % (Shin)": "28.5%"},
            {"Player": "Rayan Cherki", "Pos": "MID", "Team": "Man City", "Opponent": "Bournemouth (H)", "Anytime Goal Odds": "3.20", "Raw Implied %": "31.3%", "De-Vigged Goal % (Shin)": "27.6%"},
        ]
        st.dataframe(pd.DataFrame(goalscorer_data), hide_index=True, use_container_width=True)

# ==============================================================================
# HUB 4: QUANTITATIVE STRATEGY & BACKTEST LAB
# ==============================================================================
with hub4:
    st.caption("Advanced quantitative research: Fact-check media claims with NLP, backtest algorithm performance across multi-gameweek horizons, and simulate custom risk scenarios.")
    
    subtab_lab_article, subtab_lab_backtest, subtab_lab_whatif = st.tabs([
        "📰 Article Sentiment & Fact-Checker",
        "📈 Walk-Forward Backtest Simulator",
        "🎛️ What-If Scenario Optimization"
    ])

    # --- 4.1 ARTICLE SENTIMENT & FACT-CHECKER ---
    with subtab_lab_article:
        st.header("📰 Article Sentiment & Data-Driven Counter-Argument Engine")
        st.caption("Paste publication articles, expert quotes, or media claims. Evaluates claims using quantitative rates (xG/xA/xM/Elo) and allows custom rate overrides.")

        uploaded_file = st.file_uploader("Or Upload Article / Text Document (.txt, .md, .pdf):", type=["txt", "md", "pdf"], key="article_file_upload")

        user_article = st.text_area(
            "Paste Publication Article / Expert Quotes below:",
            height=150,
            placeholder="e.g. 'Haaland is indispensable for GW1 against Bournemouth... Cole Palmer is underpriced at £9.5m on penalties...'",
            key="article_input_text"
        )

        # Extract text from uploaded file if present
        combined_text = user_article
        if uploaded_file is not None:
            try:
                fname = uploaded_file.name.lower()
                if fname.endswith(".pdf"):
                    raw_bytes = uploaded_file.read()
                    streams = re.findall(b'stream[\r\n]+(.*?)[\r\n]+endstream', raw_bytes, re.DOTALL)
                    pdf_chunks = []
                    for s in streams:
                        decoded = re.findall(rb'\((.*?)\)[\s]*T[jJ]', s)
                        for d in decoded:
                            try:
                                pdf_chunks.append(d.decode('latin1'))
                            except Exception:
                                pass
                    pdf_text = " ".join(pdf_chunks)
                    combined_text = (user_article + "\n\n" + pdf_text).strip()
                else:
                    file_text = uploaded_file.read().decode("utf-8", errors="ignore")
                    combined_text = (user_article + "\n\n" + file_text).strip()
            except Exception as e:
                st.warning(f"Could not parse uploaded file: {e}")

        if st.button("🔍 Analyze Article & Generate Quantitative Counter-Arguments", key="btn_analyze_art"):
            if combined_text.strip() and not matrix.empty:
                # Safe API Key extraction without crashing when secrets.toml is absent on Cloud Run
                api_key = None
                try:
                    api_key = st.secrets.get("GEMINI_API_KEY")
                except Exception:
                    api_key = None
                if not api_key:
                    api_key = os.getenv("GEMINI_API_KEY")

                analyzer = ArticleSentimentEngine(matrix, api_key=api_key)
                analysis_df = analyzer.analyze_article_llm(combined_text)
                
                if not analysis_df.empty:
                    mode_label = "🤖 Gemini 2.5 Flash LLM" if analyzer.client else "⚡ Local Rule-Based NLP"
                    st.success(f"✅ Successfully extracted {len(analysis_df)} players using {mode_label} engine!")
                    st.subheader(f"Found {len(analysis_df)} Players Mentioned in Article:")
                    for _, row in analysis_df.iterrows():
                        v_type = row["verdict"]
                        box_class = "verdict-supported" if "SUPPORTED" in v_type else ("verdict-skeptical" if "SKEPTICAL" in v_type else "verdict-caution")
                        
                        verdict_html = textwrap.dedent(f"""
                        <div class="{box_class}">
                            <h4 style="margin:0 0 6px 0;">{row['name']} ({row['position']}, {row['team']} - £{row['cost']:.1f}m)</h4>
                            <div><strong>Verdict:</strong> {row['verdict']}</div>
                            <div style="margin-top:4px;"><strong>Quantitative Analysis:</strong> {row['quantitative_reasoning']}</div>
                        </div>
                        """).strip()
                        st.markdown(verdict_html, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("⚙️ Custom Rate Multiplier Overrides")
                    st.caption("Adjust player multipliers to override expected points before running the solver.")
                    
                    for _, row in analysis_df.iterrows():
                        pid = row["player_id"]
                        curr_val = st.session_state["manual_overrides"].get(pid, row["suggested_multiplier"])
                        new_val = st.slider(f"Rate Multiplier for {row['name']} ({row['team']})", min_value=0.5, max_value=2.0, value=float(curr_val), step=0.05, key=f"sl_override_{pid}")
                        st.session_state["manual_overrides"][pid] = new_val
                    
                    st.success("✅ Rate overrides saved in session state! Future solver runs will use these custom weights.")
                else:
                    st.info("No active Premier League players recognized in the provided text.")


    # --- 4.2 BACKTEST SIMULATION ---
    with subtab_lab_backtest:
        st.header("📈 Stateful Walk-Forward Backtest Simulation")
        st.caption("Validates quantitative policy against historic gameweek data with realistic rolling horizon transfers, formation-valid auto-subs, and chip activations.")

        bk_c1, bk_c2 = st.columns([1, 1])
        with bk_c1:
            bk_start = st.number_input("Start Gameweek", min_value=2, max_value=37, value=2, step=1, key="bk_start_gw")
        with bk_c2:
            bk_end = st.number_input("End Gameweek", min_value=2, max_value=38, value=8, step=1, key="bk_end_gw")
        
        if st.button("🚀 Run Walk-Forward Backtest", type="primary", key="btn_run_backtest"):
            with st.spinner("Running sequential multi-gameweek simulation with formation-valid substitutions..."):
                harness = WalkForwardBacktestHarness(start_gw=int(bk_start), end_gw=int(bk_end), history_df=history_df, simulate_chips=True)
                res_df = harness.run_simulation(verbose=False)
                if not res_df.empty:
                    st.session_state["backtest_results_df"] = res_df

        if "backtest_results_df" in st.session_state and not st.session_state["backtest_results_df"].empty:
            res_df = st.session_state["backtest_results_df"].copy()
            
            # Benchmark Trajectories
            res_df["Top 10k Benchmark"] = (res_df["gameweek"] - res_df["gameweek"].min() + 1) * 65.0
            res_df["World #1 Champion Benchmark"] = (res_df["gameweek"] - res_df["gameweek"].min() + 1) * 70.5
            res_df["Global Average Benchmark"] = (res_df["gameweek"] - res_df["gameweek"].min() + 1) * 48.0
            res_df["Overfit FPL SOTA Engine"] = res_df["cumulative_points"]

            final_model_pts = res_df["Overfit FPL SOTA Engine"].iloc[-1]
            final_top10k_pts = res_df["Top 10k Benchmark"].iloc[-1]
            final_avg_pts = res_df["Global Average Benchmark"].iloc[-1]
            gw_count = len(res_df)
            avg_ppg = final_model_pts / max(1, gw_count)

            b1, b2, b3, b4 = st.columns(4)
            with b1: st.metric("Cumulative SOTA Points", f"{final_model_pts:.0f} pts", delta=f"{avg_ppg:.1f} pts / GW")
            with b2: st.metric("vs. Top 10k Benchmark", f"{final_top10k_pts:.0f} pts", delta=f"{final_model_pts - final_top10k_pts:+.0f} pts")
            with b3: st.metric("vs. Global Average", f"{final_avg_pts:.0f} pts", delta=f"{final_model_pts - final_avg_pts:+.0f} pts")
            with b4: st.metric("Total Hits Deducted", f"-{res_df['hits_cost'].sum()} pts", delta="Controlled Hit Budget")

            st.markdown("---")
            st.subheader("📊 Cumulative Equity Curves: Model vs Global Benchmarks")
            st.caption("Visualizes sequential cumulative point accumulation vs. Top 10k, World Champion (#1), and Global Average benchmarks.")
            
            chart_df = res_df.set_index("gameweek")[["Overfit FPL SOTA Engine", "World #1 Champion Benchmark", "Top 10k Benchmark", "Global Average Benchmark"]]
            st.line_chart(chart_df, color=["#00ff87", "#ffd700", "#38bdf8", "#94a3b8"])

            st.markdown("---")
            st.subheader("📋 Gameweek Execution Ledger & Auto-Sub Log")
            disp_ledger = res_df[["gameweek", "sota_engine_points", "active_chip", "transfers_made", "hits_cost", "auto_subs_used", "bank", "free_transfers", "cumulative_points"]].rename(columns={
                "gameweek": "Gameweek", "sota_engine_points": "GW Points", "active_chip": "Active Chip",
                "transfers_made": "Transfers", "hits_cost": "Hit Cost", "auto_subs_used": "Auto-Subs Used",
                "bank": "Bank (£m)", "free_transfers": "Free Transfers", "cumulative_points": "Cumulative Points"
            })
            st.dataframe(disp_ledger, hide_index=True, use_container_width=True)


    # --- 4.3 WHAT-IF SCENARIO STUDIO ---
    with subtab_lab_whatif:
        st.subheader("🎛️ Interactive What-If Scenario Studio & Risk Optimization")
        st.caption("Simulate custom budget allocations, double-defense variance penalties, and solve customized MILP squads in real time.")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            scenario_budget = st.number_input("Scenario Budget (£m)", min_value=90.0, max_value=105.0, value=100.0, step=0.5, key="sc_budget")
        with sc2:
            scenario_risk = st.slider("Double-Defense Risk Aversion (λ)", min_value=0.0, max_value=0.5, value=0.15, step=0.05, key="sc_risk")
        with sc3:
            scenario_fmt = st.selectbox("Target Formation", ["Automatic", "3-4-3", "3-5-2", "4-4-2", "4-3-3", "4-5-1", "5-3-2"], key="sc_fmt")

        st.markdown("---")
        if st.button("🚀 Solve Customized What-If Squad", key="btn_solve_whatif"):
            with st.spinner("Building custom rate engine & solving MILP..."):
                sc_squad, sc_bank = build_gw1_start_of_season_squad(
                    history_df,
                    budget=scenario_budget,
                    formation=scenario_fmt,
                    risk_aversion=scenario_risk
                )
                
                if not sc_squad.empty:
                    sc_xp = sc_squad["GW1_xP"].fillna(0).sum()
                    m1, m2, m3 = st.columns(3)
                    with m1: st.metric("Total Projected GW1 xP", f"{sc_xp:.2f} pts")
                    with m2: st.metric("Remaining Bank", f"£{sc_bank:.1f}m")
                    with m3: st.metric("Double-Defense Stacking Penalty", f"λ = {scenario_risk:.2f}")

                    st.markdown("---")
                    st.subheader("📋 Scenario Squad Cards & Rationales")
                    for _, p_row in sc_squad.iterrows():
                        render_player_card(p_row)
