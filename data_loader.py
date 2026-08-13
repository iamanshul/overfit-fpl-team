# -*- coding: utf-8 -*-
"""
Data Ingestion Pipeline for Jetski FPL Engine (data_loader.py)
Handles API syncs with browser headers, SQLite warehousing, and CSV fallbacks.
"""

import os
import io
import time
import datetime
import sqlite3
import pandas as pd
import numpy as np
import requests
from config import (
    DB_PATH, CSV_PATH, ELO_CACHE_PATH, HTTP_HEADERS, DATA_DIR,
    ODDS_API_KEY, ODDS_API_URL, ODDS_CACHE_PATH, ODDS_CACHE_TTL_HOURS, ODDS_MIN_REFRESH_INTERVAL_HOURS
)
from devig_engine import SharpOddsEngine

FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
CLUB_ELO_URL = "http://api.clubelo.com/"

FPL_TO_ELO_MAP = {
    'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa', 'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford', 'Brighton': 'Brighton', 'Chelsea': 'Chelsea',
    'Crystal Palace': 'Crystal Palace', 'Everton': 'Everton', 'Fulham': 'Fulham',
    'Ipswich': 'Ipswich', 'Leicester': 'Leicester', 'Liverpool': 'Liverpool',
    'Man City': 'Man City', 'Man Utd': 'Man United', 'Newcastle': 'Newcastle',
    'Nott\'m Forest': 'Nottingham', 'Southampton': 'Southampton',
    'Spurs': 'Tottenham', 'West Ham': 'West Ham', 'Wolves': 'Wolverhampton'
}

def get_db_connection(commit=False):
    """Context manager helper for SQLite DB connection."""
    conn = sqlite3.connect(DB_PATH)
    return conn

import concurrent.futures

def initialize_database():
    """Ensures SQLite database tables exist and have up-to-date schema."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY,
                name TEXT,
                short_name TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS players_meta (
                id INTEGER PRIMARY KEY,
                web_name TEXT,
                team_id INTEGER,
                position TEXT,
                now_cost INTEGER,
                status TEXT,
                news TEXT,
                chance_of_playing INTEGER,
                selected_by_percent REAL DEFAULT 0.0,
                transfers_in_event INTEGER DEFAULT 0,
                transfers_out_event INTEGER DEFAULT 0,
                penalties_order INTEGER DEFAULT 0,
                direct_freekicks_order INTEGER DEFAULT 0,
                corners_and_indirect_freekicks_order INTEGER DEFAULT 0,
                has_midweek_uefa INTEGER DEFAULT 0
            )
        """)
        # Schema migration checks for players_meta
        cur.execute("PRAGMA table_info(players_meta)")
        cols = [col[1] for col in cur.fetchall()]
        if 'selected_by_percent' not in cols:
            cur.execute("ALTER TABLE players_meta ADD COLUMN selected_by_percent REAL DEFAULT 0.0")
        if 'transfers_in_event' not in cols:
            cur.execute("ALTER TABLE players_meta ADD COLUMN transfers_in_event INTEGER DEFAULT 0")
        if 'transfers_out_event' not in cols:
            cur.execute("ALTER TABLE players_meta ADD COLUMN transfers_out_event INTEGER DEFAULT 0")
        if 'penalties_order' not in cols:
            cur.execute("ALTER TABLE players_meta ADD COLUMN penalties_order INTEGER DEFAULT 0")
        if 'direct_freekicks_order' not in cols:
            cur.execute("ALTER TABLE players_meta ADD COLUMN direct_freekicks_order INTEGER DEFAULT 0")
        if 'corners_and_indirect_freekicks_order' not in cols:
            cur.execute("ALTER TABLE players_meta ADD COLUMN corners_and_indirect_freekicks_order INTEGER DEFAULT 0")
        if 'has_midweek_uefa' not in cols:
            cur.execute("ALTER TABLE players_meta ADD COLUMN has_midweek_uefa INTEGER DEFAULT 0")
        if 'age' not in cols:
            cur.execute("ALTER TABLE players_meta ADD COLUMN age INTEGER DEFAULT 26")
        if 'height_cm' not in cols:
            cur.execute("ALTER TABLE players_meta ADD COLUMN height_cm INTEGER DEFAULT 182")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS meta_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp DATETIME
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sharp_odds_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_team TEXT,
                away_team TEXT,
                commence_time TEXT,
                home_win_odds REAL,
                draw_odds REAL,
                away_win_odds REAL,
                over_25_odds REAL,
                under_25_odds REAL,
                home_win_prob REAL,
                draw_prob REAL,
                away_win_prob REAL,
                home_cs_prob REAL,
                away_cs_prob REAL,
                bookmaker TEXT,
                last_updated DATETIME
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fixtures (
                id INTEGER PRIMARY KEY,
                event INTEGER,
                team_h INTEGER,
                team_a INTEGER,
                kickoff_time DATETIME,
                finished INTEGER,
                team_h_difficulty INTEGER,
                team_a_difficulty INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS player_match_history (
                element_id INTEGER,
                fixture_id INTEGER,
                round INTEGER,
                kickoff_time DATETIME,
                opponent_team INTEGER,
                was_home INTEGER,
                total_points INTEGER,
                minutes INTEGER,
                goals_scored INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                goals_conceded INTEGER,
                yellow_cards INTEGER,
                red_cards INTEGER,
                bonus INTEGER,
                bps INTEGER,
                influence REAL,
                creativity REAL,
                threat REAL,
                expected_goals REAL,
                expected_assists REAL,
                expected_goal_involvements REAL,
                expected_goals_conceded REAL,
                value INTEGER,
                transfers_balance INTEGER DEFAULT 0,
                selected INTEGER DEFAULT 0,
                PRIMARY KEY (element_id, fixture_id)
            )
        """)
        # Composite Performance Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pmh_element_time ON player_match_history(element_id, kickoff_time);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_event ON fixtures(event, kickoff_time);")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_pmh_element_kickoff ON player_match_history(element_id, kickoff_time)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pmh_round ON player_match_history(round)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_event ON fixtures(event, kickoff_time)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pm_team_pos ON players_meta(team_id, position)")

        conn.commit()
    finally:
        conn.close()

def fetch_player_history_parallel(elements):
    """Fetches full match history per element from FPL API in parallel."""
    print(f"⚡ Fetching match history for {len(elements)} players in parallel...")
    all_history = []
    
    def fetch_single(pid):
        try:
            r = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{pid}/", headers=HTTP_HEADERS, timeout=6)
            if r.status_code == 200:
                hist = r.json().get('history', [])
                for row in hist:
                    row['element_id'] = pid
                return hist
        except Exception:
            pass
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(fetch_single, p['id']) for p in elements]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                all_history.extend(res)
    return all_history

def sync_fpl_api_data():
    """Fetches official FPL static data, player match histories, and all 380 fixtures using proper headers."""
    initialize_database()
    try:
        resp = requests.get(FPL_API_URL, headers=HTTP_HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"⚠️ FPL API returned status {resp.status_code}. Using local warehouse.")
            return False
            
        data = resp.json()
        conn = get_db_connection()
        try:
            # Sync Teams
            team_rows = [(t['id'], t['name'], t['short_name']) for t in data['teams']]
            conn.executemany("REPLACE INTO teams (id, name, short_name) VALUES (?,?,?)", team_rows)
            
            # Identification of UEFA teams playing midweek European fixtures
            uefa_teams_short = ['ARS', 'AVL', 'CHE', 'LIV', 'MCI', 'MUN', 'TOT']
            team_short_map = {t['id']: t['short_name'] for t in data['teams']}

            pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            meta_rows = []
            for p in data['elements']:
                pos = pos_map.get(p['element_type'], 'MID')
                chance = p['chance_of_playing_next_round'] if p['chance_of_playing_next_round'] is not None else 100
                sel_pct = float(p.get('selected_by_percent', 0.0))
                tr_in = p.get('transfers_in_event', 0)
                tr_out = p.get('transfers_out_event', 0)
                pen_ord = p.get('penalties_order', 0) or 0
                fk_ord = p.get('direct_freekicks_order', 0) or 0
                ck_ord = p.get('corners_and_indirect_freekicks_order', 0) or 0
                t_short = team_short_map.get(p['team'], '')
                has_uefa = 1 if t_short in uefa_teams_short else 0
                
                meta_rows.append((
                    p['id'], p['web_name'], p['team'], pos, p['now_cost'], p['status'], p['news'],
                    chance, sel_pct, tr_in, tr_out, pen_ord, fk_ord, ck_ord, has_uefa
                ))
            conn.executemany("""
                REPLACE INTO players_meta (
                    id, web_name, team_id, position, now_cost, status, news, chance_of_playing,
                    selected_by_percent, transfers_in_event, transfers_out_event,
                    penalties_order, direct_freekicks_order, corners_and_indirect_freekicks_order, has_midweek_uefa
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, meta_rows)
            
            # Sync All 380 Fixtures
            resp_fix = requests.get(FIXTURES_URL, headers=HTTP_HEADERS, timeout=10)
            if resp_fix.status_code == 200:
                fix_data = resp_fix.json()
                fix_rows = []
                for f in fix_data:
                    fix_rows.append((
                        f.get('id'), f.get('event'), f.get('team_h'), f.get('team_a'),
                        f.get('kickoff_time'), 1 if f.get('finished') else 0,
                        f.get('team_h_difficulty', 3), f.get('team_a_difficulty', 3)
                    ))
                conn.executemany("REPLACE INTO fixtures (id, event, team_h, team_a, kickoff_time, finished, team_h_difficulty, team_a_difficulty) VALUES (?,?,?,?,?,?,?,?)", fix_rows)
                print(f"✅ Synced {len(fix_rows)} Premier League fixtures.")

            # Sync Player Match History in Parallel
            history_data = fetch_player_history_parallel(data['elements'])
            if history_data:
                hist_rows = []
                for h in history_data:
                    hist_rows.append((
                        h.get('element_id'), h.get('fixture'), h.get('round'), h.get('kickoff_time'),
                        h.get('opponent_team'), 1 if h.get('was_home') else 0, h.get('total_points', 0),
                        h.get('minutes', 0), h.get('goals_scored', 0), h.get('assists', 0),
                        h.get('clean_sheets', 0), h.get('goals_conceded', 0), h.get('yellow_cards', 0),
                        h.get('red_cards', 0), h.get('bonus', 0), h.get('bps', 0),
                        float(h.get('influence', 0.0)), float(h.get('creativity', 0.0)), float(h.get('threat', 0.0)),
                        float(h.get('expected_goals', 0.0)), float(h.get('expected_assists', 0.0)),
                        float(h.get('expected_goal_involvements', 0.0)), float(h.get('expected_goals_conceded', 0.0)),
                        h.get('value', 50), h.get('transfers_balance', 0), h.get('selected', 0)
                    ))
                conn.executemany("""
                    REPLACE INTO player_match_history (
                        element_id, fixture_id, round, kickoff_time, opponent_team, was_home,
                        total_points, minutes, goals_scored, assists, clean_sheets, goals_conceded,
                        yellow_cards, red_cards, bonus, bps, influence, creativity, threat,
                        expected_goals, expected_assists, expected_goal_involvements, expected_goals_conceded,
                        value, transfers_balance, selected
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, hist_rows)
                print(f"✅ Synced {len(hist_rows)} historical match records from API.")
            else:
                # Fallback to historical CSV if DB is empty or off-season
                cur = conn.cursor()
                cur.execute("SELECT count(*) FROM player_match_history")
                if cur.fetchone()[0] == 0 and os.path.exists(CSV_PATH):
                    print(f"   -> Copying historical dataset from {CSV_PATH} into SQLite DB...")
                    csv_df = pd.read_csv(CSV_PATH)
                    
                    # Remap historical CSV element_id to 2025/26 live players_meta IDs via web_name matching
                    meta_lookup = pd.read_sql("SELECT id as live_id, web_name, position FROM players_meta", conn)
                    name_to_id = dict(zip(meta_lookup['web_name'], meta_lookup['live_id']))
                    
                    exact_map = {
                        'B.Fernandes': 426, # Bruno Fernandes (Man Utd, £12.0m)
                        'Bruno G.': 452,    # Bruno Guimarães (Newcastle, £7.0m)
                        'Palmer': 154       # Cole Palmer (Chelsea, £9.5m)
                    }

                    def map_name_to_id(row):
                        pname = str(row.get('player_name', row.get('name', ''))).strip()
                        if pname in exact_map:
                            return exact_map[pname]
                        if pname in ['M.Fernandes', 'Fernandes'] and row.get('team_name') in ['Man Utd', 'MUN']:
                            return 426
                        pname_clean = pname.replace('J.Maddison', 'Maddison').replace('D.Welbeck', 'Welbeck')
                        if pname_clean in name_to_id:
                            return name_to_id[pname_clean]
                        return row.get('element', row.get('element_id', None))

                    csv_df['element_id'] = csv_df.apply(map_name_to_id, axis=1)
                    if 'fixture' in csv_df.columns and 'fixture_id' not in csv_df.columns:
                        csv_df['fixture_id'] = csv_df['fixture']
                    
                    # Align columns and drop duplicates
                    db_cols = ['element_id', 'fixture_id', 'round', 'kickoff_time', 'opponent_team', 'was_home',
                               'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
                               'yellow_cards', 'red_cards', 'bonus', 'bps', 'influence', 'creativity', 'threat',
                               'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
                               'value', 'transfers_balance', 'selected']
                    avail_cols = [c for c in db_cols if c in csv_df.columns]
                    csv_clean = csv_df[avail_cols].dropna(subset=['element_id']).drop_duplicates(subset=['element_id', 'fixture_id'])
                    csv_clean.to_sql('player_match_history', conn, if_exists='append', index=False)
                    print(f"✅ Loaded {len(csv_clean)} historical records with 2025/26 ID re-mapping from CSV fallback.")

            # Save Last Updated Timestamp in meta_cache
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_players = str(len(data['elements']))
            conn.execute("REPLACE INTO meta_cache (key, value, timestamp) VALUES ('last_updated', ?, ?)", (now_str, now_str))
            conn.execute("REPLACE INTO meta_cache (key, value, timestamp) VALUES ('total_players', ?, ?)", (total_players, now_str))
            conn.commit()
            print(f"✅ Successfully synced live FPL data at {now_str}.")
            return True
        finally:
            conn.close()
    except Exception as e:
        print(f"⚠️ FPL API sync error: {e}")
        return False

def get_last_updated_info(max_fresh_hours=2.0):
    """Returns dictionary with last updated timestamp, age in hours, and freshness boolean."""
    initialize_database()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT value, timestamp FROM meta_cache WHERE key = 'last_updated'")
        row = cur.fetchone()
        if row:
            ts_str = row[0]
            try:
                dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                age_hours = (datetime.datetime.now() - dt).total_seconds() / 3600.0
            except Exception:
                dt = None
                age_hours = 999.0
            return {
                "last_updated": ts_str,
                "age_hours": round(age_hours, 1),
                "is_fresh": age_hours < max_fresh_hours
            }
        else:
            return {
                "last_updated": "Never (Initial Setup)",
                "age_hours": 999.0,
                "is_fresh": False
            }
    finally:
        conn.close()

def check_and_auto_update_data(max_age_hours=2.0):
    """Auto updates FPL API data if older than max_age_hours."""
    info = get_last_updated_info(max_fresh_hours=max_age_hours)
    if not info["is_fresh"] or info["age_hours"] >= max_age_hours:
        print(f"🔄 Data is {info['age_hours']} hours old (threshold: {max_age_hours}h). Triggering auto-update...")
        return sync_fpl_api_data()
    return False

def fetch_clubelo_ratings(target_date=None):
    """Fetches Elo ratings from ClubElo for current or historical target_date (YYYY-MM-DD) with cache fallback."""
    if target_date is None and os.path.exists(ELO_CACHE_PATH):
        age_days = (time.time() - os.path.getmtime(ELO_CACHE_PATH)) / 86400.0
        if age_days < 2.0:
            try:
                df = pd.read_csv(ELO_CACHE_PATH)
                return dict(zip(df['Team'], df['Elo']))
            except Exception:
                pass

    query_date = target_date or (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    try:
        resp = requests.get(f"{CLUB_ELO_URL}{query_date}", headers=HTTP_HEADERS, timeout=10)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text), header=None, names=['Rank', 'Team', 'country', 'level', 'Elo', 'From', 'To'], on_bad_lines='skip')
            df = df[df['country'] == 'ENG']
            elo_dict = dict(zip(df['Team'], df['Elo']))
            
            if target_date is None:
                df_cache = df[['Team', 'Elo']].copy()
                df_cache.to_csv(ELO_CACHE_PATH, index=False)
            return elo_dict
    except Exception as e:
        print(f"⚠️ ClubElo fetch error for {query_date}: {e}")

    # Default fallback Elo map
    default_elo = {
        'Arsenal': 2020, 'Aston Villa': 1850, 'Bournemouth': 1740, 'Brentford': 1760,
        'Brighton': 1810, 'Chelsea': 1840, 'Crystal Palace': 1730, 'Everton': 1700,
        'Fulham': 1750, 'Ipswich': 1610, 'Leicester': 1630, 'Liverpool': 2030,
        'Man City': 2070, 'Man Utd': 1790, 'Newcastle': 1850, 'Nott\'m Forest': 1740,
        'Southampton': 1610, 'Spurs': 1860, 'West Ham': 1730, 'Wolves': 1680
    }
    return default_elo

def load_player_history():
    """Loads player match dataset from SQLite DB, automatically syncing from live FPL API if DB is uninitialized."""
    initialize_database()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM player_match_history")
        count = cur.fetchone()[0]
        if count == 0:
            print("📦 Uninitialized database detected. Auto-syncing live official Premier League data...")
            conn.close()
            sync_fpl_api_data()
            conn = get_db_connection()

        df = pd.read_sql("SELECT * FROM player_match_history", conn)
        if not df.empty:
            meta = pd.read_sql("""
                SELECT m.id as element_id, m.web_name as name, m.position, m.now_cost, m.status, m.news, m.chance_of_playing,
                       m.selected_by_percent, m.transfers_in_event, m.transfers_out_event,
                       m.penalties_order, m.direct_freekicks_order, m.corners_and_indirect_freekicks_order,
                       m.has_midweek_uefa, m.age, m.height_cm, t.name as team
                FROM players_meta m
                LEFT JOIN teams t ON m.team_id = t.id
            """, conn)
            teams_opp = pd.read_sql("SELECT id as opponent_team_id, name as opponent_team_name FROM teams", conn)
            
            # Drop stale team/name/cost columns from history prior to merge so live meta takes priority
            cols_to_drop = [c for c in ['team', 'team_name', 'name', 'position', 'cost', 'now_cost', 'status', 'news', 'chance_of_playing', 'age', 'height_cm'] if c in df.columns]
            df = df.drop(columns=cols_to_drop)
            
            df = df.merge(meta, on='element_id', how='left')
            df = df.merge(teams_opp, left_on='opponent_team', right_on='opponent_team_id', how='left')
            if 'opponent_team_id' in df.columns:
                df = df.drop(columns=['opponent_team_id'])
            df['player_id'] = df['element_id']
            df['gameweek'] = df['round']
            df['cost'] = df['now_cost'] / 10.0
            return df
    except Exception as e:
        print(f"⚠️ Error reading DB player history: {e}")
    finally:
        conn.close()

    # Guaranteed fallback to packaged CSV dataset
    if os.path.exists(CSV_PATH):
        try:
            df_csv = pd.read_csv(CSV_PATH)
            if 'player_name' in df_csv.columns and 'name' not in df_csv.columns:
                df_csv['name'] = df_csv['player_name']
            if 'team_name' in df_csv.columns and 'team' not in df_csv.columns:
                df_csv['team'] = df_csv['team_name']
            if 'round' in df_csv.columns and 'gameweek' not in df_csv.columns:
                df_csv['gameweek'] = df_csv['round']
            if 'element_id' in df_csv.columns and 'player_id' not in df_csv.columns:
                df_csv['player_id'] = df_csv['element_id']
            if 'player_id' in df_csv.columns and 'element_id' not in df_csv.columns:
                df_csv['element_id'] = df_csv['player_id']
            if 'cost' not in df_csv.columns:
                if 'now_cost' in df_csv.columns:
                    df_csv['cost'] = df_csv['now_cost'] / 10.0
                elif 'value' in df_csv.columns:
                    df_csv['cost'] = df_csv['value'] / 10.0
                else:
                    df_csv['cost'] = 5.0
            return df_csv
        except Exception as e:
            print(f"⚠️ Error reading CSV player history: {e}")

    return pd.DataFrame()

ODDS_TO_FPL_TEAM_MAP = {
    'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa', 'Bournemouth': 'Bournemouth',
    'AFC Bournemouth': 'Bournemouth', 'Brentford': 'Brentford', 'Brighton': 'Brighton',
    'Brighton and Hove Albion': 'Brighton', 'Chelsea': 'Chelsea', 'Coventry City': 'Coventry City',
    'Coventry': 'Coventry City', 'Crystal Palace': 'Crystal Palace', 'Everton': 'Everton',
    'Fulham': 'Fulham', 'Hull City': 'Hull City', 'Hull': 'Hull City',
    'Ipswich Town': 'Ipswich Town', 'Ipswich': 'Ipswich Town', 'Leeds': 'Leeds',
    'Leeds United': 'Leeds', 'Liverpool': 'Liverpool', 'Manchester City': 'Man City',
    'Man City': 'Man City', 'Manchester United': 'Man Utd', 'Man United': 'Man Utd',
    'Newcastle United': 'Newcastle', 'Newcastle': 'Newcastle', 'Nottingham Forest': "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest", 'Tottenham Hotspur': 'Spurs', 'Tottenham': 'Spurs',
    'Spurs': 'Spurs', 'Sunderland': 'Sunderland', 'Sunderland AFC': 'Sunderland',
    'West Ham United': 'West Ham', 'West Ham': 'West Ham', 'Wolverhampton Wanderers': 'Wolves',
    'Wolves': 'Wolves', 'Leicester City': 'Leicester', 'Leicester': 'Leicester',
    'Southampton': 'Southampton'
}

def generate_fallback_odds_from_elo():
    """Generates synthetic devigged market odds from live ClubElo ratings if API is unreachable or outside 48h deadline window."""
    elo_dict = fetch_clubelo_ratings()
    fixtures = get_full_fpl_schedule()
    rows = []
    if not fixtures.empty:
        gw1 = fixtures[fixtures['event'] == 1]
        for _, r in gw1.iterrows():
            h_team, a_team = r['home_team'], r['away_team']
            h_elo = float(elo_dict.get(h_team, 1750.0))
            a_elo = float(elo_dict.get(a_team, 1750.0))
            net_delta = (h_elo + 60.0) - a_elo
            
            p_h = np.clip(0.38 + net_delta / 800.0, 0.10, 0.85)
            p_a = np.clip(0.34 - net_delta / 800.0, 0.05, 0.75)
            p_d = max(0.10, 1.0 - p_h - p_a)
            tot_p = p_h + p_d + p_a
            p_h, p_d, p_a = p_h/tot_p, p_d/tot_p, p_a/tot_p
            
            cs_h = np.clip(0.25 + net_delta / 2000.0, 0.10, 0.65)
            cs_a = np.clip(0.20 - net_delta / 2000.0, 0.05, 0.55)
            
            rows.append({
                "home_team": h_team,
                "away_team": a_team,
                "commence_time": str(r.get('kickoff_time', '')),
                "home_win_odds": round(1.0 / max(p_h, 0.01), 2),
                "draw_odds": round(1.0 / max(p_d, 0.01), 2),
                "away_win_odds": round(1.0 / max(p_a, 0.01), 2),
                "over_25_odds": 1.85,
                "under_25_odds": 1.95,
                "home_win_prob": round(float(p_h), 3),
                "draw_prob": round(float(p_d), 3),
                "away_win_prob": round(float(p_a), 3),
                "home_cs_prob": round(float(cs_h), 3),
                "away_cs_prob": round(float(cs_a), 3),
                "bookmaker": "ClubElo Quantitative Engine (Zero-Quota Baseline)"
            })
    return pd.DataFrame(rows)

def fetch_live_sharp_odds(cache_ttl_hours=ODDS_CACHE_TTL_HOURS, force_refresh=False):
    """
    Fetches real-time Premier League odds from The-Odds-API (UK sharp bookmakers).
    Guards betting API quota strictly:
    1. Zero automatic calls until within 48h (2 days) of the upcoming Gameweek deadline.
    2. 5-day (120-hour) local disk cache.
    3. 24-hour strict anti-exhaustion throttle between manual refreshes.
    4. Scoped strictly to English Premier League ('soccer_epl').
    5. Fallback to ClubElo quantitative model outside the 48h window.
    """
    initialize_database()
    conn = get_db_connection()
    
    # 1. Check local file/DB cache freshness
    if not force_refresh and os.path.exists(ODDS_CACHE_PATH):
        try:
            mtime = os.path.getmtime(ODDS_CACHE_PATH)
            age_hours = (time.time() - mtime) / 3600.0
            if age_hours < cache_ttl_hours:
                df_cached = pd.read_csv(ODDS_CACHE_PATH)
                if not df_cached.empty:
                    return df_cached
        except Exception:
            pass

    # 2. Check 24-hour strict throttle
    try:
        cur = conn.cursor()
        cur.execute("SELECT timestamp FROM meta_cache WHERE key = 'last_odds_api_sync'")
        row = cur.fetchone()
        if row and row[0]:
            last_dt = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            time_since_call_hours = (datetime.datetime.now() - last_dt).total_seconds() / 3600.0
            if time_since_call_hours < ODDS_MIN_REFRESH_INTERVAL_HOURS and not force_refresh:
                if os.path.exists(ODDS_CACHE_PATH):
                    return pd.read_csv(ODDS_CACHE_PATH)
    except Exception:
        pass

    # 3. Check 48-hour (2 days before GW) deadline window if not manually forced
    is_near_deadline = False
    if not force_refresh:
        try:
            cur = conn.cursor()
            cur.execute("SELECT MIN(kickoff_time) FROM fixtures WHERE finished = 0 AND kickoff_time IS NOT NULL")
            row_ko = cur.fetchone()
            if row_ko and row_ko[0]:
                ko_str = str(row_ko[0])
                if "T" in ko_str:
                    ko_dt = datetime.datetime.fromisoformat(ko_str.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    ko_dt = datetime.datetime.strptime(ko_str[:19], "%Y-%m-%d %H:%M:%S")
                diff_hours = (ko_dt - datetime.datetime.utcnow()).total_seconds() / 3600.0
                if 0 <= diff_hours <= 48.0:
                    is_near_deadline = True
        except Exception:
            pass

    # If outside the 48h deadline window and not manually forced, use cache or ClubElo fallback
    if not force_refresh and not is_near_deadline:
        if os.path.exists(ODDS_CACHE_PATH):
            try:
                return pd.read_csv(ODDS_CACHE_PATH)
            except Exception:
                pass
        return generate_fallback_odds_from_elo()

    # 3. Fetch from The-Odds-API if API key is present
    if ODDS_API_KEY:
        try:
            url = ODDS_API_URL
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": "uk",
                "markets": "h2h,totals",
                "oddsFormat": "decimal"
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                remaining = resp.headers.get("x-requests-remaining", "500")
                used = resp.headers.get("x-requests-used", "0")
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                cur = conn.cursor()
                cur.execute("REPLACE INTO meta_cache (key, value, timestamp) VALUES ('last_odds_api_sync', ?, ?)", (now_str, now_str))
                cur.execute("REPLACE INTO meta_cache (key, value, timestamp) VALUES ('odds_quota_remaining', ?, ?)", (str(remaining), now_str))
                cur.execute("REPLACE INTO meta_cache (key, value, timestamp) VALUES ('odds_quota_used', ?, ?)", (str(used), now_str))
                conn.commit()

                data = resp.json()
                rows = []
                for match in data:
                    h_raw = match.get("home_team", "")
                    a_raw = match.get("away_team", "")
                    h_team = ODDS_TO_FPL_TEAM_MAP.get(h_raw, h_raw)
                    a_team = ODDS_TO_FPL_TEAM_MAP.get(a_raw, a_raw)
                    commence_time = match.get("commence_time", "")
                    
                    bms = match.get("bookmakers", [])
                    if not bms:
                        continue
                    
                    pref = ["Betfair Sportsbook", "Bet365", "Sky Bet", "Unibet (UK)", "Betfred (UK)", "William Hill"]
                    bm = bms[0]
                    for p in pref:
                        for b in bms:
                            if b.get("title") == p:
                                bm = b
                                break
                        if bm.get("title") == p:
                            break
                    bm_title = bm.get("title", "Market Consensus")
                    
                    h_odds, d_odds, a_odds = 2.0, 3.2, 3.5
                    for m in bm.get("markets", []):
                        if m.get("key") == "h2h":
                            for o in m.get("outcomes", []):
                                if o.get("name") == h_raw: h_odds = float(o.get("price", 2.0))
                                elif o.get("name") == a_raw: a_odds = float(o.get("price", 3.5))
                                elif o.get("name") == "Draw": d_odds = float(o.get("price", 3.2))
                    
                    over_odds, under_odds = 1.85, 1.95
                    for m in bm.get("markets", []):
                        if m.get("key") == "totals":
                            for o in m.get("outcomes", []):
                                if o.get("name") == "Over" and float(o.get("point", 2.5)) == 2.5: over_odds = float(o.get("price", 1.85))
                                elif o.get("name") == "Under" and float(o.get("point", 2.5)) == 2.5: under_odds = float(o.get("price", 1.95))
                    
                    probs_h2h = SharpOddsEngine.devig_shins_method(np.array([h_odds, d_odds, a_odds]))
                    probs_tot = SharpOddsEngine.devig_shins_method(np.array([over_odds, under_odds]))
                    
                    p_h, p_d, p_a = probs_h2h[0], probs_h2h[1], probs_h2h[2]
                    p_over, p_under = probs_tot[0], probs_tot[1]
                    
                    cs_h = np.clip(p_h * 0.52 + p_under * 0.28, 0.10, 0.65)
                    cs_a = np.clip(p_a * 0.48 + p_under * 0.28 - 0.05, 0.05, 0.55)
                    
                    rows.append({
                        "home_team": h_team,
                        "away_team": a_team,
                        "commence_time": commence_time,
                        "home_win_odds": round(h_odds, 2),
                        "draw_odds": round(d_odds, 2),
                        "away_win_odds": round(a_odds, 2),
                        "over_25_odds": round(over_odds, 2),
                        "under_25_odds": round(under_odds, 2),
                        "home_win_prob": round(float(p_h), 3),
                        "draw_prob": round(float(p_d), 3),
                        "away_win_prob": round(float(p_a), 3),
                        "home_cs_prob": round(float(cs_h), 3),
                        "away_cs_prob": round(float(cs_a), 3),
                        "bookmaker": bm_title
                    })

                if rows:
                    df_res = pd.DataFrame(rows)
                    df_res.to_csv(ODDS_CACHE_PATH, index=False)
                    
                    cur.execute("DELETE FROM sharp_odds_cache")
                    for _, r in df_res.iterrows():
                        cur.execute("""
                            INSERT INTO sharp_odds_cache (
                                home_team, away_team, commence_time, home_win_odds, draw_odds, away_win_odds,
                                over_25_odds, under_25_odds, home_win_prob, draw_prob, away_win_prob,
                                home_cs_prob, away_cs_prob, bookmaker, last_updated
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            r['home_team'], r['away_team'], r['commence_time'], r['home_win_odds'], r['draw_odds'],
                            r['away_win_odds'], r['over_25_odds'], r['under_25_odds'], r['home_win_prob'],
                            r['draw_prob'], r['away_win_prob'], r['home_cs_prob'], r['away_cs_prob'], r['bookmaker'], now_str
                        ))
                    conn.commit()
                    return df_res
        except Exception as e:
            print(f"⚠️ Live Odds API error: {e}. Falling back to cache / ClubElo.")
        finally:
            conn.close()

    if os.path.exists(ODDS_CACHE_PATH):
        try:
            return pd.read_csv(ODDS_CACHE_PATH)
        except Exception:
            pass

    return generate_fallback_odds_from_elo()

def get_odds_quota_info():
    """Returns metadata about Odds API quota, last sync, and cache freshness."""
    initialize_database()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT key, value, timestamp FROM meta_cache WHERE key IN ('last_odds_api_sync', 'odds_quota_remaining', 'odds_quota_used')")
        data = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        
        last_sync = data.get('last_odds_api_sync', ('Never', None))[0]
        remaining = data.get('odds_quota_remaining', ('490', None))[0]
        used = data.get('odds_quota_used', ('10', None))[0]
        
        age_hours = 999.0
        if last_sync != 'Never':
            try:
                dt = datetime.datetime.strptime(last_sync, "%Y-%m-%d %H:%M:%S")
                age_hours = round((datetime.datetime.now() - dt).total_seconds() / 3600.0, 1)
            except Exception:
                pass
                
        return {
            "last_sync": last_sync,
            "remaining": remaining,
            "used": used,
            "age_hours": age_hours,
            "is_fresh": age_hours < ODDS_CACHE_TTL_HOURS
        }
    finally:
        conn.close()


def get_full_fpl_schedule():
    """Loads all 380 Premier League fixtures mapped to team names, ClubElo per game, and FDR difficulty."""
    initialize_database()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM fixtures")
        count = cur.fetchone()[0]
        if count == 0:
            sync_fpl_api_data()
            
        fixtures_df = pd.read_sql("SELECT * FROM fixtures", conn)
        teams_df = pd.read_sql("SELECT * FROM teams", conn)
        elo_dict = fetch_clubelo_ratings()
        
        if not fixtures_df.empty and not teams_df.empty:
            team_map = dict(zip(teams_df['id'], teams_df['name']))
            short_map = dict(zip(teams_df['id'], teams_df['short_name']))
            
            fixtures_df['home_team'] = fixtures_df['team_h'].map(team_map)
            fixtures_df['away_team'] = fixtures_df['team_a'].map(team_map)
            fixtures_df['home_short'] = fixtures_df['team_h'].map(short_map)
            fixtures_df['away_short'] = fixtures_df['team_a'].map(short_map)
            fixtures_df['matchup'] = fixtures_df['home_team'] + " vs " + fixtures_df['away_team']
            fixtures_df['status'] = np.where(fixtures_df['finished'] == 1, 'Finished', 'Upcoming')
            
            # Map ClubElo Ratings per game
            def get_elo(t_name):
                return float(elo_dict.get(t_name, 1750.0))
                
            fixtures_df['home_elo'] = fixtures_df['home_team'].apply(get_elo)
            fixtures_df['away_elo'] = fixtures_df['away_team'].apply(get_elo)
            
            # Home Advantage = +60 Elo points
            fixtures_df['net_elo_delta'] = (fixtures_df['home_elo'] + 60.0) - fixtures_df['away_elo']
            
            # Win & Clean Sheet Estimates per match
            fixtures_df['home_cs_est'] = (np.clip(0.25 + (fixtures_df['home_elo'] - fixtures_df['away_elo']) / 2000.0, 0.10, 0.65) * 100).astype(int).astype(str) + "%"
            fixtures_df['away_cs_est'] = (np.clip(0.20 + (fixtures_df['away_elo'] - fixtures_df['home_elo'] - 60.0) / 2000.0, 0.05, 0.55) * 100).astype(int).astype(str) + "%"
            
            return fixtures_df
    except Exception as e:
        print(f"⚠️ Schedule load error: {e}")
    finally:
        conn.close()
    return pd.DataFrame()

def get_clubelo_visualization_df():
    """Returns Elo rating dataframe for all Premier League teams for visualization."""
    elo_dict = fetch_clubelo_ratings()
    data = []
    for team, elo in elo_dict.items():
        data.append({
            "Team": team,
            "ClubElo_Rating": float(elo),
            "Goal_Expectation_Scaling": round(1.0 + (float(elo) - 1750.0) / 1000.0, 2),
            "Baseline_Clean_Sheet_Prob": f"{int(min(max(0.20 + (float(elo)-1750.0)/2000.0, 0.10), 0.65)*100)}%"
        })
    df = pd.DataFrame(data).sort_values("ClubElo_Rating", ascending=False)
    return df

def get_player_availability_df():
    """Loads all player availability, injury news, and chance of playing from SQLite DB."""
    initialize_database()
    conn = get_db_connection()
    try:
        meta_df = pd.read_sql("SELECT * FROM players_meta", conn)
        teams_df = pd.read_sql("SELECT * FROM teams", conn)
        if not meta_df.empty and not teams_df.empty:
            team_map = dict(zip(teams_df['id'], teams_df['name']))
            meta_df['team'] = meta_df['team_id'].map(team_map)
            meta_df['cost'] = meta_df['now_cost'] / 10.0
            
            def map_status(row):
                st = row['status']
                ch = row['chance_of_playing']
                if st == 'a' and ch == 100:
                    return '🟢 Available (100%)'
                elif st == 'd' or (ch > 0 and ch < 100):
                    return f'🟡 Doubtful ({ch}%)'
                elif st == 'i':
                    return '🔴 Injured (0%)'
                elif st == 's':
                    return '🔴 Suspended (0%)'
                else:
                    return f'🔴 Unavailable ({ch}%)'
                    
            meta_df['availability_status'] = meta_df.apply(map_status, axis=1)
            meta_df['news_note'] = np.where(meta_df['news'].isna() | (meta_df['news'] == ''), 'Fully Fit / Available', meta_df['news'])
            
            cols = ['id', 'web_name', 'position', 'team', 'cost', 'availability_status', 'chance_of_playing', 'news_note']
            return meta_df[cols].sort_values(['chance_of_playing', 'cost'], ascending=[True, False]).rename(columns={'id': 'player_id', 'web_name': 'name'})
    except Exception as e:
        print(f"⚠️ Availability load error: {e}")
    finally:
        conn.close()
    return pd.DataFrame()


def get_price_change_radar_df():
    """
    Returns player price rise/fall velocity radar data based on net transfer flow.
    Calculates target progress towards +/- 100% net transfer threshold for midnight price changes.
    """
    conn = get_db_connection()
    try:
        query = """
            SELECT p.id, p.web_name, p.position, t.name as team, p.now_cost,
                   p.selected_by_percent, p.transfers_in_event, p.transfers_out_event
            FROM players_meta p
            LEFT JOIN teams t ON p.team_id = t.id
        """
        df = pd.read_sql(query, conn)
        if not df.empty:
            df["cost"] = df["now_cost"] / 10.0
            df["net_transfers"] = df["transfers_in_event"] - df["transfers_out_event"]
            threshold = 40000.0
            df["target_progress_pct"] = np.clip((df["net_transfers"] / threshold) * 100.0, -100.0, 100.0)
            
            def get_price_status(row):
                prog = row["target_progress_pct"]
                if prog >= 90.0:
                    return "🔥 IMMINENT RISE (+£0.1m)"
                elif prog >= 40.0:
                    return "⚡ BUYING MOMENTUM"
                elif prog <= -90.0:
                    return "❄️ IMMINENT FALL (-£0.1m)"
                elif prog <= -40.0:
                    return "⚠️ SELLING PRESSURE"
                else:
                    return "STABLE"
                    
            df["status_badge"] = df.apply(get_price_status, axis=1)
            return df.rename(columns={"id": "player_id", "web_name": "name"})
    except Exception as e:
        print(f"⚠️ Price radar error: {e}")
    finally:
        conn.close()
    return pd.DataFrame()

