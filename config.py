# -*- coding: utf-8 -*-
"""
Central Quantitative Configuration for Jetski FPL Engine
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Safely load local .env without third-party dependencies (gitignored)
ENV_FILE = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_FILE):
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Database and Data File Paths
DB_PATH = os.path.join(DATA_DIR, "fpl_system.db")
CSV_PATH = os.path.join(DATA_DIR, "fpl_all_player_data.csv")
ELO_CACHE_PATH = os.path.join(DATA_DIR, "elo_cache.csv")
ODDS_CACHE_PATH = os.path.join(DATA_DIR, "sharp_odds_cache.csv")

# The-Odds-API Integration Configuration (Never hardcoded in Git; injected via env / Secret Manager)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
ODDS_CACHE_TTL_HOURS = 48.0              # 48-hour local cache (only ~15 API calls/month)
ODDS_MIN_REFRESH_INTERVAL_HOURS = 12.0   # Strict anti-exhaustion throttle: minimum 12h between calls

# Model Predictive Control (MPC) Horizon Parameters
ROLLING_HORIZON_WEEKS = 6    # Optimize over 6 rolling gameweeks
HORIZON_DECAY_FACTOR = 0.88   # Gamma: Time-discount factor for future variance

# Financial & Transfer State Dynamics
MAX_FREE_TRANSFERS = 5       # Max stackable FT limit (FPL rule update)
FREE_TRANSFER_OPTION_VALUE = 1.8  # Option value of holding 1 FT
TRANSFER_HIT_COST = 4.0       # Point cost per hit (-4 pts)
BANK_SALVAGE_WEIGHT = 0.0     # Unspent bank cash gives 0 pts at start of season; forces 100% squad budget spend

# Squad & Formation Constraints
SQUAD_SIZE = 15
SQUAD_QUOTAS = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3
}

XI_QUOTAS_MIN = {
    "GKP": 1,
    "DEF": 3,
    "MID": 2,
    "FWD": 1
}

XI_QUOTAS_MAX = {
    "GKP": 1,
    "DEF": 5,
    "MID": 5,
    "FWD": 3
}

MAX_PLAYERS_PER_TEAM = 3

# Chip Reservation Hurdle Curves (Rho thresholds)
# Net xP delta required to auto-trigger chip activation
CHIP_RESERVATION_CURVES = {
    "wildcard": 15.0,      # Need +15.0 xP over 6-GW horizon
    "freehit": 18.0,       # Need +18.0 xP in target gameweek
    "benchboost": 12.0,    # Need +12.0 xP across bench players
    "triplecaptain": 10.0  # Need +10.0 xP extra on captain
}

# Network Request Headers (Browser user-agent to bypass Cloudflare/Varnish blocks)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
