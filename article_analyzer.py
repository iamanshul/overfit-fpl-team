# -*- coding: utf-8 -*-
"""
Article Sentiment & Qualitative Counter-Argument Engine (article_analyzer.py)
Analyzes user-pasted publication text, evaluates player claims against underlying data (xG/xA/xM/Elo),
explains quantitative reasoning, and applies custom rate overrides before MILP solver.
"""

import os
import json
import re
import pandas as pd
import numpy as np

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class ArticleSentimentEngine:
    """
    Evaluates qualitative article claims against underlying quantitative data.
    Provides statistical counter-arguments and allows user rate overrides.
    Powered by Gemini 2.5 Flash LLM with fallback to regex rule-based engine.
    """

    def __init__(self, matrix_df, api_key=None):
        self.matrix = matrix_df.copy() if not matrix_df.empty else pd.DataFrame()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if (GENAI_AVAILABLE and self.api_key) else None

    def analyze_article_llm(self, article_text):
        """
        Uses Gemini 2.5 Flash to extract mentioned players, sentiment, injury risks,
        and suggested rate multipliers. Falls back to regex analyze_article if unconfigured.
        """
        if not self.client or self.matrix.empty or not article_text.strip():
            return self.analyze_article(article_text)

        try:
            player_list = self.matrix[["player_id", "name", "team", "position", "cost", "r_goal", "r_assist", "xM", "xP_horizon_sum"]].head(100).to_dict(orient="records")
            
            prompt = f"""
            You are an expert FPL quantitative analyst.
            Analyze the following publication article/quotes and identify any mentioned FPL players.
            
            Article Text:
            {article_text}

            Active FPL Players in Database:
            {json.dumps(player_list[:60])}

            Return a JSON array of objects for players mentioned in the article with this exact schema:
            [
              {{
                "player_id": int,
                "name": "Player Name",
                "verdict": "✅ SUPPORTED (Reason)" or "⚠️ CAUTION (Reason)" or "⚠️ SKEPTICAL (Reason)",
                "quantitative_reasoning": "Clear 1-2 sentence evaluation combining article claims with underlying stats",
                "suggested_multiplier": float (between 0.70 and 1.30)
              }}
            ]
            Only return valid raw JSON array.
            """

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )

            results_data = json.loads(response.text)
            res_df = pd.DataFrame(results_data)
            if not res_df.empty and "player_id" in res_df.columns:
                merged = res_df.merge(
                    self.matrix[["player_id", "position", "team", "cost"]],
                    on="player_id",
                    how="left",
                    suffixes=("", "_mat")
                )
                if "position_mat" in merged.columns:
                    merged["position"] = merged["position"].fillna(merged["position_mat"])
                if "cost_mat" in merged.columns:
                    merged["cost"] = merged["cost"].fillna(merged["cost_mat"])
                return merged
        except Exception as e:
            print(f"⚠️ Gemini API fallback to rule engine: {e}")

        return self.analyze_article(article_text)

    def analyze_article(self, article_text):
        """
        Parses article text, identifies mentioned players, and evaluates claims.
        """
        if self.matrix.empty or not article_text.strip():
            return pd.DataFrame()

        results = []
        players_in_matrix = self.matrix["name"].tolist()
        matched_ids = set()

        for name in players_in_matrix:
            row = self.matrix[self.matrix["name"] == name].iloc[0]
            pid = int(row["player_id"])
            if pid in matched_ids:
                continue

            # Build search tokens: web_name, stripped last name (e.g. B.Fernandes -> Fernandes), and aliases
            search_tokens = [name]
            if "." in name:
                search_tokens.append(name.split(".")[-1])
            
            # Common FPL Aliases
            alias_map = {
                "B.Fernandes": ["Bruno Fernandes", "Bruno"],
                "Bruno G.": ["Bruno Guimaraes", "Guimaraes"],
                "Alexander-Arnold": ["TAA", "Trent"],
                "De Bruyne": ["KDB", "Kevin De Bruyne"],
                "Van Dijk": ["VVD", "Virgil"]
            }
            if name in alias_map:
                search_tokens.extend(alias_map[name])

            matched = False
            matched_pos = -1
            for token in search_tokens:
                if len(token.strip()) >= 3:
                    m = re.search(r'\b' + re.escape(token.strip()) + r'\b', article_text, re.IGNORECASE)
                    if m:
                        matched = True
                        matched_pos = m.start()
                        break

            if matched:
                matched_ids.add(pid)
                
                pos = row.get("position", "MID")
                team = row.get("team", "UNK")
                cost = float(row.get("cost", 5.0))
                r_goal = float(row.get("r_goal", 0.0))
                r_assist = float(row.get("r_assist", 0.0))
                xM = float(row.get("xM", 75.0))
                xp_sum = float(row.get("xP_horizon_sum", 20.0))
                xp_per_pound = xp_sum / max(cost, 1.0)

                # Extract context window around player mention (120 chars before and after)
                ctx_start = max(0, matched_pos - 120)
                ctx_end = min(len(article_text), matched_pos + 120)
                context_str = article_text[ctx_start:ctx_end].lower()

                # 1. Check for Severe Injury / Ruled Out / Surgery / Suspension
                injury_out_match = re.search(r'out for|ruled out|surgery|sidelined|torn|fracture|broken|suspended|banned|not in squad|miss out|unfit|hamstring tear|knee injury|acl|ankle break', context_str)
                # 2. Check for Minor Knock / Fitness Doubt / Late Assessment / Illness
                injury_doubt_match = re.search(r'doubt|knock|strain|tightness|illness|late fitness test|question mark|touch and go|assessed|minor knock|hamstring|groin', context_str)
                # 3. Check for Rotation / Rest / Benched / Minutes Management
                rotation_match = re.search(r'bench|rest|rotate|fatigue|minutes managed|substitute|impact sub|backup|squad rotation', context_str)
                # 4. Check for High Form / Essential / Praise / In-form / Penalty duty
                form_match = re.search(r'in form|starting|key player|essential|praise|penalty|hat-trick|sharp|masterclass|undroppable|guaranteed starter', context_str)

                if injury_out_match:
                    verdict = "🔴 RULED OUT / SEVERE INJURY"
                    reasoning = f"Article reports {name} is ruled out/unavailable ({injury_out_match.group(0)}). Zero starting probability."
                    suggested_mult = 0.00
                elif injury_doubt_match:
                    verdict = "🟡 DOUBTFUL / KNOCK (Monitor Presser)"
                    reasoning = f"Article mentions fitness doubt for {name} ({injury_doubt_match.group(0)}). Projected minutes discounted by 50%."
                    suggested_mult = 0.50
                elif rotation_match:
                    verdict = "⚠️ ROTATION RISK (Managed Minutes)"
                    reasoning = f"Article suggests rotation/minutes management for {name} ({rotation_match.group(0)}). Sub risk reduces EV."
                    suggested_mult = 0.75
                elif form_match:
                    verdict = "✅ SUPPORTED (High Confidence / Form)"
                    reasoning = f"Article highlights positive tactical role/form for {name} ({form_match.group(0)}) with high manager backing."
                    suggested_mult = 1.20
                elif r_goal > 0.40 or (pos == "FWD" and xp_sum > 25.0):
                    verdict = "✅ SUPPORTED (Elite Underlying Threat)"
                    reasoning = f"{name} ({pos}, {team} - £{cost:.1f}m) has high npxG90 ({r_goal:.2f}) and top 6-GW xP ({xp_sum:.1f} pts)."
                    suggested_mult = 1.15
                elif xp_per_pound < 3.5 and cost > 9.0:
                    verdict = "⚠️ SKEPTICAL (Premium Price Dilution)"
                    reasoning = f"{name} ({pos}, {team} - £{cost:.1f}m) requires heavy budget capital. Generates only {xp_per_pound:.2f} xP per £1.0m cost."
                    suggested_mult = 0.95
                elif xM < 60.0:
                    verdict = "⚠️ CAUTION (Low Historical Minutes)"
                    reasoning = f"{name} ({pos}, {team} - £{cost:.1f}m) averages low expected minutes (xM: {xM:.0f} mins/game)."
                    suggested_mult = 0.90
                else:
                    verdict = "ℹ️ NEUTRAL / FAIR VALUE"
                    reasoning = f"{name} ({pos}, {team} - £{cost:.1f}m) is fairly priced with steady baseline (xM: {xM:.0f} mins, rGoal: {r_goal:.2f})."
                    suggested_mult = 1.00

                results.append({
                    "player_id": int(row["player_id"]),
                    "name": name,
                    "position": pos,
                    "team": team,
                    "cost": cost,
                    "verdict": verdict,
                    "quantitative_reasoning": reasoning,
                    "suggested_multiplier": suggested_mult,
                    "applied_override": 1.00
                })

        return pd.DataFrame(results)

    def apply_user_overrides(self, override_dict):
        """
        Applies user-defined multiplier overrides to the xP horizon matrix.
        override_dict: {player_id: multiplier_float}
        Returns updated matrix DataFrame.
        """
        if self.matrix.empty or not override_dict:
            return self.matrix

        updated = self.matrix.copy()
        xp_cols = [c for c in updated.columns if c.startswith("xP_")]

        for pid, mult in override_dict.items():
            if pid in updated["player_id"].values:
                for col in xp_cols:
                    updated.loc[updated["player_id"] == pid, col] *= mult

        return updated
