# ⚽ Jetski FPL: When Frontier AI Meets Fantasy Football 🤖🌊

> *"Can cutting-edge AI and quantitative trading math actually win the worldwide Fantasy Premier League championship?"*

Let’s be honest: Fantasy Premier League (FPL) is chaotic, emotional, and intensely competitive. Every weekend, 11 million managers worldwide make decisions based on gut feel, Twitter hype, or loyalty to their favorite clubs—only for a 94th-minute center-back yellow card to ruin their weekend.

As someone working on the **leading edge of AI and autonomous agentic systems**, I wanted to run a real-life experiment: **What happens when you treat FPL as a high-stakes stochastic portfolio optimization problem?**

This repository is that experiment in the wild. No black-box voodoo, no emotional panic-selling on a Sunday night—just pure mathematical optimization, discrete Poisson physics, and a team of autonomous AI agents debating transfer strategy over a 6-week rolling horizon.

---

## 🧠 The Philosophy: Why Most FPL Models Fail (And What We Do Differently)

Most automated fantasy bots fall into the classic **"GW1 Hyper-Optimization Trap"**: they blow £85m on 11 players for Gameweek 1, fill the bench with dead £4.0m ghost assets, and by Gameweek 2 they're forced into taking -8 point transfer hits just to field a legal eleven.

We built a **Two-Tier Decision Architecture**:

1. **Tier 1: 15-Man Multi-Period Portfolio Optimization ($H = 6$ Weeks)**
   * We don't just pick 11 starters; we build a resilient 15-man portfolio.
   * **Fixture Rotation Synergies**: We pair budget £4.5m defenders who alternate green home fixtures.
   * **Nailed Bench Security**: Every bench player has real 90-minute playing security ($xM \ge 60\text{ mins}$) so surprise benchings never leave you with a 0-pointer.
   * **The 5-Free-Transfer Runway**: In the new rules, you can roll up to **5 Free Transfers**. Our team is built to roll transfers in GW2–4, building a massive tactical war chest for mini-wildcards without hits.

2. **Tier 2: Tactical Single-Gameweek Starting XI & Captaincy**
   * Given the 15 players in our portfolio, we solve the weekly knapsack problem for the highest-ceiling Starting 11 and Captain.

---

## 🔬 Under the Hood: The Quantitative Engine

* 🧮 **Mixed-Integer Linear Programming (MILP)**: Exact branch-and-cut optimization (via PuLP/CBC) solving formation quotas, 3-player club limits, 50% profit sell-on tax ($P_{\text{sell}} = P_{\text{buy}} + \lfloor (P_{\text{curr}} - P_{\text{buy}})/2 \rfloor$), and legal auto-sub invariants.
* 🧤 **Discrete Poisson Point Physics**:
  * Clean Sheet probabilities via bivariate Dixon-Coles goal rates.
  * Exact conceded goal deductions ($-1\text{ pt}$ per 2 goals conceded for DEFs/GKPs).
  * Goalkeeper save points ($+1\text{ pt}$ per 3 saves based on defensive shot volume).
* 👑 **Captaincy Ceiling Upside Multiplier**: Armbands shouldn't just chase safe 5-point averages; we optimize for explosive 12+ point haul volatility on elite talismans.
* 🤖 **Multi-Round Adversarial Critic Loop**: An AI critic that audits our draft across 6 stress vectors (*Budget Leakage, Haaland/Bruno Effective Ownership shields, Bench fragility*) and asks: *"Why Player X over Market Rival Y?"* before locking in the final squad.
* 📰 **Anti-Blind-Trust Article Fact-Checker**: You can paste any news article from *The Athletic* or press conference quotes into the UI. The engine extracts the claims, fact-checks them against underlying $npxG_{90}/xA_{90}$ data, and lets you slide custom multipliers before solving.

---

## 🕹️ Interactive Web Dashboard (Streamlit)

Everything runs through a slick interactive Streamlit studio:
* **Tab 1: Team 11 & Starting Lineup**: Pitch visualizer with upcoming 3-match fixture difficulty badges (Green/Yellow/Red).
* **Tab 2: Start-of-Season GW1 Studio**: Compare formations (3-4-3 vs 3-5-2 vs 4-4-2) and run 3-round Adversarial Convergence loops.
* **Tab 3: Squad & Transfer Hub**: Live selling price calculations, bank ledgers, and transfer execution.
* **Tab 4: Qualitative Article Sentiment Engine**: Paste expert articles and adjust player multipliers on the fly.
* **Tab 7 & 8: 6-GW Transfer Roadmap & Chip Evaluator**: Multi-period horizon simulator evaluating Wildcard and Free Hit equity against dynamic hurdle curves.

---

## ⚡ Quickstart (Run It Locally)

```bash
# 1. Clone the repo
git clone https://github.com/[YOUR_USERNAME]/jetski-fpl-team.git
cd jetski-fpl-team

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Launch the dashboard!
streamlit run app.py
```

### 🧪 Running the Tests
```bash
python -m unittest discover tests
```
*(All 24 unit tests passing 100% green).*

---

## ☁️ Cloud Deployment

Ready to run from your phone on the train? Deploy directly to **Google Cloud Run**:

```bash
# Build & Deploy to Cloud Run
gcloud builds submit --tag gcr.io/[YOUR_PROJECT_ID]/jetski-fpl-team
gcloud run deploy jetski-fpl-team \
    --image gcr.io/[YOUR_PROJECT_ID]/jetski-fpl-team \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi
```

---

## 📜 License & Disclaimers

Licensed under the [MIT License](LICENSE). 

*Disclaimer: This is an active research experiment applying modern AI, stochastic calculus, and integer optimization to Fantasy Premier League. No algorithm can predict a 90th-minute hamstring pull or a Pep roulette benching—but math gives us the ultimate edge.* 🚀
