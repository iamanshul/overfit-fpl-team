# ⚽ Jetski FPL Quantitative Decision Engine

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

A state-of-the-art Fantasy Premier League (FPL) quantitative decision engine combining **Mixed-Integer Linear Programming (MILP)** Model Predictive Control, multi-period transfer optimization with 5-FT stacking, Dixon-Coles bivariate Poisson modeling, LightGBM component rate estimation, and an interactive **Adversarial Critic Studio** web dashboard.

---

## 🚀 Key Features

* **Multi-Period Portfolio MILP Optimizer**: Solves rolling 6-gameweek horizons incorporating 5-FT stacking, 50% selling price profit tax accounting, and formation-legal auto-sub ordering.
* **Discrete Poisson Point Physics**: Explicitly models clean sheet expectations, goals conceded deductions (-1 pt / 2 goals conceded for DEF/GKP), and goalkeeper save points (+1 pt / 3 saves).
* **Two-Tier Architecture**: Decouples 15-man resilient multi-week portfolio construction (rotation pairings & playing bench) from single-gameweek starting XI tactical selection.
* **Adversarial Critic Studio**: 3-round iterative convergence loop ($R_1 \rightarrow R_2 \rightarrow R_3$) stress-testing drafts against budget leakage, Effective Ownership (EO) shields, and bench fragility with transparent "Why X over Y?" trade-off comparisons.
* **Interactive Article Sentiment & Fact-Checker**: Ingests qualitative journalist and press conference claims, tests them against underlying $npxG_{90}$/$xA_{90}$ rates, and applies dynamic horizon availability multipliers ($A_{p, w}$).

---

## 🛠️ Local Installation & Quickstart

```bash
# 1. Clone repository
git clone https://github.com/[YOUR_USERNAME]/jetski-fpl-team.git
cd jetski-fpl-team

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install production dependencies
pip install -r requirements.txt

# 4. Launch Streamlit Web Application
streamlit run app.py
```

---

## 🧪 Automated Test Suite

Run the full unit test suite:
```bash
python -m unittest discover tests
```

---

## ☁️ Cloud Deployment (Google Cloud Run)

Build and deploy with Google Cloud CLI:
```bash
# Build container image
gcloud builds submit --tag gcr.io/[PROJECT_ID]/jetski-fpl-team

# Deploy to Cloud Run
gcloud run deploy jetski-fpl-team \
    --image gcr.io/[PROJECT_ID]/jetski-fpl-team \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
