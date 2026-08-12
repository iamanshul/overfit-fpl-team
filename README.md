# Overfit FPL ⚽

I spend my day job working on frontier AI, but like millions of others, my weekends revolve around Fantasy Premier League.

Most FPL models make the same mistake: they hyper-optimize for the next gameweek, buy a bunch of one-week punts, and leave you with four non-playing bench players and no flexibility for GW2.

This project is a fun, real-world experiment to see how modern optimization and multi-agent reasoning hold up against the actual chaos of the Premier League.

---

## How It Works

Instead of just picking 11 players for this weekend, the engine thinks in two steps:

1. **The 15-Man Squad (6-Week Horizon)**: Builds a balanced portfolio over 6 gameweeks. It sets up £4.5m defender rotation pairings (playing the one with the easier home fixture), keeps a playing bench for auto-subs, and leaves room to stack up to **5 Free Transfers** so you can make big fixture moves without taking hits.
2. **The Weekly 11**: Picks the best starting lineup and captaincy from those 15 for that specific gameweek.

Under the hood, it handles the real rules of the game:
- **Exact Accounting**: 50% profit tax on player sales, cashflow balances, and bank management.
- **Scoring Physics**: Clean sheet odds, points deductions for goals conceded, and goalkeeper save rates.
- **Adversarial Critic**: Double-checks the squad across budget efficiency, Effective Ownership (EO) risk, and bench depth before locking it in.
- **Article Fact-Checker**: You can paste in news or press conference quotes; it checks claims against underlying stats ($npxG$, $xA$, expected minutes) and lets you adjust player weights directly in the UI.

---

## Getting Started

### Local Setup
```bash
git clone https://github.com/iamanshul/overfit-fpl-team.git
cd overfit-fpl-team


python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

### Tests
```bash
python -m unittest discover tests
```

---

## License

[MIT](LICENSE)
