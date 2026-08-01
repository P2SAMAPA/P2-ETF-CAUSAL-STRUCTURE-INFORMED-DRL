# P2-CSI-RANKER

**Causal-Structure-Informed Ranker — Causal Discovery + Momentum Ranking**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine combines **causal discovery** with **momentum ranking** to produce actionable trading signals. Unlike the original DRL-based concept, this implementation uses a practical, proven approach:

1. **PCMCI+ causal discovery** identifies causal relationships between variables
2. **Causal graph metrics** (incoming/outgoing links) are computed for each ETF
3. **Momentum factors** (short, medium, long-term) capture trend information
4. **Composite scoring** combines causality and momentum into one actionable number
5. **Cross-sectional z-scores** rank ETFs from best to worst

**Key Insight:** ETFs that are causal drivers (high incoming causality) combined with positive momentum are likely to continue outperforming.

---

## How It Works

| Step | What happens |
|------|-------------|
| 1. **PCMCI+** | Discover causal relationships in time series data |
| 2. **Causal Graph** | Build adjacency matrix of causal links between variables |
| 3. **Causality Metrics** | Compute incoming/outgoing causality for each ETF |
| 4. **Momentum Factors** | Calculate short (10d), medium (30d), long (60d) momentum |
| 5. **Composite Score** | Weighted combination: 40% ST momentum + 20% MT momentum + 10% LT momentum - 15% volatility + 15% net causality |
| 6. **Ranking** | Cross-sectional z-scores across all ETFs |
| 7. **Action** | Top 20% = BUY, Bottom 20% = SELL |

---

## Key Metrics

| Metric | What it tells you | Trading Implication |
|--------|-------------------|---------------------|
| **z-score** | Cross-sectional ranking | > 0.5 = BUY, < -0.5 = SELL |
| **Causal Links** | Number of detected causal relationships | More = stronger causal influence |
| **Net Causality** | Incoming - Outgoing | Positive = causal receiver (leader) |
| **Momentum** | Short-term price trend | Positive = upward momentum |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term causal structure |
| 252d | Core signal (primary) |
| 504d | Medium-term causal relationships |
| 1008d | Structural causal links |
| 2016d+ | Secular causal structure |

**Primary Window:** 252d (~1 year)

---

## Universes

| Universe | Tickers |
|----------|---------|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Recommended Universe:** COMBINED (for best cross-sectional differentiation)

---

## Outputs

The engine produces two JSON files:

### Tab 1 — `csi_ranker_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-08-01",
  "universes": {
    "COMBINED": {
      "top_buys": [
        {"ticker": "XLE", "z_score": 0.231},
        {"ticker": "GDX", "z_score": 0.180},
        {"ticker": "SLV", "z_score": 0.173}
      ],
      "top_sells": [
        {"ticker": "SPY", "z_score": -0.212},
        {"ticker": "XLY", "z_score": -0.204},
        {"ticker": "IWO", "z_score": -0.174}
      ],
      "full_scores": {
        "XLE": {
          "z_score": 0.231,
          "best_window": 252,
          "action": "STRONG BUY",
          "causal_links": 5,
          "position": 0.7
        }
      }
    }
  }
}
Tab 2 — csi_ranker_breakdown_YYYY-MM-DD.json
json
{
  "run_date": "2026-08-01",
  "universes": {
    "COMBINED": {
      "windows": {
        "252": {
          "top_buys": [
            {"ticker": "XLE", "z_score": 0.231}
          ],
          "full_ranking": [
            ["XLE", 0.231, "STRONG BUY"],
            ["GDX", 0.180, "STRONG BUY"],
            ["SPY", -0.212, "STRONG SELL"]
          ]
        }
      }
    }
  }
}
Dashboard Features
Tab	What it shows
Best Window per ETF	Each ETF's highest z-score window, with action
Explore by Window	All ETFs ranked for a selected window
Action Interpretation
Action	z-score range	What to do
STRONG BUY	> 0.5 (top 10%)	Add significantly
BUY	> 0.5 (top 20%)	Add moderately
HOLD	-0.5 to 0.5	Maintain position
REDUCE	< -0.5 (bottom 20%)	Reduce moderately
STRONG SELL	< -0.5 (bottom 10%)	Exit position
Setup
bash
git clone https://github.com/P2SAMAPA/P2-CSI-RANKER
cd P2-CSI-RANKER
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
References
Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. Science Advances.

Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods. Econometrica.

Jansen, M. (2023). Machine Learning for Algorithmic Trading. Packt Publishing.
