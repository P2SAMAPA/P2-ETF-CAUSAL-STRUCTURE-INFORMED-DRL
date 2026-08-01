# P2-CSI-DRL

**Causal-Structure-Informed DRL — Graph Neural Network for Causal Trading**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine implements a DRL trading agent that receives a dynamically updated causal graph (from PCMCI+) as its state representation. A Graph Neural Network encodes this causal graph, making the agent robust to non-stationary relationships.

### Theory

**PCMCI+ Causal Discovery:**
- Identifies causal relationships in time series
- Handles lagged effects and confounding variables
- Returns a causal graph of the market structure

**Graph Neural Network (GNN):**
- Encodes the causal graph structure
- Node features = asset characteristics
- Edge features = causal strengths
- Graph embedding = state representation

**DRL Agent:**
- Policy network on GNN embedding
- Robust to structural changes
- If a causal link breaks, GNN representation changes immediately

---

## Key Metrics

| Metric | What it tells you |
|--------|-------------------|
| **z-score** | Cross-sectional ranking of causal structure signal |
| **Causal Links** | Number of detected causal relationships |
| **Action Probabilities** | BUY/HOLD/SELL likelihood |
| **Position** | Optimal position size |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term causal structure |
| 252d | Core signal (primary) |
| 504d | Medium-term causal relationships |
| 1008d | Structural causal links |
| 2016d+ | Secular causal structure |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-CSI-DRL
cd P2-CSI-DRL
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
