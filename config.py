"""
config.py  —  Configuration for CSI-DRL Engine
==============================================

Defines:
  - UNIVERSES: ETF ticker sets
  - CAUSAL: PCMCI+ causal discovery parameters
  - GNN: Graph Neural Network parameters
  - RL: Reinforcement learning parameters
  - WINDOWS: Time windows for causal graph updates
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-csi-drl-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [63, 252, 504, 1008, 2016, 4032, 4536]
WINDOW_LABELS = {
    63: "63d  (~3 months) — Short-term",
    252: "252d (~1 year) — Core Signal",
    504: "504d (~2 years) — Medium-term",
    1008: "1008d (~4 years) — Structural",
    2016: "2016d (~8 years) — Secular",
    4032: "4032d (~16 years) — Long-term",
    4536: "4536d (~18 years) — Full History",
}
PRIMARY_WINDOW = 252


# ── Causal Discovery (PCMCI+) Parameters ──────────────────────────────────

CAUSAL = {
    "max_lag": 5,               # Maximum lag for causal links
    "alpha": 0.05,              # Significance level for causal tests
    "pc_alpha": 0.1,            # PC algorithm significance
    "n_permutations": 100,      # Permutations for significance testing
    "min_samples": 50,          # Minimum samples for causal discovery
}


# ── GNN Parameters ──────────────────────────────────────────────────────────

GNN = {
    "hidden_dim": 64,           # GNN hidden dimension
    "n_layers": 3,              # Number of GNN layers
    "node_dim": 16,             # Node feature dimension
    "edge_dim": 8,              # Edge feature dimension
    "pooling": "mean",          # Graph pooling method
}


# ── RL Parameters ───────────────────────────────────────────────────────────

RL = {
    "learning_rate": 0.001,     # Learning rate
    "batch_size": 64,           # Batch size for training
    "buffer_size": 10000,       # Replay buffer size
    "gamma": 0.99,              # Discount factor
    "tau": 0.005,               # Target network update rate
    "action_dim": 3,            # {BUY, HOLD, SELL}
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
