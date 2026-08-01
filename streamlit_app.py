import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfApi
from datetime import date, timedelta
import config
import os

st.set_page_config(page_title="CSI-DRL Engine", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2.3rem;font-weight:700;color:#1a1a2e;margin-bottom:0.2rem}
.sub-header{font-size:1rem;color:#555;margin-bottom:1.5rem}
.uni-title{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
           padding-left:0.5rem;border-left:5px solid #e94560}
.hero-card{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(233,69,96,0.3)}
.win-card{background:linear-gradient(135deg,#0f3460 0%,#533483 100%);
          color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
          box-shadow:0 4px 12px rgba(83,52,131,0.3)}
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.badge-buy{background:#27ae60;border-radius:6px;padding:2px 12px;font-size:0.75rem;
           font-weight:700;color:white}
.badge-sell{background:#e74c3c;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-hold{background:#f39c12;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 CSI-DRL Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Causal-Structure-Informed DRL · GNN on Causal Graphs · '
    'PCMCI+ Causal Discovery · Learning on the Structure of Reality</div>',
    unsafe_allow_html=True)

HF_TOKEN = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
RESULTS_REPO = config.RESULTS_REPO

US_HOLIDAYS = {
    date(2025,1,1),date(2025,1,20),date(2025,2,17),date(2025,4,18),
    date(2025,5,26),date(2025,6,19),date(2025,7,4),date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def next_trading_day() -> str:
    d = date.today() + timedelta(days=1)
    while d.weekday() >= 5 or d in US_HOLIDAYS:
        d += timedelta(days=1)
    return d.strftime("%B %d, %Y")

def get_action(z_score: float) -> str:
    if z_score > 1.0:
        return "STRONG BUY"
    elif z_score > 0.5:
        return "BUY"
    elif z_score > -0.5:
        return "HOLD"
    elif z_score > -1.0:
        return "REDUCE"
    else:
        return "STRONG SELL"

def action_badge(action: str) -> str:
    if "BUY" in action:
        return f'<span class="badge-buy">🟢 {action}</span>'
    elif "SELL" in action:
        return f'<span class="badge-sell">🔴 {action}</span>'
    else:
        return f'<span class="badge-hold">🟡 {action}</span>'

def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


@st.cache_data(ttl=3600)
def list_repo_files():
    if not HF_TOKEN:
        return []
    try:
        api = HfApi(token=HF_TOKEN)
        return api.list_repo_files(repo_id=RESULTS_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception:
        return []


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f], reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json_from_hf(path):
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not set"}
    try:
        api = HfApi(token=HF_TOKEN)
        content = api.hf_hub_download(repo_id=RESULTS_REPO, filename=path, repo_type="dataset", token=HF_TOKEN)
        with open(content, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🧠 CSI-DRL")
st.sidebar.markdown(f"**Next Trading Day**")
st.sidebar.markdown(f"`{next_trading_day()}`")
st.sidebar.markdown(f"**Max Lag:** {config.CAUSAL['max_lag']}")
st.sidebar.markdown(f"**GNN Hidden Dim:** {config.GNN['hidden_dim']}")
st.sidebar.markdown(f"**GNN Layers:** {config.GNN['n_layers']}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Macro signals:**")
for col, desc, w, sign in config.MACRO_SIGNALS:
    arrow = "↑risk-on" if sign > 0 else "↑risk-off"
    st.sidebar.markdown(f"  • {col} ({arrow}, w={w:.0%})")

# ── Load data ─────────────────────────────────────────────────────────────────
files = list_repo_files()
if not files:
    st.error("No results found. Run trainer.py first.")
    st.stop()

tab1_path = find_latest(files, "csi_drl_")
tab2_path = find_latest(files, "csi_drl_breakdown_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json_from_hf(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2 = load_json_from_hf(tab2_path) if tab2_path else None

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")
st.sidebar.success(f"✅ {len(data1.get('universes', {}))} universes")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])

UNIVERSE_ORDER = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED": "🌐 Combined",
}

ntd = next_trading_day()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - BEST WINDOW PER ETF
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Best Window per ETF — CSI-DRL Signal")

    with st.expander("📖 How CSI-DRL Works", expanded=True):
        st.markdown("""
**Causal-Structure-Informed DRL** learns on the structure of reality:

| Step | What happens |
|------|-------------|
| 1. PCMCI+ | Discover causal relationships in time series |
| 2. Causal Graph | Build dynamic graph of causal links |
| 3. GNN Encoding | Graph Neural Network encodes the structure |
| 4. Policy Network | DRL agent acts on GNN embedding |
| 5. Robustness | If a causal link breaks, GNN changes immediately |

**Interpretation:**
- **Higher z-score** → Stronger causal-structure-informed signal
- **Causal Links** → Number of detected causal relationships
- **Action Probabilities** → BUY/HOLD/SELL likelihood
        """)

    for universe_name in UNIVERSE_ORDER:
        uni_data = data1.get("universes", {}).get(universe_name, {})
        if not uni_data:
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        full_scores = uni_data.get("full_scores", {})

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        # ── TOP BUYS ──────────────────────────────────────────────────────────
        buy_etfs = []
        for ticker, data in full_scores.items():
            z = safe_float(data.get("z_score", 0))
            if z > 0.5:
                buy_etfs.append((ticker, z, data))

        buy_etfs = sorted(buy_etfs, key=lambda x: x[1], reverse=True)

        if buy_etfs:
            cols = st.columns(3)
            for idx, (ticker, z_score, data) in enumerate(buy_etfs[:3]):
                best_window = data.get("window", "N/A")
                action = data.get("action", "HOLD")
                causal_links = int(safe_float(data.get("causal_links", 0)))

                with cols[idx]:
                    st.markdown(f"""
<div class="hero-card">
  <div class="ticker">⭐ {ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">Causal Links = {causal_links}</div>
  <div class="score">best window = {best_window}d</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No BUY signals in this universe")

        # ── FULL RANKING ──────────────────────────────────────────────────────
        with st.expander(f"📋 Full ranking — {label}"):
            if full_scores:
                rows = []
                for t, info in full_scores.items():
                    z = safe_float(info.get("z_score", 0))
                    rows.append({
                        "ETF": t,
                        "z-score": round(z, 4),
                        "Action": info.get("action", "HOLD"),
                        "Causal Links": int(safe_float(info.get("causal_links", 0))),
                        "Best Window (d)": info.get("window", "N/A")
                    })
                df_rank = pd.DataFrame(rows).sort_values("z-score", ascending=False)

                styled_df = df_rank.style.map(
                    lambda x: 'background-color: #27ae60; color: white;' if isinstance(x, (int, float)) and x > 0.5 else '',
                    subset=['z-score']
                ).map(
                    lambda x: 'background-color: #f1c40f; color: black;' if isinstance(x, (int, float)) and -0.5 < x <= 0.5 else '',
                    subset=['z-score']
                ).map(
                    lambda x: 'background-color: #e74c3c; color: white;' if isinstance(x, (int, float)) and x <= -0.5 else '',
                    subset=['z-score']
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · Higher z-score = stronger causal structure signal")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - EXPLORE BY WINDOW
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore CSI-DRL Signals by Window")

    if not data2:
        st.warning("Window-level data not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in data2.get("universes", {}).values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data.")
        st.stop()

    win_labels = {
        63: "63d  (~3 months)",
        252: "252d (~1 year)",
        504: "504d (~2 years)",
        1008: "1008d (~4 years)",
        2016: "2016d (~8 years)",
        4032: "4032d (~16 years)",
        4536: "4536d (~18 years)",
    }

    default_idx = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: win_labels.get(w, f"{w}d"),
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance", expanded=False):
        st.markdown("""
- **63d** — Short-term causal structure
- **252d** — Annual causal structure: recommended primary
- **504d–1008d** — Medium-term causal relationships
- **2016d+** — Very long-run causal structure
- **4032d / 4536d** — Full history causal structure
        """)

    st.markdown(f"### CSI-DRL Rankings at **{win_labels.get(selected_win, f'{selected_win}d')}** window")

    for universe_name in UNIVERSE_ORDER:
        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        uni_data = data2.get("universes", {}).get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        # ── TOP BUYS ──────────────────────────────────────────────────────────
        top_buys = win_data.get("top_buys", [])
        if top_buys:
            cols = st.columns(3)
            for idx, etf in enumerate(top_buys[:3]):
                ticker = etf["ticker"]
                z_score = safe_float(etf.get("z_score", 0))
                action = get_action(z_score)

                with cols[idx]:
                    st.markdown(f"""
<div class="win-card">
  <div class="ticker">⭐ {ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="next-day">window = {selected_win}d · 📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)

        # ── FULL RANKING ──────────────────────────────────────────────────────
        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df_win = pd.DataFrame(rows)
                df_win.columns = ["ETF", "z-score", "Action"]
                df_win.insert(0, "Rank", range(1, len(df_win) + 1))

                styled_df = df_win.style.map(
                    lambda x: 'background-color: #27ae60; color: white;' if isinstance(x, (int, float)) and x > 0.5 else '',
                    subset=['z-score']
                ).map(
                    lambda x: 'background-color: #f1c40f; color: black;' if isinstance(x, (int, float)) and -0.5 < x <= 0.5 else '',
                    subset=['z-score']
                ).map(
                    lambda x: 'background-color: #e74c3c; color: white;' if isinstance(x, (int, float)) and x <= -0.5 else '',
                    subset=['z-score']
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
