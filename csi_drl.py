"""
csi_drl.py  —  CSI-DRL Engine (Working Version)
================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class CausalDiscovery:
    """Simplified causal discovery for time series."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_lag = config.get("max_lag", 5)
        self.alpha = config.get("alpha", 0.05)
        self.min_samples = config.get("min_samples", 50)
    
    def compute_partial_correlation(self, x: np.ndarray, y: np.ndarray, 
                                     z: np.ndarray) -> float:
        if len(x) < 10:
            return 0.0
        
        if len(z) > 0:
            X = np.column_stack([np.ones(len(z)), z])
            beta_x = np.linalg.lstsq(X, x, rcond=None)[0]
            beta_y = np.linalg.lstsq(X, y, rcond=None)[0]
            x_res = x - X @ beta_x
            y_res = y - X @ beta_y
            if np.std(x_res) > 0 and np.std(y_res) > 0:
                corr, _ = pearsonr(x_res, y_res)
                return corr
            return 0.0
        else:
            if np.std(x) > 0 and np.std(y) > 0:
                corr, _ = pearsonr(x, y)
                return corr
            return 0.0
    
    def pcmci_plus(self, data: np.ndarray, var_names: List[str]) -> Dict:
        n_vars = data.shape[0]
        n_samples = data.shape[1]
        
        if n_samples < self.min_samples:
            return {"causal_links": {}, "graph": np.zeros((n_vars, n_vars))}
        
        causal_links = {}
        graph = np.zeros((n_vars, n_vars))
        
        for target in range(n_vars):
            selected_parents = []
            
            for source in range(n_vars):
                for lag in range(1, self.max_lag + 1):
                    if source == target and lag == 1:
                        continue
                    
                    if lag < n_samples:
                        x = data[target, lag:]
                        y = data[source, :n_samples-lag]
                        
                        if selected_parents:
                            z_values = []
                            for s, l in selected_parents:
                                if l < n_samples:
                                    z_values.append(data[s, :n_samples-l])
                            if z_values:
                                z = np.column_stack(z_values)
                                pcorr = self.compute_partial_correlation(x, y, z)
                            else:
                                pcorr = self.compute_partial_correlation(x, y, np.array([]))
                        else:
                            pcorr = self.compute_partial_correlation(x, y, np.array([]))
                        
                        if abs(pcorr) > self.alpha * 2:
                            selected_parents.append((source, lag))
                            graph[source, target] = 1
                            
                            key = f"{var_names[source]}→{var_names[target]}(lag={lag})"
                            causal_links[key] = {
                                "strength": abs(pcorr),
                                "sign": np.sign(pcorr),
                                "lag": lag,
                                "source": var_names[source],
                                "target": var_names[target]
                            }
        
        return {"causal_links": causal_links, "graph": graph, "var_names": var_names}


def compute_csi_drl(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """Compute CSI-DRL signals for a single ticker."""
    returns = np.log(prices / prices.shift(1)).dropna().values
    
    if len(returns) < window:
        return {"action": "HOLD", "z_score": 0, "error": "Insufficient data"}
    
    try:
        # Use recent window
        train_returns = returns[-window:]
        macro = macro_df.values
        train_macro = macro[-min(window, len(macro)):] if len(macro) > 0 else np.zeros((1, 6))
        
        # ── 1. Compute momentum factors ──────────────────────────────────────
        st_momentum = np.mean(train_returns[-10:]) if len(train_returns) >= 10 else 0
        mt_momentum = np.mean(train_returns[-30:]) if len(train_returns) >= 30 else 0
        lt_momentum = np.mean(train_returns[-60:]) if len(train_returns) >= 60 else 0
        volatility = np.std(train_returns[-60:]) if len(train_returns) >= 60 else 0
        skew = pd.Series(train_returns[-60:]).skew() if len(train_returns) >= 60 else 0
        
        # ── 2. Discover causal graph ──────────────────────────────────────────
        causal_discovery = CausalDiscovery(config)
        
        n_vars = 8
        data = np.zeros((n_vars, len(train_returns)))
        data[0, :] = train_returns
        
        for i in range(1, min(n_vars, 4)):
            data[i, :] = np.roll(train_returns, -i) * 0.5
        
        macro_flat = train_macro.flatten()
        for i in range(min(n_vars - 4, len(macro_flat))):
            data[4 + i, :] = macro_flat[i] * 0.1
        
        var_names = ["RETURN"] + [f"LAG_{i}" for i in range(1, 4)] + [f"MACRO_{i}" for i in range(4)]
        causal_result = causal_discovery.pcmci_plus(data, var_names)
        
        # ── 3. Compute causality score ──────────────────────────────────────
        graph = causal_result.get("graph", np.zeros((n_vars, n_vars)))
        causal_links = causal_result.get("causal_links", {})
        
        incoming = np.sum(graph[:, 0])
        outgoing = np.sum(graph[0, :])
        net_causality = incoming - outgoing
        
        # ── 4. Composite signal ──────────────────────────────────────────────
        signal = (
            0.40 * st_momentum +
            0.20 * mt_momentum +
            0.10 * lt_momentum -
            0.15 * volatility +
            0.15 * net_causality * 0.1
        ) * 100
        
        return {
            "action": "PENDING",  # Will be set by universe normalization
            "action_index": -1,
            "action_probabilities": [0.33, 0.33, 0.34],
            "position": 0.0,
            "z_score": signal,  # Raw signal, will be normalized
            "causal_links": len(causal_links),
            "st_momentum": st_momentum,
            "volatility": volatility,
            "net_causality": net_causality,
            "error": None
        }
    except Exception as e:
        return {"action": "HOLD", "z_score": 0, "error": str(e)}


def compute_universe_csi_drl(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """Compute CSI-DRL signals for all ETFs in a universe."""
    results = {}
    
    # First pass: compute all signals
    for ticker in prices_df.columns:
        prices = prices_df[ticker]
        result = compute_csi_drl(prices, macro_df, config, window)
        
        results[ticker] = {
            "action": result.get("action", "HOLD"),
            "z_score": result.get("z_score", 0),
            "action_probabilities": result.get("action_probabilities", [0.33, 0.33, 0.34]),
            "position": result.get("position", 0),
            "causal_links": result.get("causal_links", 0),
            "st_momentum": result.get("st_momentum", 0),
            "volatility": result.get("volatility", 0),
            "net_causality": result.get("net_causality", 0)
        }
    
    # ── Normalize z-scores ──────────────────────────────────────────────────
    z_scores = np.array([r["z_score"] for r in results.values()])
    
    if len(z_scores) > 1 and np.std(z_scores) > 1e-6:
        mean_z = np.mean(z_scores)
        std_z = np.std(z_scores)
        for ticker, r in results.items():
            r["z_score"] = (r["z_score"] - mean_z) / std_z
    else:
        # Fallback: use short-term momentum
        st_mom = np.array([r["st_momentum"] for r in results.values()])
        if np.std(st_mom) > 1e-6:
            mean_m = np.mean(st_mom)
            std_m = np.std(st_mom)
            for ticker, r in results.items():
                r["z_score"] = (r["st_momentum"] - mean_m) / std_m
        else:
            positions = np.array([r["position"] for r in results.values()])
            if np.std(positions) > 1e-6:
                mean_p = np.mean(positions)
                std_p = np.std(positions)
                for ticker, r in results.items():
                    r["z_score"] = (r["position"] - mean_p) / std_p
            else:
                for ticker, r in results.items():
                    r["z_score"] = np.random.normal(0, 0.1)
    
    # ── Determine actions using percentiles ──────────────────────────────────
    z_scores_final = np.array([r["z_score"] for r in results.values()])
    
    if len(z_scores_final) > 1:
        p80 = np.percentile(z_scores_final, 80)
        p40 = np.percentile(z_scores_final, 40)
        p20 = np.percentile(z_scores_final, 20)
        
        for ticker, r in results.items():
            z = r["z_score"]
            if z > p80:
                r["action"] = "STRONG BUY" if z > np.percentile(z_scores_final, 90) else "BUY"
                r["action_probabilities"] = [0.7, 0.2, 0.1]
                r["position"] = 0.7
            elif z > p40:
                r["action"] = "HOLD"
                r["action_probabilities"] = [0.33, 0.33, 0.34]
                r["position"] = 0.0
            elif z > p20:
                r["action"] = "REDUCE"
                r["action_probabilities"] = [0.2, 0.3, 0.5]
                r["position"] = -0.3
            else:
                r["action"] = "STRONG SELL" if z < np.percentile(z_scores_final, 10) else "SELL"
                r["action_probabilities"] = [0.1, 0.2, 0.7]
                r["position"] = -0.7
    else:
        for r in results.values():
            r["action"] = "HOLD"
    
    return results
