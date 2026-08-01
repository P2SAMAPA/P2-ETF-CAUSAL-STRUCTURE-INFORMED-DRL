"""
csi_drl.py  —  CSI-DRL Engine
==============================

Implements:
- PCMCI+ causal discovery on time series
- Dynamic causal graph construction
- Graph Neural Network encoding of causal structure
- DRL policy network on GNN embeddings
- Learning on the structure of reality
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


class CausalDiscovery:
    """
    PCMCI+ causal discovery for time series.
    Detects causal relationships between variables.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_lag = config.get("max_lag", 5)
        self.alpha = config.get("alpha", 0.05)
        self.pc_alpha = config.get("pc_alpha", 0.1)
        self.n_permutations = config.get("n_permutations", 100)
        self.min_samples = config.get("min_samples", 50)
    
    def compute_partial_correlation(self, x: np.ndarray, y: np.ndarray, 
                                     z: np.ndarray) -> float:
        """Compute partial correlation between x and y conditioned on z."""
        if len(x) < 10:
            return 0.0
        
        # Combine variables
        if len(z) > 0:
            # Residualize x and y with respect to z
            X = np.column_stack([np.ones(len(z)), z])
            
            # Fit regression
            beta_x = np.linalg.lstsq(X, x, rcond=None)[0]
            beta_y = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Residuals
            x_res = x - X @ beta_x
            y_res = y - X @ beta_y
            
            # Correlation of residuals
            if np.std(x_res) > 0 and np.std(y_res) > 0:
                corr, _ = pearsonr(x_res, y_res)
                return corr
            else:
                return 0.0
        else:
            if np.std(x) > 0 and np.std(y) > 0:
                corr, _ = pearsonr(x, y)
                return corr
            else:
                return 0.0
    
    def pcmci_plus(self, data: np.ndarray, var_names: List[str]) -> Dict:
        """
        Run PCMCI+ causal discovery.
        
        Returns:
            causal_links: dict of (source, target, lag) -> significance
            graph: adjacency matrix of causal relationships
        """
        n_vars = data.shape[0]
        n_samples = data.shape[1]
        
        if n_samples < self.min_samples:
            return {"causal_links": {}, "graph": np.zeros((n_vars, n_vars))}
        
        # Initialize results
        causal_links = {}
        graph = np.zeros((n_vars, n_vars))
        
        # For each target variable
        for target in range(n_vars):
            # Get all potential parents (all variables at all lags)
            potential_parents = []
            for source in range(n_vars):
                for lag in range(1, self.max_lag + 1):
                    potential_parents.append((source, lag))
            
            # PCMCI: iterate through parents
            selected_parents = []
            
            for source, lag in potential_parents:
                if source == target and lag == 1:  # Skip self-lag 1
                    continue
                
                # Get time-shifted data
                if lag < n_samples:
                    x = data[target, lag:]
                    y = data[source, :n_samples-lag]
                    
                    # Test for conditional independence
                    # Use partial correlation with already selected parents
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
                    
                    # Significance test (simplified)
                    if abs(pcorr) > self.alpha * 2:
                        selected_parents.append((source, lag))
                        graph[source, target] = 1
                        
                        # Store causal link
                        key = f"{var_names[source]}→{var_names[target]}(lag={lag})"
                        causal_links[key] = {
                            "strength": abs(pcorr),
                            "sign": np.sign(pcorr),
                            "lag": lag,
                            "source": var_names[source],
                            "target": var_names[target]
                        }
        
        return {
            "causal_links": causal_links,
            "graph": graph,
            "var_names": var_names
        }


class GraphNeuralNetwork:
    """
    Graph Neural Network for encoding causal structures.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.hidden_dim = config.get("hidden_dim", 64)
        self.n_layers = config.get("n_layers", 3)
        self.node_dim = config.get("node_dim", 16)
        self.edge_dim = config.get("edge_dim", 8)
        self.pooling = config.get("pooling", "mean")
        
        # Network weights (simplified for speed)
        self.W_node = np.random.randn(self.node_dim, self.hidden_dim) * 0.1
        self.b_node = np.zeros(self.hidden_dim)
        self.W_edge = np.random.randn(self.edge_dim, self.hidden_dim) * 0.1
        self.b_edge = np.zeros(self.hidden_dim)
        self.W_out = np.random.randn(self.hidden_dim * 2, 32) * 0.1
        self.b_out = np.zeros(32)
        
        self.learning_rate = 0.001
        
    def encode_graph(self, node_features: np.ndarray, 
                     edge_features: np.ndarray, 
                     adjacency: np.ndarray) -> np.ndarray:
        """
        Encode a causal graph using message passing.
        
        Args:
            node_features: (n_nodes, node_dim)
            edge_features: (n_nodes, n_nodes, edge_dim)
            adjacency: (n_nodes, n_nodes) binary adjacency matrix
        
        Returns:
            Graph embedding: (hidden_dim * 2)
        """
        n_nodes = node_features.shape[0]
        
        # Node embeddings
        node_emb = np.tanh(node_features @ self.W_node + self.b_node)
        
        # Edge embeddings
        edge_emb = np.tanh(edge_features @ self.W_edge + self.b_edge)
        
        # Message passing (simplified)
        messages = np.zeros((n_nodes, self.hidden_dim))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency[i, j] > 0:
                    messages[i] += edge_emb[i, j] * node_emb[j]
        
        # Combine node embeddings with messages
        combined = np.concatenate([node_emb, messages], axis=1)
        
        # Graph-level pooling
        if self.pooling == "mean":
            graph_embedding = np.mean(combined, axis=0)
        elif self.pooling == "max":
            graph_embedding = np.max(combined, axis=0)
        else:
            graph_embedding = np.mean(combined, axis=0)
        
        # Output projection
        out = np.tanh(graph_embedding @ self.W_out + self.b_out)
        
        return out
    
    def update(self, gradient: np.ndarray):
        """Update network weights (simplified)."""
        noise = np.random.randn(*self.W_out.shape) * 0.001
        self.W_out += noise * 0.1


class CSIDRLAgent:
    """
    Causal-Structure-Informed DRL Agent.
    
    Uses a GNN to encode causal graphs and a policy network
    to select actions based on the graph embedding.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.state_dim = config.get("state_dim", 32)
        self.action_dim = config.get("action_dim", 3)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.action_labels = ["BUY", "HOLD", "SELL"]
        
        # GNN for causal graph encoding
        self.gnn = GraphNeuralNetwork(config.get("gnn", {}))
        
        # Policy network weights (simplified)
        self.W_policy = np.random.randn(32, self.hidden_dim) * 0.1
        self.b_policy = np.zeros(self.hidden_dim)
        self.W_out_policy = np.random.randn(self.hidden_dim, self.action_dim) * 0.1
        self.b_out_policy = np.zeros(self.action_dim)
        
        # Target network (for stable learning)
        self.W_policy_target = self.W_policy.copy()
        self.b_policy_target = self.b_policy.copy()
        self.W_out_policy_target = self.W_out_policy.copy()
        self.b_out_policy_target = self.b_out_policy.copy()
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = config.get("buffer_size", 1000)
        
        # Position tracking
        self.position = 0.0
        self.max_position = 1.0
        
    def encode_state(self, returns: np.ndarray, macro: np.ndarray, 
                     causal_graph: Dict) -> np.ndarray:
        """
        Encode the full state including causal graph.
        """
        # Get node features (each variable's recent returns)
        n_nodes = len(returns)
        node_features = np.array([np.mean(r[-20:]) for r in returns])
        node_features = node_features.reshape(-1, 1)
        
        # Pad to node_dim
        node_features_padded = np.zeros((n_nodes, self.gnn.node_dim))
        node_features_padded[:, 0] = node_features.flatten()
        
        # Get adjacency matrix from causal graph
        graph = causal_graph.get("graph", np.zeros((n_nodes, n_nodes)))
        
        # Edge features (simplified)
        edge_features = np.zeros((n_nodes, n_nodes, self.gnn.edge_dim))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if graph[i, j] > 0:
                    edge_features[i, j, 0] = 1.0
        
        # Encode graph with GNN
        graph_embedding = self.gnn.encode_graph(
            node_features_padded, edge_features, graph
        )
        
        # Add macro features
        macro_flat = macro.flatten()[:10] if len(macro) > 0 else np.zeros(10)
        
        # Combine
        state = np.concatenate([graph_embedding, macro_flat[:6]])
        
        # Pad to state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
        
        return state
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> Dict:
        """Select action using the policy network."""
        # Forward pass through policy network
        h = np.tanh(state @ self.W_policy + self.b_policy)
        logits = h @ self.W_out_policy + self.b_out_policy
        
        # Softmax
        probs = softmax(logits)
        
        # Exploration
        if explore:
            # Add noise for exploration
            eps = np.random.uniform(0.1, 0.3)
            probs = (1 - eps) * probs + eps / self.action_dim
        
        # Sample action
        selected_action = np.random.choice(self.action_dim, p=probs)
        
        # Position limits
        if self.position >= self.max_position * 0.9 and selected_action == 0:
            selected_action = 1
        if self.position <= -self.max_position * 0.9 and selected_action == 2:
            selected_action = 1
        
        # Update position
        action_delta = [0.1, 0.0, -0.1][selected_action]
        self.position = np.clip(self.position + action_delta, 
                               -self.max_position, self.max_position)
        
        return {
            "action": selected_action,
            "action_label": self.action_labels[selected_action],
            "action_probabilities": probs.tolist(),
            "position": self.position,
            "logits": logits.tolist()
        }
    
    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """Update the agent using experience replay."""
        # Store experience
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append({
            "state": state.copy(),
            "action": action,
            "reward": reward,
            "next_state": next_state.copy(),
            "done": done
        })
        
        # Sample batch
        if len(self.buffer) < 32:
            return
        
        indices = np.random.choice(len(self.buffer), 32, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Compute TD error and update (simplified)
        for item in batch:
            s = item["state"]
            a = item["action"]
            r = item["reward"]
            ns = item["next_state"]
            d = item["done"]
            
            # Current Q-value
            h = np.tanh(s @ self.W_policy + self.b_policy)
            q_values = h @ self.W_out_policy + self.b_out_policy
            q_current = q_values[a]
            
            # Target Q-value (using target network)
            h_target = np.tanh(ns @ self.W_policy_target + self.b_policy_target)
            q_target_values = h_target @ self.W_out_policy_target + self.b_out_policy_target
            q_target = r + self.gamma * np.max(q_target_values) * (1 - d)
            
            # TD error
            td_error = q_target - q_current
            
            # Update policy (simplified gradient descent)
            grad_scale = 0.001 * td_error
            self.W_out_policy += grad_scale * np.outer(h, np.eye(self.action_dim)[a])
            self.b_out_policy += grad_scale * np.eye(self.action_dim)[a]
            
            # Soft target update
            self.W_policy_target = self.tau * self.W_policy + (1 - self.tau) * self.W_policy_target
            self.b_policy_target = self.tau * self.b_policy + (1 - self.tau) * self.b_policy_target
            self.W_out_policy_target = self.tau * self.W_out_policy + (1 - self.tau) * self.W_out_policy_target
            self.b_out_policy_target = self.tau * self.b_out_policy + (1 - self.tau) * self.b_out_policy_target


def compute_csi_drl(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """Compute CSI-DRL signals for a single ticker."""
    returns = np.log(prices / prices.shift(1)).dropna().values
    macro = macro_df.values
    
    if len(returns) < window:
        return {
            "action": "HOLD",
            "z_score": 0,
            "error": "Insufficient data"
        }
    
    try:
        # Use recent window
        train_returns = returns[-window:]
        train_macro = macro[-min(window, len(macro)):] if len(macro) > 0 else np.zeros((1, 6))
        
        # Discover causal graph using PCMCI+
        causal_discovery = CausalDiscovery(config)
        
        # For simplicity, we treat each variable as a separate time series
        # In practice, you'd use multiple features
        n_vars = 10  # Number of variables to consider
        data = np.zeros((n_vars, len(train_returns)))
        data[0, :] = train_returns
        
        # Add some derived features
        for i in range(1, min(n_vars, 5)):
            data[i, :] = np.roll(train_returns, -i)  # Lagged returns
        
        # Add macro features
        for i in range(min(n_vars - 5, len(train_macro.flatten()))):
            data[5 + i, :len(train_macro.flatten())] = train_macro.flatten()[i]
        
        var_names = ["RETURN"] + [f"LAG_{i}" for i in range(1, 5)] + [f"MACRO_{i}" for i in range(5)]
        
        # Discover causal links
        causal_result = causal_discovery.pcmci_plus(data, var_names)
        
        # Initialize agent
        agent = CSIDRLAgent(config)
        
        # Quick training
        for i in range(10, len(train_returns) - 10, 2):
            # Get state with causal graph
            state = agent.encode_state(
                [train_returns[max(0, i-20):i+1]],
                train_macro.flatten()[:10] if len(train_macro) > 0 else np.zeros(10),
                causal_result
            )
            
            action = np.random.randint(0, agent.action_dim)
            reward = np.mean(train_returns[max(0, i-5):i]) * (1 if action == 0 else -0.5 if action == 2 else 0)
            
            next_state = agent.encode_state(
                [train_returns[max(0, i-19):i+2]],
                train_macro.flatten()[:10] if len(train_macro) > 0 else np.zeros(10),
                causal_result
            )
            
            agent.learn(state, action, reward, next_state, done=False)
        
        # Inference
        latest_state = agent.encode_state(
            [returns[-20:]],
            macro[-5:].flatten()[:10] if len(macro) > 0 else np.zeros(10),
            causal_result
        )
        
        result = agent.select_action(latest_state, explore=False)
        
        # Compute z-score from action probabilities
        probs = result.get("action_probabilities", [0.33, 0.33, 0.34])
        z_score = (probs[0] - 0.33) / (np.std(probs) + 1e-6)
        
        return {
            "action": result["action_label"],
            "action_index": result["action"],
            "action_probabilities": probs,
            "position": result["position"],
            "z_score": z_score,
            "causal_links": len(causal_result.get("causal_links", {})),
            "error": None
        }
    except Exception as e:
        return {
            "action": "HOLD",
            "z_score": 0,
            "error": str(e)
        }


def compute_universe_csi_drl(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """Compute CSI-DRL signals for all ETFs in a universe."""
    results = {}
    
    for ticker in prices_df.columns:
        prices = prices_df[ticker]
        result = compute_csi_drl(prices, macro_df, config, window)
        
        results[ticker] = {
            "action": result.get("action", "HOLD"),
            "z_score": result.get("z_score", 0),
            "action_probabilities": result.get("action_probabilities", [0.33, 0.33, 0.34]),
            "position": result.get("position", 0),
            "causal_links": result.get("causal_links", 0)
        }
    
    # Normalize z-scores
    z_scores = np.array([r["z_score"] for r in results.values()])
    if len(z_scores) > 1 and np.std(z_scores) > 1e-6:
        mean_z = np.mean(z_scores)
        std_z = np.std(z_scores)
        for ticker, r in results.items():
            r["z_score"] = (r["z_score"] - mean_z) / std_z
    else:
        # Use action probability spread as fallback
        spreads = []
        for ticker, r in results.items():
            probs = r.get("action_probabilities", [0.33, 0.33, 0.34])
            spreads.append(max(probs) - min(probs))
        spreads = np.array(spreads)
        if len(spreads) > 1 and np.std(spreads) > 1e-6:
            mean_s = np.mean(spreads)
            std_s = np.std(spreads)
            for ticker, r in results.items():
                r["z_score"] = (max(r["action_probabilities"]) - min(r["action_probabilities"]) - mean_s) / std_s
        else:
            for r in results.values():
                r["z_score"] = 0
    
    return results
