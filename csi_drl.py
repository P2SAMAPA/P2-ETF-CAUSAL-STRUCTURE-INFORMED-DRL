"""
csi_drl.py  —  CSI-DRL Engine (PyTorch Implementation)
========================================================

Implements:
- PCMCI+ causal discovery on time series
- Dynamic causal graph construction
- Graph Neural Network (GNN) with PyTorch
- DRL policy network on GNN embeddings
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


# ─── PyTorch GNN Models ──────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer with message passing."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        x: (n_nodes, in_dim)
        adjacency: (n_nodes, n_nodes)
        """
        # Message passing: A * x
        messages = torch.mm(adjacency, x)
        # Transform
        out = self.W(messages) + self.b
        return F.relu(out)


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for encoding causal graphs."""
    
    def __init__(self, node_dim: int = 16, hidden_dim: int = 64, 
                 edge_dim: int = 8, n_layers: int = 3):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Node feature encoder
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Edge feature encoder
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, 32)
        
        # Graph pooling (attention-based)
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_features: torch.Tensor, 
                edge_features: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        node_features: (n_nodes, node_dim)
        edge_features: (n_nodes, n_nodes, edge_dim)
        adjacency: (n_nodes, n_nodes)
        """
        n_nodes = node_features.shape[0]
        
        # Encode node features
        x = F.relu(self.node_encoder(node_features))
        
        # Encode edge features (aggregate)
        edge_agg = edge_features.mean(dim=1)  # (n_nodes, edge_dim)
        edge_emb = F.relu(self.edge_encoder(edge_agg))
        x = x + edge_emb
        
        # Graph convolution with message passing
        for conv in self.convs:
            x = conv(x, adjacency) + x  # Residual connection
        
        # Graph pooling (attention-based)
        attention_weights = F.softmax(self.attention(x), dim=0)  # (n_nodes, 1)
        graph_embedding = (x * attention_weights).sum(dim=0)  # (hidden_dim,)
        
        # Final projection
        out = F.tanh(self.out_proj(graph_embedding))
        
        return out


class PolicyNetwork(nn.Module):
    """Policy network for DRL agent."""
    
    def __init__(self, state_dim: int = 32, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetwork(nn.Module):
    """Q-network for DRL agent."""
    
    def __init__(self, state_dim: int = 32, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ─── Causal Discovery ──────────────────────────────────────────────────────

class CausalDiscovery:
    """PCMCI+ causal discovery for time series."""
    
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


# ─── CSI-DRL Agent ──────────────────────────────────────────────────────

class CSIDRLAgent:
    """Causal-Structure-Informed DRL Agent with PyTorch."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Dimensions
        self.state_dim = config.get("state_dim", 32)
        self.action_dim = config.get("action_dim", 3)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.node_dim = config.get("node_dim", 16)
        self.edge_dim = config.get("edge_dim", 8)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.action_labels = ["BUY", "HOLD", "SELL"]
        
        # GNN
        self.gnn = GraphNeuralNetwork(
            node_dim=self.node_dim,
            hidden_dim=self.hidden_dim,
            edge_dim=self.edge_dim,
            n_layers=config.get("n_layers", 3)
        ).to(self.device)
        
        # Policy network
        self.policy = PolicyNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim
        ).to(self.device)
        
        # Target policy network
        self.policy_target = PolicyNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim
        ).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = min(config.get("buffer_size", 1000), 500)
        
        # Position tracking
        self.position = 0.0
        self.max_position = 1.0
        
        # Learning counter
        self.learn_step = 0
        
        # Causal graph cache
        self.causal_graph = None
        self.graph_embedding = None
        
    def _numpy_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).to(self.device)
    
    def encode_state(self, returns: np.ndarray, macro: np.ndarray, 
                     causal_graph: Dict) -> np.ndarray:
        """Encode the full state including causal graph."""
        n_nodes = len(returns)
        
        # Node features
        node_features = torch.zeros((n_nodes, self.node_dim)).to(self.device)
        for i, r in enumerate(returns):
            if len(r) > 0:
                recent = r[-20:]
                node_features[i, 0] = np.mean(recent)
                node_features[i, 1] = np.std(recent)
                node_features[i, 2] = recent[-1] if len(recent) > 0 else 0
                node_features[i, 3] = np.mean(recent[-5:]) if len(recent) >= 5 else 0
        
        # Adjacency matrix
        graph = torch.tensor(causal_graph.get("graph", np.zeros((n_nodes, n_nodes))), 
                            dtype=torch.float32).to(self.device)
        
        # Edge features
        edge_features = torch.zeros((n_nodes, n_nodes, self.edge_dim)).to(self.device)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if graph[i, j] > 0:
                    edge_features[i, j, 0] = 1.0
                    edge_features[i, j, 1] = 0.5
        
        # Encode graph with GNN
        with torch.no_grad():
            graph_embedding = self.gnn(node_features, edge_features, graph)
            graph_embedding_np = graph_embedding.cpu().numpy()
        
        # Add macro features
        macro_flat = macro.flatten()[:6] if len(macro) > 0 else np.zeros(6)
        position_feature = np.array([self.position])
        
        # Combine
        state = np.concatenate([graph_embedding_np, macro_flat, position_feature])
        
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
        
        self.causal_graph = causal_graph
        self.graph_embedding = graph_embedding_np
        
        return state
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> Dict:
        """Select action using the policy network."""
        state_tensor = self._numpy_to_tensor(state)
        
        with torch.no_grad():
            logits = self.policy(state_tensor)
            probs = F.softmax(logits / 0.5, dim=-1)
            probs_np = probs.cpu().numpy()
        
        # Exploration
        if explore:
            eps = max(0.05, 0.3 * (1 - self.learn_step / 1000))
            probs_np = (1 - eps) * probs_np + eps / self.action_dim
        
        # Sample action
        selected_action = np.random.choice(self.action_dim, p=probs_np)
        
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
            "action_probabilities": probs_np.tolist(),
            "position": self.position,
            "logits": logits.cpu().numpy().tolist()
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
        
        indices = np.random.choice(len(self.buffer), min(32, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Convert batch to tensors
        states = torch.stack([self._numpy_to_tensor(b["state"]) for b in batch])
        actions = torch.tensor([b["action"] for b in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([self._numpy_to_tensor(b["next_state"]) for b in batch])
        dones = torch.tensor([b["done"] for b in batch], dtype=torch.float32).to(self.device)
        
        # Compute current Q-values
        logits = self.policy(states)
        q_values = F.softmax(logits, dim=-1)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_logits = self.policy_target(next_states)
            next_q = F.softmax(next_logits, dim=-1)
            max_next_q = next_q.max(dim=1)[0]
            q_target = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_current, q_target)
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft target update
        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.learn_step += 1
        
        return loss.item()


# ─── Wrapper Functions ──────────────────────────────────────────────────────

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
        
        # Discover causal graph
        causal_discovery = CausalDiscovery(config)
        
        # Build features for causal discovery
        n_vars = 8
        data = np.zeros((n_vars, len(train_returns)))
        data[0, :] = train_returns
        
        for i in range(1, min(n_vars, 4)):
            data[i, :] = np.roll(train_returns, -i) * 0.5
        
        macro_flat = train_macro.flatten()
        for i in range(min(n_vars - 4, len(macro_flat))):
            data[4 + i, :] = macro_flat[i] * 0.1
        
        var_names = ["RETURN"] + [f"LAG_{i}" for i in range(1, 4)] + [f"MACRO_{i}" for i in range(4)]
        
        # Discover causal links
        causal_result = causal_discovery.pcmci_plus(data, var_names)
        
        # Initialize agent
        agent = CSIDRLAgent(config)
        
        # Training loop
        for epoch in range(5):
            for i in range(15, len(train_returns) - 15, 2):
                returns_window = [train_returns[max(0, i-20):i+1]]
                macro_window = train_macro.flatten()[:10] if len(train_macro) > 0 else np.zeros(10)
                
                state = agent.encode_state(returns_window, macro_window, causal_result)
                action = np.random.randint(0, agent.action_dim)
                
                future_returns = train_returns[i+1:min(i+6, len(train_returns))]
                if len(future_returns) > 0:
                    reward = np.mean(future_returns) * (1.0 if action == 0 else -0.5 if action == 2 else 0.0)
                else:
                    reward = 0
                
                next_returns_window = [train_returns[max(0, i-19):i+2]]
                next_state = agent.encode_state(next_returns_window, macro_window, causal_result)
                
                agent.learn(state, action, reward, next_state, done=False)
        
        # Inference
        latest_returns = [returns[-20:]]
        latest_macro = macro[-5:].flatten()[:10] if len(macro) > 0 else np.zeros(10)
        
        final_state = agent.encode_state(latest_returns, latest_macro, causal_result)
        result = agent.select_action(final_state, explore=False)
        
        # Compute z-score
        probs = np.array(result.get("action_probabilities", [0.33, 0.33, 0.34]))
        buy_prob = probs[0]
        z_score = (buy_prob - 0.33) / (np.std(probs) + 1e-6)
        
        if abs(z_score) < 0.01:
            logits = np.array(result.get("logits", [0, 0, 0]))
            z_score = (logits[0] - np.mean(logits)) / (np.std(logits) + 1e-6)
        
        return {
            "action": result["action_label"],
            "action_index": result["action"],
            "action_probabilities": probs.tolist(),
            "position": result["position"],
            "z_score": z_score,
            "causal_links": len(causal_result.get("causal_links", {})),
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
        # Use position as fallback
        positions = np.array([r["position"] for r in results.values()])
        if np.std(positions) > 1e-6:
            mean_p = np.mean(positions)
            std_p = np.std(positions)
            for ticker, r in results.items():
                r["z_score"] = (r["position"] - mean_p) / std_p
        else:
            for r in results.values():
                r["z_score"] = 0
    
    return results
