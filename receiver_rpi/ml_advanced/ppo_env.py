"""
receiver/ml_advanced/ppo_env.py

Gymnasium-compatible environment that replays REAL logged telegram
graphs (from graph_observations.csv) so PPO can train offline, without
needing live hardware for every episode. Using real captured conditions
(not a synthetic channel model) avoids the sim-to-real gap that a purely
synthetic simulator would introduce.

Episode structure: one episode = one telegram. At each step, the agent
sees the TGAT representation of observations-so-far for that telegram
and chooses an action (use GW1 / GW2 / both). Reward reflects whether
that choice contributes to eventual reconstruction, per the doc's reward
shape (successful reconstruction, useful fragment, duplicate penalty,
extra-gateway cost).
"""

import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from graph_dataset import build_telegram_graphs, normalize_edge_features, TelegramGraph
from tgat_model import TGATEncoder

ACTION_GW1_ONLY = 0
ACTION_GW2_ONLY = 1
ACTION_BOTH = 2

REWARD_TELEGRAM_RECONSTRUCTED = 10.0
REWARD_NEW_FRAGMENT = 2.0
REWARD_DUPLICATE_PENALTY = -2.0
REWARD_EXTRA_GATEWAY_PENALTY = -1.0
REWARD_TELEGRAM_LOST = -10.0


class ISACLoRaReplayEnv(gym.Env):
    """One episode = one real logged telegram, replayed edge-by-edge in
    the order they were actually observed. Requires exactly 2 gateways
    (matches your current hardware)."""

    metadata = {"render_modes": []}

    def __init__(self, csv_path: str, tgat_checkpoint: str = None, seed: int = 42):
        super().__init__()
        graphs = build_telegram_graphs(csv_path)
        graphs, self.norm_stats = normalize_edge_features(graphs)
        self.graphs = [g for g in graphs if len(g.gateway_ids) >= 2]
        if not self.graphs:
            raise ValueError(
                "No telegrams with 2+ gateways found in the logged data -- "
                "this environment needs real dual-gateway observations to train on. "
                "Make sure both ESP32 gateways were running when data was collected."
            )
        print(f"ISACLoRaReplayEnv: {len(self.graphs)} usable telegram episodes loaded")

        self.rng = random.Random(seed)

        if tgat_checkpoint:
            ckpt = torch.load(tgat_checkpoint, map_location="cpu")
            self.encoder = TGATEncoder(hidden_dim=ckpt["hidden_dim"],
                                        temporal_dim=ckpt["temporal_dim"],
                                        out_dim=ckpt["out_dim"])
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
            print(f"Loaded pretrained TGAT encoder from {tgat_checkpoint}")
        else:
            self.encoder = TGATEncoder(hidden_dim=32, temporal_dim=16, out_dim=32)
            print("WARNING: no pretrained TGAT checkpoint given -- using randomly initialized "
                  "encoder. Strongly recommended to run tgat_pretrain.py first.")
        self.encoder.eval()

        repr_dim = 32
        extra_state_dim = 3
        obs_dim = repr_dim + extra_state_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._current_graph: TelegramGraph = None
        self._step_idx = 0
        self._used_gateways_this_telegram = set()
        self._recovered_fragments = set()

    def _get_obs(self):
        g = self._current_graph
        edges_so_far = min(self._step_idx, len(g.edge_gateway_idx))
        if edges_so_far == 0:
            h = torch.zeros(32)
        else:
            with torch.no_grad():
                h = self.encoder(
                    num_gateways=len(g.gateway_ids), num_fragments=len(g.fragment_ids),
                    edge_gateway_idx=torch.from_numpy(g.edge_gateway_idx[:edges_so_far]),
                    edge_fragment_idx=torch.from_numpy(g.edge_fragment_idx[:edges_so_far]),
                    edge_features=torch.from_numpy(g.edge_features[:edges_so_far]),
                )
        missing_frac = 1.0 - (len(self._recovered_fragments) / max(g.total_fragments, 1))
        progress = edges_so_far / max(len(g.edge_gateway_idx), 1)
        extra = np.array([missing_frac, progress, edges_so_far / 10.0], dtype=np.float32)
        return np.concatenate([h.numpy(), extra]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_graph = self.rng.choice(self.graphs)
        self._step_idx = 0
        self._used_gateways_this_telegram = set()
        self._recovered_fragments = set()
        return self._get_obs(), {}

    def step(self, action: int):
        g = self._current_graph
        reward = 0.0

        if self._step_idx < len(g.edge_gateway_idx):
            gw_idx = int(g.edge_gateway_idx[self._step_idx])
            frag_idx = int(g.edge_fragment_idx[self._step_idx])
            is_dup = bool(g.edge_features[self._step_idx, 3])

            gateway_used = (
                (action == ACTION_GW1_ONLY and gw_idx == 0) or
                (action == ACTION_GW2_ONLY and gw_idx == 1) or
                (action == ACTION_BOTH)
            )

            if gateway_used:
                if is_dup or frag_idx in self._recovered_fragments:
                    reward += REWARD_DUPLICATE_PENALTY
                else:
                    reward += REWARD_NEW_FRAGMENT
                    self._recovered_fragments.add(frag_idx)
                self._used_gateways_this_telegram.add(gw_idx)

            if action == ACTION_BOTH:
                reward += REWARD_EXTRA_GATEWAY_PENALTY

        self._step_idx += 1
        terminated = self._step_idx >= len(g.edge_gateway_idx)

        if terminated:
            fully_recovered = len(self._recovered_fragments) >= g.total_fragments
            if fully_recovered:
                reward += REWARD_TELEGRAM_RECONSTRUCTED
            elif g.label == 0:
                reward += REWARD_TELEGRAM_LOST

        return self._get_obs(), reward, terminated, False, {}
