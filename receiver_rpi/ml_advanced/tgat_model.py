"""
receiver/ml_advanced/tgat_model.py

Temporal Graph Attention Network for the gateway<->fragment bipartite
graph. Requires PyTorch (see requirements_colab.txt). This file is NOT
executable in the RPi4 dev sandbox used to build this project (no GPU,
no torch installed there) -- it's written against stable, standard
nn.Module patterns and is meant to be smoke-tested first via
smoke_test.py in Colab before any real training run.

Architecture, matching the doc's design:
  raw edge features -> feature embedding -> temporal encoding ->
  graph attention (gateway/fragment cross-attention) -> pooled graph
  representation h_t

This graph representation is what PPO's policy network consumes (see
ppo_env.py). For supervised pretraining (tgat_pretrain.py), a small
classification head sits on top of h_t predicting telegram success --
that pretraining task is discarded after pretraining; only the
TGATEncoder itself is kept and fed into PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_dataset import EDGE_FEATURE_NAMES

EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)  # 6


class TemporalEncoding(nn.Module):
    """Encodes the 'time_since_telegram_start_s' edge feature (already
    present as one of the 6 raw features) into a higher-dimensional
    representation using fixed sinusoidal functions (same idea as
    Transformer positional encoding, applied to continuous time instead
    of discrete position) -- lets the attention layer use relative
    timing, not just the raw scalar."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim
        freqs = torch.exp(torch.linspace(0, -5, dim // 2))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class GraphAttentionLayer(nn.Module):
    """A single graph attention step: each gateway node attends over its
    connected fragment-edges, weighted by learned attention scores --
    this is what lets the model learn things like 'gateway 2's report on
    fragment 3 matters more than gateway 1's report on the same fragment,
    given current conditions' without being told that rule explicitly."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(node_dim, out_dim)
        self.key_proj = nn.Linear(node_dim + edge_dim, out_dim)
        self.value_proj = nn.Linear(node_dim + edge_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, gateway_feats: torch.Tensor, fragment_feats: torch.Tensor,
                edge_gateway_idx: torch.Tensor, edge_fragment_idx: torch.Tensor,
                edge_feats: torch.Tensor) -> torch.Tensor:
        num_gateways = gateway_feats.size(0)

        frag_per_edge = fragment_feats[edge_fragment_idx]
        kv_input = torch.cat([frag_per_edge, edge_feats], dim=-1)

        queries = self.query_proj(gateway_feats)
        keys = self.key_proj(kv_input)
        values = self.value_proj(kv_input)

        out = torch.zeros(num_gateways, queries.size(-1), device=gateway_feats.device)
        for g in range(num_gateways):
            mask = (edge_gateway_idx == g)
            if not mask.any():
                continue
            q = queries[g].unsqueeze(0)
            k = keys[mask]
            v = values[mask]

            scores = (q @ k.T) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            out[g] = (attn @ v).squeeze(0)

        return self.out_proj(out)


class TGATEncoder(nn.Module):
    """Full encoder: embeds edge features + temporal encoding, runs
    graph attention, and pools into a single fixed-size representation
    h_t summarizing the current telegram's reception state across both
    gateways -- this h_t is what PPO's policy consumes."""

    def __init__(self, hidden_dim: int = 32, temporal_dim: int = 16, out_dim: int = 32):
        super().__init__()
        self.temporal_enc = TemporalEncoding(dim=temporal_dim)

        edge_input_dim = (EDGE_FEATURE_DIM - 1) + temporal_dim

        self.node_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.attn1 = GraphAttentionLayer(hidden_dim, edge_input_dim, hidden_dim)
        self.attn2 = GraphAttentionLayer(hidden_dim, edge_input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _prep_edge_features(self, edge_features: torch.Tensor) -> torch.Tensor:
        time_col = edge_features[:, 2]
        other_cols = torch.cat([edge_features[:, :2], edge_features[:, 3:]], dim=-1)
        temporal = self.temporal_enc(time_col)
        return torch.cat([other_cols, temporal], dim=-1)

    def forward(self, num_gateways: int, num_fragments: int,
                edge_gateway_idx: torch.Tensor, edge_fragment_idx: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        gateway_feats = self.node_embed.expand(num_gateways, -1).clone()
        fragment_feats = self.node_embed.expand(num_fragments, -1).clone()

        edge_input = self._prep_edge_features(edge_features)

        h = self.attn1(gateway_feats, fragment_feats, edge_gateway_idx, edge_fragment_idx, edge_input)
        h = F.relu(h)
        h = self.attn2(h, fragment_feats, edge_gateway_idx, edge_fragment_idx, edge_input)

        pooled = h.mean(dim=0, keepdim=True)
        return self.out_proj(pooled).squeeze(0)


class TGATPretrainHead(nn.Module):
    """Supervised pretraining head: predicts P(telegram reconstructs)
    from the TGAT encoder's representation. Discarded after pretraining
    -- only TGATEncoder's weights are kept for the PPO stage."""

    def __init__(self, encoder: TGATEncoder, repr_dim: int = 32):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(repr_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, num_gateways, num_fragments, edge_gateway_idx, edge_fragment_idx, edge_features):
        h = self.encoder(num_gateways, num_fragments, edge_gateway_idx, edge_fragment_idx, edge_features)
        logit = self.classifier(h)
        return logit.squeeze(-1)
