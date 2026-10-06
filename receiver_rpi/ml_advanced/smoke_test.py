"""
receiver/ml_advanced/smoke_test.py

RUN THIS FIRST, before any real training. Takes under a minute. Verifies
the TGAT model actually runs (forward pass, backward pass, no shape
errors) using tiny synthetic data -- catches any bug in tgat_model.py
immediately, rather than discovering it 20 minutes into a real training
run.

Usage (in Colab, after installing requirements_colab.txt):
    python3 smoke_test.py
"""

import sys
import torch
import numpy as np

from graph_dataset import TelegramGraph, EDGE_FEATURE_NAMES
from tgat_model import TGATEncoder, TGATPretrainHead


def make_dummy_graph(num_gateways=2, num_fragments=3, telegram_id=1, node_id=1, label=1) -> TelegramGraph:
    """Builds a small synthetic TelegramGraph with random-but-plausible
    values, for shape/execution testing only -- NOT for training on."""
    edges = []
    for gw in range(num_gateways):
        for fr in range(num_fragments):
            if np.random.rand() < 0.7:  # not every gateway hears every fragment
                edges.append((gw, fr))
    if not edges:
        edges = [(0, 0)]  # ensure at least one edge

    edge_gw = np.array([e[0] for e in edges], dtype=np.int64)
    edge_fr = np.array([e[1] for e in edges], dtype=np.int64)
    n = len(edges)
    edge_feat = np.stack([
        np.random.uniform(-100, -60, n),   # rssi_db
        np.random.uniform(0, 15, n),         # snr_db
        np.random.uniform(0, 5, n),           # time_since_telegram_start_s
        (np.random.rand(n) < 0.1).astype(np.float32),  # is_duplicate
        np.random.uniform(0, 1, n),           # fragments_seen_before_this_norm
        np.random.uniform(0, 1, n),           # telegram_progress
    ], axis=1).astype(np.float32)

    return TelegramGraph(
        telegram_id=telegram_id, node_id=node_id,
        gateway_ids=list(range(num_gateways)), fragment_ids=list(range(num_fragments)),
        edge_gateway_idx=edge_gw, edge_fragment_idx=edge_fr, edge_features=edge_feat,
        total_fragments=num_fragments, label=label,
    )


def main():
    print("=== TGAT smoke test ===\n")

    print("1. Building dummy graphs...")
    graphs = [make_dummy_graph(label=i % 2) for i in range(8)]
    print(f"   Built {len(graphs)} dummy graphs. OK.\n")

    print("2. Constructing TGATEncoder + pretrain head...")
    encoder = TGATEncoder(hidden_dim=32, temporal_dim=16, out_dim=32)
    model = TGATPretrainHead(encoder, repr_dim=32)
    print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters. OK.\n")

    print("3. Forward pass on each dummy graph...")
    for i, g in enumerate(graphs):
        logit = model(
            num_gateways=len(g.gateway_ids), num_fragments=len(g.fragment_ids),
            edge_gateway_idx=torch.from_numpy(g.edge_gateway_idx),
            edge_fragment_idx=torch.from_numpy(g.edge_fragment_idx),
            edge_features=torch.from_numpy(g.edge_features),
        )
        assert logit.shape == (), f"expected scalar logit, got shape {logit.shape}"
        print(f"   graph {i}: {len(g.edge_gateway_idx)} edges -> logit={logit.item():.4f}  OK")
    print()

    print("4. Backward pass (gradient check)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    optimizer.zero_grad()
    for g in graphs:
        logit = model(
            num_gateways=len(g.gateway_ids), num_fragments=len(g.fragment_ids),
            edge_gateway_idx=torch.from_numpy(g.edge_gateway_idx),
            edge_fragment_idx=torch.from_numpy(g.edge_fragment_idx),
            edge_features=torch.from_numpy(g.edge_features),
        )
        label = torch.tensor(float(g.label))
        loss = loss_fn(logit, label)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    print(f"   Backward pass completed, total_loss={total_loss:.4f}. OK.\n")

    print("5. Checking gradients actually flowed (not all zero)...")
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "no gradients found -- something is disconnected from the loss"
    assert max(grad_norms) > 0, "all gradients are exactly zero -- likely a bug"
    print(f"   {len(grad_norms)} parameter tensors have gradients, max norm={max(grad_norms):.4f}. OK.\n")

    print("=== ALL SMOKE TESTS PASSED ===")
    print("Safe to proceed to real training (tgat_pretrain.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
