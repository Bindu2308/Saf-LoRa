"""
receiver/ml_advanced/tgat_pretrain.py

Stage 1 (per the doc's recommended two-stage training): supervised
pretraining of TGATEncoder on real logged data, predicting "will this
telegram's observations lead to successful reconstruction". This gives
PPO a useful starting representation instead of learning from scratch
via pure RL, which is much slower and less stable.

Usage (in Colab):
    python3 tgat_pretrain.py --csv graph_observations.csv --epochs 50
"""

import argparse
import random
import torch
import numpy as np

from graph_dataset import build_telegram_graphs, normalize_edge_features
from tgat_model import TGATEncoder, TGATPretrainHead


def graph_to_tensors(g):
    return (
        len(g.gateway_ids), len(g.fragment_ids),
        torch.from_numpy(g.edge_gateway_idx),
        torch.from_numpy(g.edge_fragment_idx),
        torch.from_numpy(g.edge_features),
    )


def evaluate(model, graphs):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for g in graphs:
            ng, nf, egi, efi, ef = graph_to_tensors(g)
            logit = model(ng, nf, egi, efi, ef)
            pred = int(torch.sigmoid(logit).item() > 0.5)
            correct += int(pred == g.label)
            total += 1
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Pretrain TGAT encoder on real ISAC-LoRa telegram data")
    parser.add_argument("--csv", default="graph_observations.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--out", default="tgat_encoder.pt")
    parser.add_argument("--min-graphs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading graphs from {args.csv} ...")
    graphs = build_telegram_graphs(args.csv)
    print(f"Loaded {len(graphs)} telegram graphs "
          f"({sum(g.label for g in graphs)} reconstructed / {len(graphs) - sum(g.label for g in graphs)} not)")

    if len(graphs) < args.min_graphs:
        print(f"Only {len(graphs)} graphs, need at least {args.min_graphs} -- "
              f"let the receiver run longer to collect more real telegrams before pretraining.")
        return

    if len(set(g.label for g in graphs)) < 2:
        print("All graphs have the same label -- need both successes and failures to learn anything. "
              "Waiting for more diverse data (e.g. run under weaker signal conditions too).")
        return

    graphs, norm_stats = normalize_edge_features(graphs)
    print(f"Normalization stats (SAVE these -- inference.py needs them): {norm_stats}")

    random.shuffle(graphs)
    split_idx = int(len(graphs) * (1 - args.val_split))
    train_graphs, val_graphs = graphs[:split_idx], graphs[split_idx:]
    print(f"Train: {len(train_graphs)}  Val: {len(val_graphs)}")

    encoder = TGATEncoder(hidden_dim=32, temporal_dim=16, out_dim=32)
    model = TGATPretrainHead(encoder, repr_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_graphs)
        total_loss = 0.0
        for g in train_graphs:
            ng, nf, egi, efi, ef = graph_to_tensors(g)
            optimizer.zero_grad()
            logit = model(ng, nf, egi, efi, ef)
            loss = loss_fn(logit, torch.tensor(float(g.label)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_graphs)
        val_acc = evaluate(model, val_graphs)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "norm_stats": norm_stats,
                "hidden_dim": 32, "temporal_dim": 16, "out_dim": 32,
            }, args.out)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Saved best encoder checkpoint to {args.out}")
    print("Next: use this checkpoint to initialize PPO training (ppo_train.py --tgat-checkpoint ...)")


if __name__ == "__main__":
    main()
