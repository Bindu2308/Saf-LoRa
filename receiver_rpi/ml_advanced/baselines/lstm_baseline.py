"""
receiver/ml_advanced/baselines/lstm_baseline.py

LSTM baseline: treats each channel's SNR/RSSI readings as a time
sequence and predicts next-observation reliability -- the "sequence but
no graph structure" comparison point (doc section 20: LSTM handles
RSSI(t-5..t) well, but doesn't naturally represent gateway<->fragment
relationships the way a graph does).

Usage (Colab):
    python3 baselines/lstm_baseline.py --csv reliability_examples.csv --epochs 30
"""

import argparse
import csv
import random
import torch
import torch.nn as nn
import numpy as np

FEATURE_NAMES = ["avg_snr_db", "min_snr_db", "avg_rssi_db", "hmm_p_good", "historical_success_rate"]


class ReliabilityLSTM(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_NAMES), hidden_dim: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def load_sequences(csv_path: str, window: int = 5):
    """Loads rows IN ORDER (file is append-only, so row order ==
    chronological order) and builds sliding windows -> predict the label
    of the row right after the window. Simplification: treats all rows
    as one sequence rather than splitting per-channel, since with only
    ~600 real examples, per-channel splitting would leave too little
    data to train an LSTM meaningfully -- documented, not hidden."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == FEATURE_NAMES + ["label"]
        rows = [[float(v) for v in row[:-1]] + [int(row[-1])] for row in reader]

    X, y = [], []
    for i in range(len(rows) - window):
        window_feats = [r[:-1] for r in rows[i:i + window]]
        next_label = rows[i + window][-1]
        X.append(window_feats)
        y.append(next_label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="LSTM baseline for ISAC-LoRa reliability prediction")
    parser.add_argument("--csv", default="reliability_examples.csv")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="lstm_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y = load_sequences(args.csv, window=args.window)
    print(f"Built {len(X)} sequences of length {args.window}")
    if len(X) < 30:
        print("Too few sequences to train meaningfully -- collect more data first.")
        return

    split = int(len(X) * 0.8)
    idx = np.random.permutation(len(X))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    X_val, y_val = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])

    model = ReliabilityLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_pred == y_val).float().mean().item()

        print(f"epoch {epoch+1}/{args.epochs}  train_loss={loss.item():.4f}  val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "window": args.window}, args.out)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Saved best model to {args.out}")


if __name__ == "__main__":
    main()
