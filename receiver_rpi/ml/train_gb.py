"""
receiver/ml/train_gb.py

Offline training script. Run this periodically (e.g. after collecting a
few hundred telegrams' worth of real data) to (re)train the reliability
model from receiver/data/training/reliability_examples.csv.

Usage:
    cd receiver
    python3 -m ml.train_gb
    python3 -m ml.train_gb --csv path/to/other.csv --min-examples 50
"""

import argparse
import csv
import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml.feature_extractor import ReliabilityFeatures
from ml.gradient_boosting import DEFAULT_MODEL_PATH


def load_dataset(csv_path: str):
    X, y = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ReliabilityFeatures.feature_names() + ["label"], \
            f"CSV header doesn't match expected feature order: {header}"
        for row in reader:
            *features, label = row
            X.append([float(v) for v in features])
            y.append(int(label))
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train the ISAC-LoRa Gradient Boosting reliability model")
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "..", "data", "training", "reliability_examples.csv"))
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--min-examples", type=int, default=30,
                         help="refuse to train on fewer examples than this -- an undertrained model is worse than no model (falls back to neutral 0.5)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"No training data found at {args.csv} yet. Run the receiver for a while first to accumulate examples.")
        return

    X, y = load_dataset(args.csv)
    print(f"Loaded {len(X)} examples ({sum(y)} positive / {len(y) - sum(y)} negative)")

    if len(X) < args.min_examples:
        print(f"Only {len(X)} examples, need at least {args.min_examples} -- not training yet. "
              f"Keep the receiver running longer to collect more real telegrams.")
        return

    if len(set(y)) < 2:
        print("Training data has only one class (all successes or all failures so far) -- "
              "GradientBoostingClassifier needs both to learn anything meaningful. Waiting for more diverse data.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["partial", "full"]))

    print("\nFeature importances:")
    for name, importance in zip(ReliabilityFeatures.feature_names(), model.feature_importances_):
        print(f"  {name:25s} {importance:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model to {args.out}")
    print("Restart the receiver, or call GradientBoostingReliability.reload(), to pick it up.")


if __name__ == "__main__":
    main()
