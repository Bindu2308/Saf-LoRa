"""
receiver/ml_advanced/baselines/xgboost_baseline.py

Same task and same feature set as the working ml/gradient_boosting.py
(sklearn), but using XGBoost -- a genuinely different GB implementation,
useful as a "does the specific library matter" comparison point. Trained
on the SAME reliability_examples.csv the working GB model uses.

Usage (Colab):
    python3 baselines/xgboost_baseline.py --csv reliability_examples.csv
"""

import argparse
import csv
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FEATURE_NAMES = ["avg_snr_db", "min_snr_db", "avg_rssi_db", "hmm_p_good", "historical_success_rate"]


def load_dataset(csv_path: str):
    X, y = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == FEATURE_NAMES + ["label"], f"unexpected CSV header: {header}"
        for row in reader:
            *features, label = row
            X.append([float(v) for v in features])
            y.append(int(label))
    return X, y


def main():
    parser = argparse.ArgumentParser(description="XGBoost baseline for ISAC-LoRa reliability prediction")
    parser.add_argument("--csv", default="reliability_examples.csv")
    parser.add_argument("--out", default="xgboost_model.json")
    args = parser.parse_args()

    X, y = load_dataset(args.csv)
    print(f"Loaded {len(X)} examples ({sum(y)} positive / {len(y) - sum(y)} negative)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
                               eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=["partial", "full"]))

    print("\nFeature importances:")
    for name, importance in zip(FEATURE_NAMES, model.feature_importances_):
        print(f"  {name:25s} {importance:.3f}")

    model.save_model(args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
