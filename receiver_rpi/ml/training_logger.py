"""
receiver/ml/training_logger.py

Appends one training example per (gateway, telegram) at finalize time.
Label comes for free from data we already have: 1 if that gateway
received ALL of that telegram's fragments (fully reliable that cycle), 0
if it received only some (partial -- channel was degraded for that
gateway during that telegram). No extra hardware instrumentation needed.

CSV format (append-only, safe for concurrent writes from one process
since we hold a lock): matches ReliabilityFeatures.feature_names() order,
plus the label as the last column.
"""

import csv
import os
import threading

from ml.feature_extractor import ReliabilityFeatures

DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training", "reliability_examples.csv")


class TrainingLogger:
    def __init__(self, path: str = DEFAULT_LOG_PATH):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(ReliabilityFeatures.feature_names() + ["label"])

    def log_example(self, features: ReliabilityFeatures, label: int):
        with self._lock:
            with open(self.path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(features.as_vector() + [label])
