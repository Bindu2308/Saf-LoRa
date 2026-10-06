"""
receiver/ml/gradient_boosting.py

Wraps a scikit-learn GradientBoostingClassifier predicting P(this gateway
fully receives this telegram | features). Per spec section 43 ("a failure
in ML must NOT stop the PHY/network receiver"), every prediction call is
guarded: if no model is trained yet, or loading/predicting fails for any
reason, callers get a neutral fallback score (0.5) rather than an
exception -- the pipeline keeps running on hard_combiner's existing
confidence-based weighting either way.
"""

import os
import pickle
import threading

from ml.feature_extractor import ReliabilityFeatures
from logging_utils.logger import get_logger

log = get_logger("gradient_boosting")

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gradient_boosting.pkl")


class GradientBoostingReliability:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        with self._lock:
            if os.path.exists(self.model_path):
                try:
                    with open(self.model_path, "rb") as f:
                        self._model = pickle.load(f)
                    log.info(f"loaded Gradient Boosting model from {self.model_path}")
                except Exception as e:
                    log.warn(f"failed to load GB model ({e}), falling back to neutral scores")
                    self._model = None
            else:
                log.info(f"no trained GB model at {self.model_path} yet -- using neutral scores until trained")
                self._model = None

    def reload(self):
        """Call this after train_gb.py produces a new model file, to pick
        it up without restarting the receiver process."""
        self._load()

    def predict_reliability(self, features: ReliabilityFeatures) -> float:
        """Returns P(gateway fully receives this telegram), in [0, 1].
        Falls back to 0.5 (neutral -- no information) on any failure."""
        with self._lock:
            model = self._model
        if model is None:
            return 0.5
        try:
            proba = model.predict_proba([features.as_vector()])[0]
            # class 1 = "fully received"; sklearn orders classes ascending,
            # so proba[1] is P(class=1) as long as both 0 and 1 appear in
            # training data (train_gb.py's job to ensure that).
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as e:
            log.warn(f"GB prediction failed ({e}), falling back to neutral score")
            return 0.5
