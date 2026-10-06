"""
receiver/ml_advanced/inference.py

RPi4-side inference: loads the TGAT encoder + PPO policy trained in
Colab, and provides a reliability_lookup-compatible function (same
interface hard_combiner.py already expects, per main.py's
_reliability_lookup) so this can be swapped in for the current HMM+GB
system with minimal changes to main.py.

The Pi does INFERENCE ONLY -- no training happens here. Per spec section
43 ("ML failure must not stop the receiver"), every call is guarded:
if the checkpoint files aren't present, or inference fails for any
reason, this falls back to a neutral multiplier (1.0) rather than
raising -- matching the exact fallback philosophy of
ml/gradient_boosting.py.

REQUIRES: torch and stable-baselines3 installed on the RPi4
(pip3 install torch stable-baselines3 --break-system-packages).
Torch inference-only on a Pi4 CPU is fine -- it's training that needs a
GPU, not inference on this small a model.
"""

import os
import threading

from logging_utils.logger import get_logger

log = get_logger("tgat_ppo_inference")

DEFAULT_TGAT_PATH = os.path.join(os.path.dirname(__file__), "models", "tgat_encoder.pt")
DEFAULT_PPO_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_policy.zip")


class TGATPPOInference:
    def __init__(self, tgat_path: str = DEFAULT_TGAT_PATH, ppo_path: str = DEFAULT_PPO_PATH):
        self.tgat_path = tgat_path
        self.ppo_path = ppo_path
        self._lock = threading.Lock()
        self._encoder = None
        self._policy = None
        self._norm_stats = None
        self._load()

    def _load(self):
        with self._lock:
            if not (os.path.exists(self.tgat_path) and os.path.exists(self.ppo_path)):
                log.info(f"no trained TGAT/PPO checkpoints found at {self.tgat_path} / {self.ppo_path} "
                         f"yet -- using neutral scores until trained in Colab and copied here")
                return
            try:
                import torch
                from stable_baselines3 import PPO
                from tgat_model import TGATEncoder

                ckpt = torch.load(self.tgat_path, map_location="cpu")
                self._encoder = TGATEncoder(hidden_dim=ckpt["hidden_dim"],
                                             temporal_dim=ckpt["temporal_dim"],
                                             out_dim=ckpt["out_dim"])
                self._encoder.load_state_dict(ckpt["encoder_state_dict"])
                self._encoder.eval()
                self._norm_stats = ckpt["norm_stats"]

                self._policy = PPO.load(self.ppo_path)
                log.info(f"loaded TGAT+PPO from {self.tgat_path} / {self.ppo_path}")
            except Exception as e:
                log.warn(f"failed to load TGAT/PPO ({e}), falling back to neutral scores")
                self._encoder = None
                self._policy = None

    def reload(self):
        self._load()

    def action_for_telegram(self, edge_gateway_idx, edge_fragment_idx, edge_features,
                             num_gateways: int, num_fragments: int,
                             missing_frac: float, progress: float) -> int:
        """Returns the PPO action: 0=GW1 only, 1=GW2 only, 2=both.
        Falls back to 2 (both -- the safe, maximally-cooperative default)
        on any failure, matching this project's established fallback
        philosophy."""
        with self._lock:
            encoder, policy = self._encoder, self._policy
        if encoder is None or policy is None:
            return 2  # both -- safe default, equivalent to current non-ML cooperative behavior

        try:
            import torch
            import numpy as np

            with torch.no_grad():
                h = encoder(num_gateways, num_fragments, edge_gateway_idx, edge_fragment_idx, edge_features)
            extra = np.array([missing_frac, progress, len(edge_gateway_idx) / 10.0], dtype='float32')
            obs = np.concatenate([h.numpy(), extra]).astype('float32')

            action, _ = policy.predict(obs, deterministic=True)
            return int(action)
        except Exception as e:
            log.warn(f"TGAT+PPO inference failed ({e}), falling back to 'both' action")
            return 2
