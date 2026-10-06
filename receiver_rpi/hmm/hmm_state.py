"""
receiver/hmm/hmm_state.py

A 2-state (GOOD/BAD) Hidden Markov Model tracking the interference state
of a single (node_id, gateway_id) channel over time, using the forward
algorithm (online, one observation at a time -- no need to store full
history, which matters since this runs continuously on a Pi).

One instance of ChannelHMM per (node_id, gateway_id) pair; the
HmmStateTracker below manages that dict so callers don't have to.
"""

import math
import threading

from . import hmm_config as cfg


def _gaussian_pdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        std = 0.01
    coef = 1.0 / (std * math.sqrt(2 * math.pi))
    exponent = -((x - mean) ** 2) / (2 * std * std)
    return coef * math.exp(exponent)


class ChannelHMM:
    """Forward-algorithm state tracker for one (node, gateway) channel."""

    def __init__(self):
        self.p_good = cfg.INITIAL_P_GOOD
        self.p_bad = 1.0 - self.p_good
        self.num_observations = 0

    def update(self, snr_db: float) -> float:
        """Feed a new SNR observation, update the state belief, and
        return the updated P(GOOD)."""
        # Predict step: propagate belief through the transition model.
        pred_good = self.p_good * cfg.P_GOOD_TO_GOOD + self.p_bad * cfg.P_BAD_TO_GOOD
        pred_bad = self.p_good * cfg.P_GOOD_TO_BAD + self.p_bad * cfg.P_BAD_TO_BAD

        # Update step: weight by how well this SNR matches each state's
        # emission distribution.
        like_good = _gaussian_pdf(snr_db, cfg.GOOD_SNR_MEAN_DB, cfg.GOOD_SNR_STD_DB)
        like_bad = _gaussian_pdf(snr_db, cfg.BAD_SNR_MEAN_DB, cfg.BAD_SNR_STD_DB)

        unnorm_good = pred_good * like_good
        unnorm_bad = pred_bad * like_bad
        total = unnorm_good + unnorm_bad

        if total > 0:
            self.p_good = unnorm_good / total
            self.p_bad = unnorm_bad / total
        # else: degenerate observation (shouldn't happen with valid SNR),
        # keep previous belief unchanged rather than dividing by zero.

        self.num_observations += 1
        return self.p_good


class HmmStateTracker:
    """Manages one ChannelHMM per (node_id, gateway_id) pair. Thread-safe,
    since observations arrive from multiple gateway server threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._channels: dict[tuple[int, int], ChannelHMM] = {}

    def update(self, node_id: int, gateway_id: int, snr_db: float) -> float:
        """Feed an observation for (node_id, gateway_id), return updated P(GOOD)."""
        key = (node_id, gateway_id)
        with self._lock:
            hmm = self._channels.setdefault(key, ChannelHMM())
            return hmm.update(snr_db)

    def p_good(self, node_id: int, gateway_id: int) -> float:
        """Current P(GOOD) for a channel without feeding a new observation.
        Returns the prior if this channel has never been observed."""
        key = (node_id, gateway_id)
        with self._lock:
            hmm = self._channels.get(key)
            return hmm.p_good if hmm is not None else cfg.INITIAL_P_GOOD
