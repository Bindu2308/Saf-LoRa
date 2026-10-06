"""
receiver/ml/feature_extractor.py

Builds the feature vector for the Gradient Boosting reliability model
(architecture doc's "Gradient Boosting feature vector" deliverable).

Given what real per-fragment data actually contains -- hard-decision
packets with RSSI/SNR, no soft LLRs (see cooperation/soft_combiner.py's
docstring for why) -- features are built at the (gateway, node, telegram)
granularity: "how reliable was gateway G's reception of node N's traffic
around this telegram", not per-bit.
"""

from dataclasses import dataclass


@dataclass
class ReliabilityFeatures:
    avg_snr_db: float
    min_snr_db: float
    avg_rssi_db: float
    hmm_p_good: float
    historical_success_rate: float  # this gateway's recent completion rate for this node

    def as_vector(self) -> list[float]:
        """Fixed feature order -- must match training and inference exactly."""
        return [self.avg_snr_db, self.min_snr_db, self.avg_rssi_db,
                self.hmm_p_good, self.historical_success_rate]

    @staticmethod
    def feature_names() -> list[str]:
        return ["avg_snr_db", "min_snr_db", "avg_rssi_db", "hmm_p_good", "historical_success_rate"]


def extract_features(snr_values: list[float], rssi_values: list[float],
                      hmm_p_good: float, historical_success_rate: float) -> ReliabilityFeatures:
    """snr_values/rssi_values: per-fragment readings from this gateway for
    this telegram (usually 1-3 values, one per fragment received)."""
    avg_snr = sum(snr_values) / len(snr_values) if snr_values else 0.0
    min_snr = min(snr_values) if snr_values else 0.0
    avg_rssi = sum(rssi_values) / len(rssi_values) if rssi_values else 0.0
    return ReliabilityFeatures(
        avg_snr_db=avg_snr,
        min_snr_db=min_snr,
        avg_rssi_db=avg_rssi,
        hmm_p_good=hmm_p_good,
        historical_success_rate=historical_success_rate,
    )
