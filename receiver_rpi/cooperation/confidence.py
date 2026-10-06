"""
receiver/cooperation/confidence.py

Simple, deterministic reliability weight for an observation -- no ML.
This is what hard_combiner.py uses to pick between multiple gateways'
reports of the same fragment (architecture doc section 12: "weights can
initially be based on measurable RF quality").
"""

from network.protocol import GatewayObservation


def weight_of(obs: GatewayObservation) -> float:
    """Higher is more trustworthy. CRC-passing observations are weighted
    far above CRC-failing ones (a clean decode beats any amount of raw
    confidence from a failed one); within the same crc_ok bucket, break
    ties by demod_confidence."""
    base = 1000.0 if obs.crc_ok else 0.0
    return base + obs.demod_confidence


def combined_weight_of(obs: GatewayObservation, reliability_multiplier: float = 1.0) -> float:
    """Same as weight_of(), scaled by an external reliability signal (e.g.
    HMM P(GOOD) * Gradient Boosting predicted reliability for this
    gateway/node pair right now). reliability_multiplier defaults to 1.0
    (no change), so callers that don't have HMM/GB wired up yet get
    identical behavior to weight_of() -- this is what keeps the ML layer
    optional per spec section 43 (ML failure must not stop the pipeline)."""
    return weight_of(obs) * max(reliability_multiplier, 0.0)
