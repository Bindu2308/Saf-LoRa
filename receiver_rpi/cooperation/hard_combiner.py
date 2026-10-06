"""
receiver/cooperation/hard_combiner.py

This is the combiner actually in use right now. Per architecture doc
section 13, hard combining is the correct fallback when soft (per-bit LLR)
information isn't available -- and today it genuinely isn't: Gateway 1/2
currently forward a single scalar avg_symbol_confidence per fragment, not
a per-bit LLR array (that's a Phase 8 addition requiring gateway changes,
see soft_combiner.py). So hard combining is the real, working path.

For each fragment_id in a telegram, this picks the single best observation
across all gateways that reported it (by confidence.weight_of, optionally
scaled by an HMM/GB reliability signal), and returns its payload bytes if
that best observation's CRC passed.
"""

from typing import Optional, Callable

from network.protocol import GatewayObservation
from association.telegram_table import TelegramContext
from cooperation.confidence import weight_of, combined_weight_of

# Signature: (GatewayObservation) -> reliability multiplier in [0, ~2].
# 1.0 is neutral (matches plain weight_of() behavior). main.py wires this
# to HMM P(GOOD) * Gradient Boosting predicted reliability, computed from
# THIS observation's actual SNR/RSSI (not a placeholder) once trained;
# left as None here so this module has zero dependency on the ML layer
# and keeps working standalone if that layer is ever removed or fails.
ReliabilityLookup = Callable[[GatewayObservation], float]


def best_observation_for_fragment(ctx: TelegramContext, fragment_id: int,
                                   reliability_lookup: Optional[ReliabilityLookup] = None) -> Optional[GatewayObservation]:
    # Snapshot under the lock -- ctx.fragments can be mutated concurrently
    # by another gateway's connection thread calling add_observation()
    # (spec: "dictionary changed size during iteration" is exactly this
    # race). Copying to a list while holding the lock, then releasing
    # before scoring/max(), keeps the lock held only briefly.
    with ctx.fragments_lock:
        per_gateway = ctx.fragments.get(fragment_id)
        if not per_gateway:
            return None
        candidates = list(per_gateway.values())

    if reliability_lookup is None:
        return max(candidates, key=weight_of)

    def scored(obs: GatewayObservation) -> float:
        multiplier = reliability_lookup(obs)
        return combined_weight_of(obs, multiplier)

    return max(candidates, key=scored)


def recovered_payload_for_fragment(ctx: TelegramContext, fragment_id: int,
                                    reliability_lookup: Optional[ReliabilityLookup] = None) -> Optional[bytes]:
    """Returns the fragment's payload bytes if the best available
    observation for it has a passing CRC, else None (not yet recoverable)."""
    best = best_observation_for_fragment(ctx, fragment_id, reliability_lookup)
    if best is None or not best.crc_ok:
        return None
    return bytes(best.payload)
