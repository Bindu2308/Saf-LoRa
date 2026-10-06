"""
receiver/observation/validator.py

Sanity-checks a parsed GatewayObservation before it's allowed into the
association/reconstruction pipeline (architecture doc section 7). This is
deliberately conservative -- a bad observation here could otherwise corrupt
a telegram's fragment table.
"""

from network.protocol import GatewayObservation
from config.receiver_config import FRAGMENT_PAYLOAD_SIZE


class ValidationError(ValueError):
    pass


def validate(obs: GatewayObservation) -> None:
    """Raises ValidationError with a specific reason if invalid. Callers
    should catch this, log it, and drop the observation -- never let a bad
    observation propagate into the telegram table."""
    if obs.gateway_id not in (1, 2, 3):
        raise ValidationError(f"unexpected gateway_id={obs.gateway_id}")
    if obs.total_fragments == 0:
        raise ValidationError("total_fragments == 0")
    if obs.fragment_id >= obs.total_fragments:
        raise ValidationError(f"fragment_id={obs.fragment_id} >= total_fragments={obs.total_fragments}")
    if len(obs.payload) != FRAGMENT_PAYLOAD_SIZE:
        raise ValidationError(f"payload length {len(obs.payload)} != {FRAGMENT_PAYLOAD_SIZE}")
    # node_id == 0 is not inherently invalid (nodes could be zero-indexed),
    # so it's not rejected here -- flag if your node numbering starts at 1
    # and 0 should never appear.
