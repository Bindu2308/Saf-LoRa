"""
receiver/association/duplicate_detector.py

The core dedup logic (same gateway + node + telegram + fragment = duplicate,
different gateway = diversity) is enforced structurally by
TelegramContext.add_observation's dict-keyed-by-gateway_id design -- a
duplicate simply can't be inserted as a second entry. This module exists
as the explicit "is this a duplicate?" check for callers (e.g. metrics)
that want to classify an observation without mutating any state.
"""

from network.protocol import GatewayObservation
from association.telegram_table import TelegramContext


def is_duplicate(ctx: TelegramContext, obs: GatewayObservation) -> bool:
    """True if `ctx` already has an observation from this exact gateway
    for this exact fragment (i.e. inserting `obs` would just replace it,
    not add new diversity)."""
    per_gateway = ctx.fragments.get(obs.fragment_id)
    if per_gateway is None:
        return False
    return obs.gateway_id in per_gateway
