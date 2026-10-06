"""
receiver/reconstruction/telegram_reconstructor.py

Given a TelegramContext, tries to recover each fragment's payload (via
hard_combiner, picking the best gateway observation per fragment) and,
if all fragments are recoverable, assembles the original message (via
fec/mds.py). This is called after every new observation is associated
(architecture doc: "Do I have enough information? If yes, decode.").
"""

from typing import Optional

from association.telegram_table import TelegramContext
from cooperation.hard_combiner import recovered_payload_for_fragment, ReliabilityLookup
from fec.mds import reconstruct_telegram
from logging_utils.logger import get_logger

log = get_logger("reconstruction")


def try_reconstruct(ctx: TelegramContext,
                     reliability_lookup: Optional[ReliabilityLookup] = None) -> Optional[bytes]:
    """Returns the reconstructed message bytes if possible right now,
    else None. Does not mutate ctx.finalized -- that's the caller's job
    (main.py), since only the caller knows whether it's also responsible
    for delivering/logging the result exactly once."""
    fragment_payloads: dict[int, bytes] = {}
    for frag_id in range(ctx.total_fragments):
        payload = recovered_payload_for_fragment(ctx, frag_id, reliability_lookup)
        if payload is not None:
            fragment_payloads[frag_id] = payload

    message = reconstruct_telegram(fragment_payloads, ctx.total_fragments)
    if message is not None:
        log.info(f"telegram node={ctx.node_id} id={ctx.telegram_id}: "
                 f"RECONSTRUCTED ({len(fragment_payloads)}/{ctx.total_fragments} fragments)")
    return message
