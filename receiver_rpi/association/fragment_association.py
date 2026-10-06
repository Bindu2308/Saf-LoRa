"""
receiver/association/fragment_association.py

Given a validated GatewayObservation, finds (or creates) the right
TelegramContext -- keyed by (node_id, telegram_id) per architecture doc
section 8 -- and inserts the observation. This is the module that answers
"these two observations belong to the same original transmission."
"""

from network.protocol import GatewayObservation
from association.telegram_table import TelegramTable, TelegramContext
from logging_utils.logger import get_logger

log = get_logger("association")


class FragmentAssociation:
    def __init__(self, table: TelegramTable):
        self.table = table

    def associate(self, obs: GatewayObservation) -> TelegramContext:
        ctx = self.table.get_or_create(obs.node_id, obs.telegram_id, obs.total_fragments)

        if ctx.total_fragments != obs.total_fragments:
            log.warn(f"telegram node={obs.node_id} id={obs.telegram_id}: "
                     f"total_fragments mismatch (context has {ctx.total_fragments}, "
                     f"observation says {obs.total_fragments}) -- keeping original")

        is_new = ctx.add_observation(obs)
        if is_new:
            log.info(f"GW{obs.gateway_id}: node={obs.node_id} telegram={obs.telegram_id} "
                     f"fragment={obs.fragment_id}/{obs.total_fragments} crc_ok={obs.crc_ok} "
                     f"conf={obs.demod_confidence:.2f} (new)")
        else:
            log.debug(f"GW{obs.gateway_id}: node={obs.node_id} telegram={obs.telegram_id} "
                      f"fragment={obs.fragment_id} -- duplicate observation from same gateway, updated")

        return ctx
