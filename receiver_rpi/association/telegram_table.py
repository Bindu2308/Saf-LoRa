"""
receiver/association/telegram_table.py

One TelegramContext per (node_id, telegram_id), per architecture doc
sections 8-11. Crucially: fragments[fragment_id] is a dict keyed by
gateway_id, NOT a single value -- multiple gateways' observations of the
SAME fragment are preserved side by side (the whole point of cooperative
reception), never collapsed into "fragment received: yes/no".
"""

import time
import threading
from dataclasses import dataclass, field

from network.protocol import GatewayObservation


@dataclass
class TelegramContext:
    node_id: int
    telegram_id: int
    total_fragments: int
    created_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    # fragments[fragment_id][gateway_id] = GatewayObservation
    fragments: dict[int, dict[int, GatewayObservation]] = field(default_factory=dict)
    finalized: bool = False  # set once reconstructed or timed out, prevents double-processing
    finalize_lock: threading.Lock = field(default_factory=threading.Lock)  # guards the check-and-set of finalized across concurrent gateway threads
    fragments_lock: threading.Lock = field(default_factory=threading.Lock)  # guards all reads/writes of the fragments dict itself, since two gateway threads can touch the same telegram concurrently

    def add_observation(self, obs: GatewayObservation) -> bool:
        """Returns True if this was a genuinely new (gateway, fragment)
        pair, False if it replaced an existing duplicate observation from
        the same gateway for the same fragment (spec section 10: same
        gateway sending the same fragment twice is a duplicate, NOT new
        diversity -- but two DIFFERENT gateways reporting the same
        fragment IS diversity and both are kept, per section 11)."""
        self.last_update = time.time()
        with self.fragments_lock:
            per_gateway = self.fragments.setdefault(obs.fragment_id, {})
            is_new = obs.gateway_id not in per_gateway
            # Always keep the latest observation for a given (fragment, gateway)
            # pair -- a retransmitted/duplicate report might have better SNR.
            per_gateway[obs.gateway_id] = obs
        return is_new

    def is_expired(self, timeout_ms: int) -> bool:
        return (time.time() - self.last_update) * 1000.0 > timeout_ms

    def fragment_ids_with_any_observation(self) -> set[int]:
        return set(self.fragments.keys())

    def summary(self) -> str:
        # Snapshot under the lock before iterating -- same race as
        # add_observation()/best_observation_for_fragment(): this can be
        # called (e.g. from the timeout sweep thread) while another
        # gateway thread is concurrently adding a new observation.
        with self.fragments_lock:
            snapshot = {frag_id: dict(gw_map) for frag_id, gw_map in self.fragments.items()}

        lines = [f"Telegram node={self.node_id} id={self.telegram_id} total_fragments={self.total_fragments}"]
        for frag_id in sorted(snapshot.keys()):
            gw_marks = []
            for gw_id, obs in sorted(snapshot[frag_id].items()):
                mark = "OK" if obs.crc_ok else "FAIL"
                gw_marks.append(f"GW{gw_id}:{mark}")
            lines.append(f"  fragment {frag_id}: {', '.join(gw_marks)}")
        return "\n".join(lines)


class TelegramTable:
    def __init__(self):
        self._lock = threading.Lock()
        self._table: dict[tuple[int, int], TelegramContext] = {}

    def get_or_create(self, node_id: int, telegram_id: int, total_fragments: int) -> TelegramContext:
        key = (node_id, telegram_id)
        with self._lock:
            ctx = self._table.get(key)
            if ctx is None:
                ctx = TelegramContext(node_id=node_id, telegram_id=telegram_id, total_fragments=total_fragments)
                self._table[key] = ctx
            return ctx

    def all_contexts(self) -> list[TelegramContext]:
        with self._lock:
            return list(self._table.values())

    def remove(self, node_id: int, telegram_id: int):
        with self._lock:
            self._table.pop((node_id, telegram_id), None)
