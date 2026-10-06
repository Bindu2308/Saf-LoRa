"""
receiver/ml_advanced/graph_logger.py

The existing ml/training_logger.py logs ONE aggregated row per
(gateway, telegram) -- perfect for Gradient Boosting, but too coarse for
the TGAT graph model, which needs one row per gateway<->fragment EDGE
(so a telegram with 2 fragments heard by 2 gateways = up to 4 rows, not 1).

This logger captures that finer granularity: every individual observation,
with its timing, duplicate status, and telegram progress at that moment --
then back-fills the eventual outcome (did the telegram reconstruct) once
the telegram finalizes, since that's the label needed for supervised
TGAT pretraining.

Integration: only 3 lines need to be added to receiver/main.py (see the
comment at the bottom of this file) -- this does NOT replace
training_logger.py, it runs alongside it.
"""

import csv
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training", "graph_observations.csv")

FIELDNAMES = [
    "telegram_id", "node_id", "gateway_id", "fragment_id",
    "rssi_db", "snr_db", "timestamp_ms", "time_since_telegram_start_ms",
    "is_duplicate", "fragments_seen_before_this", "total_fragments",
    "hmm_p_good_at_time", "telegram_reconstructed",  # label, filled in at finalize
]


@dataclass
class _PendingTelegram:
    first_seen_ms: float
    rows: list = field(default_factory=list)  # each a dict matching FIELDNAMES, minus the label
    seen_gateway_fragment: set = field(default_factory=set)  # {(gateway_id, fragment_id)} for duplicate detection


class GraphLogger:
    def __init__(self, path: str = DEFAULT_LOG_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._pending: dict[tuple, _PendingTelegram] = {}  # key = (node_id, telegram_id)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def record_observation(self, node_id: int, telegram_id: int, gateway_id: int,
                            fragment_id: int, total_fragments: int,
                            rssi_db: float, snr_db: float, hmm_p_good: float):
        """Call this from _on_observation, for every validated observation
        (before we know whether the telegram will ultimately succeed)."""
        key = (node_id, telegram_id)
        now_ms = time.time() * 1000.0

        with self._lock:
            pending = self._pending.get(key)
            if pending is None:
                pending = _PendingTelegram(first_seen_ms=now_ms)
                self._pending[key] = pending

            gw_frag = (gateway_id, fragment_id)
            is_duplicate = gw_frag in pending.seen_gateway_fragment
            pending.seen_gateway_fragment.add(gw_frag)

            row = {
                "telegram_id": telegram_id,
                "node_id": node_id,
                "gateway_id": gateway_id,
                "fragment_id": fragment_id,
                "rssi_db": rssi_db,
                "snr_db": snr_db,
                "timestamp_ms": now_ms,
                "time_since_telegram_start_ms": now_ms - pending.first_seen_ms,
                "is_duplicate": int(is_duplicate),
                "fragments_seen_before_this": len(pending.seen_gateway_fragment) - (1 if not is_duplicate else 0),
                "total_fragments": total_fragments,
                "hmm_p_good_at_time": hmm_p_good,
            }
            pending.rows.append(row)

    def finalize_telegram(self, node_id: int, telegram_id: int, reconstructed: bool):
        """Call this once, when a telegram is finalized (reconstructed or
        timed out) -- writes out every observation row for this telegram
        with the now-known label."""
        key = (node_id, telegram_id)
        with self._lock:
            pending = self._pending.pop(key, None)
            if pending is None or not pending.rows:
                return
            with open(self.path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                for row in pending.rows:
                    row["telegram_reconstructed"] = int(reconstructed)
                    writer.writerow(row)

    def drop_stale(self, max_age_ms: float = 60000.0):
        """Safety valve: if a telegram somehow never gets finalized (bug
        elsewhere), don't leak memory forever. Call this periodically."""
        now_ms = time.time() * 1000.0
        with self._lock:
            stale_keys = [k for k, v in self._pending.items() if now_ms - v.first_seen_ms > max_age_ms]
            for k in stale_keys:
                del self._pending[k]


# ============================================================
# Integration into receiver/main.py (add these lines by hand,
# don't auto-patch -- main.py has been edited enough times this
# session that a manual, visible diff is safer than another
# automated rewrite):
#
# 1. In Receiver.__init__, alongside self.training_logger:
#      from ml_advanced.graph_logger import GraphLogger
#      self.graph_logger = GraphLogger()
#
# 2. In _on_observation, right after the HMM update (p_good is already
#    computed there):
#      self.graph_logger.record_observation(
#          obs.node_id, obs.telegram_id, obs.gateway_id, obs.fragment_id,
#          obs.total_fragments, obs.received_power_db, obs.demod_confidence,
#          p_good,
#      )
#
# 3. In _log_training_examples (called at both successful finalize and
#    timeout), add:
#      self.graph_logger.finalize_telegram(ctx.node_id, ctx.telegram_id,
#                                            reconstructed=ctx.finalized and <was it a success>)
#    Simplest: pass an explicit `success: bool` argument into
#    _log_training_examples from both call sites (main reconstruction path
#    passes True, timeout path passes False) and forward it here.
# ============================================================
