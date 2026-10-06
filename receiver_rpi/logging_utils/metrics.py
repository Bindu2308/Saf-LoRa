"""
receiver/logging_utils/metrics.py

Tracks the counters the architecture doc section 18/19 calls out as
useful for research results: observations per gateway, fragments
recovered, telegrams reconstructed vs timed-out, gateway diversity.
"""

import threading
from dataclasses import dataclass, field


@dataclass
class Metrics:
    observations_by_gateway: dict[int, int] = field(default_factory=dict)
    fragments_recovered: int = 0
    telegrams_reconstructed: int = 0
    telegrams_timed_out_incomplete: int = 0
    fragments_seen_by_both_gateways: int = 0  # gateway diversity indicator

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_observation(self, gateway_id: int):
        with self._lock:
            self.observations_by_gateway[gateway_id] = self.observations_by_gateway.get(gateway_id, 0) + 1

    def record_fragment_recovered(self):
        with self._lock:
            self.fragments_recovered += 1

    def record_telegram_reconstructed(self):
        with self._lock:
            self.telegrams_reconstructed += 1

    def record_telegram_timed_out(self):
        with self._lock:
            self.telegrams_timed_out_incomplete += 1

    def record_fragment_diversity(self, num_gateways_reporting: int):
        with self._lock:
            if num_gateways_reporting >= 2:
                self.fragments_seen_by_both_gateways += 1

    def summary(self) -> str:
        with self._lock:
            obs_line = ", ".join(f"GW{k}={v}" for k, v in sorted(self.observations_by_gateway.items()))
            return (
                f"observations[{obs_line}] "
                f"fragments_recovered={self.fragments_recovered} "
                f"telegrams_reconstructed={self.telegrams_reconstructed} "
                f"telegrams_timed_out={self.telegrams_timed_out_incomplete} "
                f"fragments_seen_by_both_gateways={self.fragments_seen_by_both_gateways}"
            )
