"""
receiver/network/connection_manager.py

Tracks per-gateway connection status, exactly as described in the
architecture doc (GatewayStatus: connected / last_seen / observations_received).
Thread-safe since gateway_server.py calls this from per-connection threads.
"""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class GatewayStatus:
    gateway_id: int
    connected: bool = False
    last_seen: float = 0.0
    observations_received: int = 0
    observations_rejected: int = 0


class ConnectionManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._status: dict[int, GatewayStatus] = {}

    def on_connect(self, gateway_id: int):
        with self._lock:
            st = self._status.setdefault(gateway_id, GatewayStatus(gateway_id=gateway_id))
            st.connected = True
            st.last_seen = time.time()

    def on_disconnect(self, gateway_id: int):
        with self._lock:
            if gateway_id in self._status:
                self._status[gateway_id].connected = False

    def on_observation(self, gateway_id: int, accepted: bool):
        with self._lock:
            st = self._status.setdefault(gateway_id, GatewayStatus(gateway_id=gateway_id))
            st.last_seen = time.time()
            if accepted:
                st.observations_received += 1
            else:
                st.observations_rejected += 1

    def snapshot(self) -> dict[int, GatewayStatus]:
        with self._lock:
            return dict(self._status)
