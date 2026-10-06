"""
receiver/network/gateway_server.py

One TCP listener per gateway port (5000 for Gateway 1, 5001 for Gateway 2,
per the architecture doc and lora_config.h). Each accepted connection is
read on its own thread using the same length-prefixed framing the C++
GatewayClient writes: [2-byte big-endian length][observation bytes].

This module only does networking + framing + calling protocol.py to parse.
It hands each successfully parsed GatewayObservation to a callback --
it does NOT do association, combining, or reconstruction itself (spec:
"gateways/network layer don't decide, they observe and report").

NEW: also supports sending a downlink message back down the same
connection (D-FRAG's ACK relay -- see send_downlink()). The gateway's
firmware (gateway_esp32/main.cpp) reads this off the same TCP socket and
relays it as a LoRa packet to the node.
"""

import socket
import struct
import threading
from typing import Callable

from network.protocol import parse_observation, ProtocolError, WIRE_SIZE
from network.connection_manager import ConnectionManager
from logging_utils.logger import get_logger

log = get_logger("gateway_server")

ObservationCallback = Callable[["object"], None]  # takes a GatewayObservation


class GatewayServer:
    def __init__(self, port: int, expected_gateway_id: int,
                 on_observation: ObservationCallback,
                 conn_mgr: ConnectionManager,
                 host: str = "0.0.0.0"):
        self.port = port
        self.expected_gateway_id = expected_gateway_id
        self.on_observation = on_observation
        self.conn_mgr = conn_mgr
        self.host = host
        self._sock: socket.socket | None = None
        self._running = False
        self._accept_thread: threading.Thread | None = None
        # Tracks the CURRENT active client connection so send_downlink()
        # has something to write to. If the gateway reconnects, this is
        # updated to the new connection automatically (see _client_loop).
        self._current_conn: socket.socket | None = None
        self._conn_lock = threading.Lock()

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(4)
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        log.info(f"listening on {self.host}:{self.port} (expecting gateway_id={self.expected_gateway_id})")

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def send_downlink(self, packet: bytes) -> bool:
        """Sends a raw downlink packet (e.g. a D-FRAG ACK relay message)
        down the current gateway connection, if one exists. Returns
        True if the write succeeded, False if there's no connection or
        the write failed -- callers should treat False as "couldn't
        relay right now," not as an error to crash on (spec section 43:
        ML/downlink failures must not stop the receiver)."""
        with self._conn_lock:
            conn = self._current_conn
        if conn is None:
            return False
        try:
            conn.sendall(packet)
            return True
        except OSError as e:
            log.warn(f"send_downlink failed on port {self.port}: {e}")
            return False

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break  # socket closed during stop()
            log.info(f"gateway connected from {addr} on port {self.port}")
            with self._conn_lock:
                self._current_conn = conn
            t = threading.Thread(target=self._client_loop, args=(conn, addr), daemon=True)
            t.start()

    def _client_loop(self, conn: socket.socket, addr):
        conn.settimeout(30.0)  # spec section 43: a stalled gateway connection shouldn't hang us forever
        self.conn_mgr.on_connect(self.expected_gateway_id)
        try:
            while self._running:
                header = self._recv_exact(conn, 2)
                if header is None:
                    break
                (length,) = struct.unpack(">H", header)
                if length != WIRE_SIZE:
                    log.warn(f"malformed frame from {addr}: length={length}, expected {WIRE_SIZE}; dropping connection")
                    break
                body = self._recv_exact(conn, length)
                if body is None:
                    break
                try:
                    obs = parse_observation(body)
                except ProtocolError as e:
                    log.warn(f"malformed observation from {addr}: {e}")
                    self.conn_mgr.on_observation(self.expected_gateway_id, accepted=False)
                    continue

                if obs.gateway_id != self.expected_gateway_id:
                    log.warn(f"observation on port {self.port} claims gateway_id={obs.gateway_id}, "
                             f"expected {self.expected_gateway_id} -- accepting anyway but flagging")

                self.conn_mgr.on_observation(obs.gateway_id, accepted=True)
                self.on_observation(obs)
        except (socket.timeout, ConnectionResetError, OSError) as e:
            log.info(f"gateway {addr} disconnected: {e}")
        finally:
            self.conn_mgr.on_disconnect(self.expected_gateway_id)
            with self._conn_lock:
                if self._current_conn is conn:
                    self._current_conn = None
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None  # peer closed
            buf += chunk
        return buf

    def has_connection(self) -> bool:
        """True if a gateway is currently connected. Used by the downlink
        scheduler to know which gateways are candidates."""
        with self._conn_lock:
            return self._current_conn is not None
