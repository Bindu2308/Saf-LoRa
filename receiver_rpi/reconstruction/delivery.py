"""
receiver/reconstruction/delivery.py

Final handoff of a reconstructed telegram to "the application." For now
this just logs it and keeps an in-memory list -- swap in a real sink
(file, MQTT, HTTP callback, etc.) once you have a downstream consumer.
"""

import threading
import time
from dataclasses import dataclass

from logging_utils.logger import get_logger

log = get_logger("delivery")


@dataclass
class DeliveredMessage:
    node_id: int
    telegram_id: int
    message: bytes
    delivered_at: float


class Delivery:
    def __init__(self):
        self._lock = threading.Lock()
        self._history: list[DeliveredMessage] = []

    def deliver(self, node_id: int, telegram_id: int, message: bytes):
        record = DeliveredMessage(node_id=node_id, telegram_id=telegram_id,
                                   message=message, delivered_at=time.time())
        with self._lock:
            self._history.append(record)
        try:
            text = message.decode("utf-8")
        except UnicodeDecodeError:
            text = repr(message)
        log.info(f"[DELIVERY] node={node_id} telegram={telegram_id}: \"{text}\"")

    def history(self) -> list[DeliveredMessage]:
        with self._lock:
            return list(self._history)
