"""
receiver/logging_utils/logger.py

Minimal leveled logger (ERROR/WARN/INFO/DEBUG) per spec section 34/35.
Deliberately dependency-free (stdlib print, timestamped) so the receiver
has zero pip requirements for Phase 1-8 bring-up.
"""

import time


LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3, "TRACE": 4}

# Change this to "DEBUG" or "TRACE" for more verbose output during bring-up.
CURRENT_LEVEL = "INFO"


class Logger:
    def __init__(self, name: str):
        self.name = name

    def _log(self, level: str, msg: str):
        if LEVELS[level] > LEVELS[CURRENT_LEVEL]:
            return
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [{level}] [{self.name}] {msg}", flush=True)

    def error(self, msg: str): self._log("ERROR", msg)
    def warn(self, msg: str): self._log("WARN", msg)
    def info(self, msg: str): self._log("INFO", msg)
    def debug(self, msg: str): self._log("DEBUG", msg)
    def trace(self, msg: str): self._log("TRACE", msg)


def get_logger(name: str) -> Logger:
    return Logger(name)
