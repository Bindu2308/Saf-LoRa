"""
receiver/config/receiver_config.py

IMPORTANT: Python can't #include the C++ common/config/lora_config.h, so
these values are a manual mirror of it. If you ever change a gateway's
port, timeout, or fragment size in lora_config.h, update this file too --
nothing enforces that they stay in sync automatically.
"""

GATEWAY1_TCP_PORT = 5000
GATEWAY2_TCP_PORT = 5001
GATEWAY3_TCP_PORT = 5002
GATEWAY_PORTS = {1: GATEWAY1_TCP_PORT, 2: GATEWAY2_TCP_PORT, 3: GATEWAY3_TCP_PORT}

# How long (ms) a telegram waits for more fragments before being finalized
# (reconstructed if possible, or reported incomplete). Matches
# TELEGRAM_TIMEOUT_MS in lora_config.h.
TELEGRAM_TIMEOUT_MS = 5000

FRAGMENT_PAYLOAD_SIZE = 8  # matches FRAGMENT_PAYLOAD_SIZE / OBS_PAYLOAD_SIZE

# Observation protocol version this receiver understands. Must match
# OBS_PROTOCOL_VERSION in common/protocol/observation.h.
OBSERVATION_PROTOCOL_VERSION = 1

# How often (seconds) the background sweep checks for timed-out telegrams.
TIMEOUT_SWEEP_INTERVAL_S = 1.0

# How often (seconds) to print a metrics summary.
METRICS_PRINT_INTERVAL_S = 10.0
