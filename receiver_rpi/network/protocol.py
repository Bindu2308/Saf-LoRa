"""
receiver/network/protocol.py

Parses the exact 42-byte GatewayObservation wire format defined in
common/protocol/observation.h / observation.cpp. If that C++ struct's
layout ever changes, this file must change identically -- there is no
shared schema between the two languages, so keep them in sync by hand.

Wire layout (42 bytes total):
  [0]      protocol_version   uint8
  [1]      gateway_id         uint8
  [2]      node_id            uint8
  [3..6]   telegram_id        uint32, BIG-endian
  [7]      fragment_id        uint8
  [8]      total_fragments    uint8
  [9..16]  timestamp_ms       uint64, BIG-endian
  [17..20] received_power_db  float32, little-endian (raw memcpy on ARM)
  [21..24] noise_power_db     float32, little-endian
  [25..28] sync_confidence    float32, little-endian
  [29..32] demod_confidence   float32, little-endian
  [33]     crc_ok             uint8 (0 or 1)
  [34..41] payload            8 raw bytes
"""

import struct
from dataclasses import dataclass

WIRE_SIZE = 42

# Header (protocol_version..total_fragments..timestamp_ms): big-endian
_HEADER_FMT = ">BBBIBBQ"   # 1+1+1+4+1+1+8 = 17 bytes
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 17

# Floats: raw memcpy in the C++ code, little-endian on both ARM Pis in this project
_FLOATS_FMT = "<ffff"      # 16 bytes
_FLOATS_SIZE = struct.calcsize(_FLOATS_FMT)  # 16

assert _HEADER_SIZE + _FLOATS_SIZE + 1 + 8 == WIRE_SIZE, "protocol.py layout drifted from observation.h"

EXPECTED_PROTOCOL_VERSION = 1  # must match OBSERVATION_PROTOCOL_VERSION in receiver_config.py


@dataclass
class GatewayObservation:
    protocol_version: int
    gateway_id: int
    node_id: int
    telegram_id: int
    fragment_id: int
    total_fragments: int
    timestamp_ms: int
    received_power_db: float
    noise_power_db: float
    sync_confidence: float
    demod_confidence: float
    crc_ok: bool
    payload: bytes  # 8 raw bytes; only meaningful if crc_ok is True
    feedback: int = 0  # H1 ACK-receipt report (upper 5 bits of wire byte 8)


class ProtocolError(ValueError):
    pass


def parse_observation(data: bytes) -> GatewayObservation:
    """Parses exactly WIRE_SIZE bytes into a GatewayObservation. Raises
    ProtocolError on malformed input (wrong length or unsupported version)
    -- callers must not let a malformed message crash the server (spec
    section 43: malformed observation must be handled, not fatal)."""
    if len(data) != WIRE_SIZE:
        raise ProtocolError(f"expected {WIRE_SIZE} bytes, got {len(data)}")

    (protocol_version, gateway_id, node_id, telegram_id,
     fragment_id, total_fragments, timestamp_ms) = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])

    (received_power_db, noise_power_db,
     sync_confidence, demod_confidence) = struct.unpack(
        _FLOATS_FMT, data[_HEADER_SIZE:_HEADER_SIZE + _FLOATS_SIZE])

    crc_ok_byte = data[_HEADER_SIZE + _FLOATS_SIZE]
    payload = data[_HEADER_SIZE + _FLOATS_SIZE + 1:]

    # H1: node packs its ACK-receipt report into the upper 5 bits of the
    # total_fragments byte (F=5 needs only 3 bits). Legacy nodes send 5 -> fb=0.
    feedback = total_fragments >> 3
    total_fragments = total_fragments & 0x07

    if protocol_version != EXPECTED_PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol_version {protocol_version}")

    return GatewayObservation(
        protocol_version=protocol_version,
        gateway_id=gateway_id,
        node_id=node_id,
        telegram_id=telegram_id,
        fragment_id=fragment_id,
        total_fragments=total_fragments,
        timestamp_ms=timestamp_ms,
        received_power_db=received_power_db,
        noise_power_db=noise_power_db,
        sync_confidence=sync_confidence,
        demod_confidence=demod_confidence,
        crc_ok=bool(crc_ok_byte),
        payload=payload,
        feedback=feedback,
    )
