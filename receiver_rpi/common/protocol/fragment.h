#pragma once
#include <cstdint>
#include <vector>
#include <array>

namespace isac_lora {

// NOTE ON DEVIATION FROM THE GENERIC SPEC (section 7):
// The generic spec proposes an explicit payload_length field. This
// project's ESP32 firmware (already flashed and verified working) uses a
// fixed-size 8-byte payload with NO explicit length field -- the final
// fragment is space-padded instead. We preserve that exact wire format
// here rather than changing it, per "preserve compatibility with existing
// working interfaces." If variable-length payloads are needed later, add
// payload_length as an explicit new field bump to
// OBSERVATION_PROTOCOL_VERSION rather than silently changing this format.
//
// Wire format (15 bytes total, matches confirmed ESP32 output):
//   [0]      node_id            (1 byte)
//   [1..4]   telegram_id        (4 bytes, big-endian uint32)
//   [5]      fragment_id        (1 byte)
//   [6]      total_fragments    (1 byte)
//   [7..14]  payload            (8 bytes, space-padded on final fragment)

constexpr size_t FRAGMENT_WIRE_SIZE = 15;
constexpr size_t FRAGMENT_PAYLOAD_SIZE = 8;

struct Fragment {
    uint8_t  node_id = 0;
    uint32_t telegram_id = 0;
    uint8_t  fragment_id = 0;
    uint8_t  total_fragments = 0;
    std::array<uint8_t, FRAGMENT_PAYLOAD_SIZE> payload{};

    // Serialize to the exact 15-byte wire format.
    std::array<uint8_t, FRAGMENT_WIRE_SIZE> serialize() const;

    // Parse from a 15-byte buffer. Returns false if `data` isn't long enough.
    static bool parse(const uint8_t* data, size_t len, Fragment* out);
};

} // namespace isac_lora
