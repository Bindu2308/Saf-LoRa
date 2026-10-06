#pragma once
#include <cstdint>
#include <array>
#include <vector>

namespace isac_lora {

constexpr uint8_t OBS_PROTOCOL_VERSION = 1;
constexpr size_t OBS_PAYLOAD_SIZE = 8; // matches FRAGMENT_PAYLOAD_SIZE

// Produced by a gateway after processing one candidate detection. Sent
// as-is to the RPi4 regardless of crc_ok -- per the architecture doc
// (section 13), a CRC failure doesn't mean the observation is worthless;
// the RPi4's HMM/combining stage may still use timing/confidence/power
// info from a failed decode. Real per-bit LLR array (doc section 11) is a
// later refinement (Phase 8) -- this phase carries avg_symbol_confidence
// as the soft-information proxy, which is what the current demodulator
// actually produces. Do not claim LLRs are present until they really are.
struct GatewayObservation {
    uint8_t  protocol_version = OBS_PROTOCOL_VERSION;
    uint8_t  gateway_id = 0;

    uint8_t  node_id = 0;
    uint32_t telegram_id = 0;
    uint8_t  fragment_id = 0;
    uint8_t  total_fragments = 0;

    uint64_t timestamp_ms = 0;

    float received_power_db = 0.0f;
    float noise_power_db = 0.0f;
    float sync_confidence = 0.0f;   // from signal detector candidate
    float demod_confidence = 0.0f;  // avg chirp-FFT peak confidence across payload symbols

    uint8_t crc_ok = 0; // 0/1, not bool, for stable wire size

    std::array<uint8_t, OBS_PAYLOAD_SIZE> payload{}; // valid contents only if crc_ok==1

    static constexpr size_t WIRE_SIZE =
        1 + 1 + 1 + 4 + 1 + 1 + 8 + 4 + 4 + 4 + 4 + 1 + OBS_PAYLOAD_SIZE; // = 42 bytes

    std::array<uint8_t, WIRE_SIZE> serialize() const;
    static bool parse(const uint8_t* data, size_t len, GatewayObservation* out);
};

} // namespace isac_lora
