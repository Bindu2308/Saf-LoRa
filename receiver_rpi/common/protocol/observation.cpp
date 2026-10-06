#include "observation.h"
#include <cstring>

namespace isac_lora {

namespace {
void put_u32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>((v >> 24) & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 16) & 0xFF);
    p[2] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[3] = static_cast<uint8_t>(v & 0xFF);
}
uint32_t get_u32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}
void put_u64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>((v >> (56 - 8 * i)) & 0xFF);
}
uint64_t get_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}
// Floats are transferred as raw bytes. Both Gateway (ARM, Raspberry Pi) and
// RPi4 (also ARM) are little-endian, so this is safe within this project;
// if a non-ARM host is ever added, switch to an explicit IEEE754 packer.
void put_f32(uint8_t* p, float v) { std::memcpy(p, &v, 4); }
float get_f32(const uint8_t* p) { float v; std::memcpy(&v, p, 4); return v; }
} // namespace

std::array<uint8_t, GatewayObservation::WIRE_SIZE> GatewayObservation::serialize() const {
    std::array<uint8_t, WIRE_SIZE> out{};
    size_t i = 0;
    out[i++] = protocol_version;
    out[i++] = gateway_id;
    out[i++] = node_id;
    put_u32(&out[i], telegram_id); i += 4;
    out[i++] = fragment_id;
    out[i++] = total_fragments;
    put_u64(&out[i], timestamp_ms); i += 8;
    put_f32(&out[i], received_power_db); i += 4;
    put_f32(&out[i], noise_power_db); i += 4;
    put_f32(&out[i], sync_confidence); i += 4;
    put_f32(&out[i], demod_confidence); i += 4;
    out[i++] = crc_ok;
    for (size_t k = 0; k < OBS_PAYLOAD_SIZE; ++k) out[i++] = payload[k];
    return out;
}

bool GatewayObservation::parse(const uint8_t* data, size_t len, GatewayObservation* out) {
    if (len < WIRE_SIZE || out == nullptr) return false;
    size_t i = 0;
    out->protocol_version = data[i++];
    out->gateway_id = data[i++];
    out->node_id = data[i++];
    out->telegram_id = get_u32(&data[i]); i += 4;
    out->fragment_id = data[i++];
    out->total_fragments = data[i++];
    out->timestamp_ms = get_u64(&data[i]); i += 8;
    out->received_power_db = get_f32(&data[i]); i += 4;
    out->noise_power_db = get_f32(&data[i]); i += 4;
    out->sync_confidence = get_f32(&data[i]); i += 4;
    out->demod_confidence = get_f32(&data[i]); i += 4;
    out->crc_ok = data[i++];
    for (size_t k = 0; k < OBS_PAYLOAD_SIZE; ++k) out->payload[k] = data[i++];
    return true;
}

} // namespace isac_lora
