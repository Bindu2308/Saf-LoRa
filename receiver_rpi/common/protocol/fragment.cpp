#include "fragment.h"

namespace isac_lora {

std::array<uint8_t, FRAGMENT_WIRE_SIZE> Fragment::serialize() const {
    std::array<uint8_t, FRAGMENT_WIRE_SIZE> out{};
    out[0] = node_id;
    out[1] = static_cast<uint8_t>((telegram_id >> 24) & 0xFF);
    out[2] = static_cast<uint8_t>((telegram_id >> 16) & 0xFF);
    out[3] = static_cast<uint8_t>((telegram_id >> 8) & 0xFF);
    out[4] = static_cast<uint8_t>(telegram_id & 0xFF);
    out[5] = fragment_id;
    out[6] = total_fragments;
    for (size_t i = 0; i < FRAGMENT_PAYLOAD_SIZE; ++i) {
        out[7 + i] = payload[i];
    }
    return out;
}

bool Fragment::parse(const uint8_t* data, size_t len, Fragment* out) {
    if (len < FRAGMENT_WIRE_SIZE || out == nullptr) return false;
    out->node_id = data[0];
    out->telegram_id = (static_cast<uint32_t>(data[1]) << 24) |
                        (static_cast<uint32_t>(data[2]) << 16) |
                        (static_cast<uint32_t>(data[3]) << 8) |
                        static_cast<uint32_t>(data[4]);
    out->fragment_id = data[5];
    out->total_fragments = data[6];
    for (size_t i = 0; i < FRAGMENT_PAYLOAD_SIZE; ++i) {
        out->payload[i] = data[7 + i];
    }
    return true;
}

} // namespace isac_lora
