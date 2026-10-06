#pragma once
//
// lora_config.h
// SINGLE SOURCE OF TRUTH for all LoRa PHY parameters.
// TX, Gateway 1, and Gateway 2 must all include this file.
// Never redefine these constants locally in a gateway-specific file.
//
#include <cstdint>

namespace isac_lora {

// ---- RF parameters (must match across TX and all gateways) ----
constexpr double   LORA_FREQ_HZ        = 868000000.0; // 868 MHz
constexpr double   LORA_SAMPLE_RATE_HZ = 2000000.0;   // 2 MHz (gateway SDR capture rate)
constexpr double   LORA_BANDWIDTH_HZ   = 125000.0;    // 125 kHz LoRa BW
constexpr int      LORA_SF             = 7;           // Spreading factor
constexpr int      LORA_CR             = 5;           // Coding rate denominator (4/5 -> CR=1 in LoRa header enc, but
                                                        // we use "5" to mean 4/5 throughout this codebase; see hamming.h)
constexpr int      LORA_PREAMBLE_SYMS  = 8;            // preamble length in symbols
constexpr uint16_t LORA_SYNC_WORD      = 0x12;         // matches RadioLib default private sync word
constexpr bool     LORA_IQ_INVERTED    = false;

// ---- Derived quantities (do not hand-edit; derived at compile time) ----
constexpr int LORA_SYMS_PER_SEC   = 1 << LORA_SF;                 // 2^SF chips per symbol (=128 for SF7)
// samples per symbol at the CHIRP's native rate (bandwidth-rate sampling, i.e. Fs == BW):
constexpr int LORA_NATIVE_SPS     = 1 << LORA_SF;                 // 128 samples/symbol at Fs=BW
// samples per symbol at the GATEWAY's actual capture rate (Fs=2MHz, BW=125kHz -> oversample factor 16):
constexpr int LORA_OVERSAMPLE     = static_cast<int>(LORA_SAMPLE_RATE_HZ / LORA_BANDWIDTH_HZ); // = 16
constexpr int LORA_GATEWAY_SPS    = LORA_NATIVE_SPS * LORA_OVERSAMPLE; // = 128*16 = 2048 samples/symbol at 2MHz capture

// IMPORTANT (see spec section 12): LORA_GATEWAY_SPS (2048) is a *sample count*, not an FFT size.
// The demodulator dechirps at the native rate conceptually; when working with 2MHz-captured IQ,
// either (a) decimate by LORA_OVERSAMPLE first, then run a 128-point FFT, or
//        (b) run a zero-padded FFT at the oversampled rate.
// This codebase uses (a): decimate-then-128-FFT. Never conflate LORA_GATEWAY_SPS with an FFT size.

// ---- Fragment / erasure coding parameters ----
constexpr int MDS_TOTAL_FRAGMENTS = 5;   // F
constexpr int MDS_MIN_FRAGMENTS   = 2;   // K
constexpr int FRAGMENT_PAYLOAD_MAX_BYTES = 8; // matches existing ESP32 firmware chunk size

// ---- Network ----
constexpr int GATEWAY1_TCP_PORT = 5000;
constexpr int GATEWAY2_TCP_PORT = 5001;
constexpr int TELEGRAM_TIMEOUT_MS = 5000;

// ---- Protocol versioning (spec section 42) ----
constexpr uint8_t OBSERVATION_PROTOCOL_VERSION = 1;

} // namespace isac_lora
