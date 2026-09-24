// common/mds_encoder.h
//
// Reed-Solomon (n=5, k=3) systematic erasure encoder, GF(256), for the
// ESP32 transmitter nodes. This is a C++ PORT of the same math already
// verified in isac_lora_sim/fragmentation.py (exhaustively tested: all
// C(5,3)=10 fragment combinations decode correctly). Uses generator=3
// for the AES-standard polynomial 0x11B -- NOT generator=2, which was a
// real bug caught during the Python version's development (2 is not
// actually primitive for this polynomial, it only cycles through 51 of
// 255 field elements).
//
// Portable plain C++ -- no Arduino-specific headers, compiles and runs
// identically on a desktop g++ or inside Arduino IDE.
//
// Usage on the transmitter:
//   uint8_t chunks[K][FRAGMENT_SIZE];  // your k=3 data chunks
//   uint8_t fragments[N][FRAGMENT_SIZE]; // output: n=5 fragments to send
//   mds_encode(chunks, fragments, FRAGMENT_SIZE);
//   // then transmit each fragments[i] with fragment_id=i, total_fragments=N

#ifndef MDS_ENCODER_H_
#define MDS_ENCODER_H_

#include <stdint.h>
#include <string.h>

namespace mds {

constexpr uint8_t N = 5;  // total fragments generated
constexpr uint8_t K = 3;  // minimum needed to reconstruct

// ---- GF(256) tables, built once at startup ----
static uint8_t GF_EXP[512];
static uint8_t GF_LOG[256];

inline void gf_init() {
    uint16_t x = 1;
    for (int i = 0; i < 255; i++) {
        GF_EXP[i] = (uint8_t)x;
        GF_LOG[x] = (uint8_t)i;
        uint16_t doubled = x << 1;
        if (doubled & 0x100) doubled ^= 0x11B;
        x = doubled ^ x;  // x_new = 3*x_old = 2*x_old XOR x_old (generator=3, verified correct)
    }
    for (int i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
}

inline uint8_t gf_mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    return GF_EXP[(uint16_t)GF_LOG[a] + (uint16_t)GF_LOG[b]];
}

inline uint8_t gf_pow(uint8_t a, uint8_t power) {
    if (a == 0) return 0;
    return GF_EXP[((uint16_t)GF_LOG[a] * power) % 255];
}

// ---- Systematic generator matrix: identity for first K rows, Vandermonde for the rest ----
inline void build_generator(uint8_t gen[N][K]) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) gen[i][j] = (i == j) ? 1 : 0;
    }
    for (int i = K; i < N; i++) {
        for (int j = 0; j < K; j++) {
            gen[i][j] = gf_pow((uint8_t)(i - K + 2), j);
        }
    }
}

// ---- Encode: k data chunks -> n fragments, each frag_size bytes.
// chunks/fragments passed as arrays of pointers (not fixed-size 2D
// arrays) so frag_size is genuinely caller-controlled, not silently
// capped by a hardcoded dimension in the function signature -- a real
// design mistake caught and fixed before this ever got compiled/tested. ----
inline void mds_encode(const uint8_t* chunks[K], uint8_t* fragments[N], size_t frag_size) {
    static bool initialized = false;
    if (!initialized) { gf_init(); initialized = true; }

    uint8_t gen[N][K];
    build_generator(gen);

    for (int row = 0; row < N; row++) {
        memset(fragments[row], 0, frag_size);
        for (int col = 0; col < K; col++) {
            uint8_t coef = gen[row][col];
            if (coef == 0) continue;
            for (size_t b = 0; b < frag_size; b++) {
                fragments[row][b] ^= gf_mul(coef, chunks[col][b]);
            }
        }
    }
}

}  // namespace mds

#endif  // MDS_ENCODER_H_
