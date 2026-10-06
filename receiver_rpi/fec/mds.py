"""
receiver/fec/mds.py

Real Reed-Solomon (n=5, k=3) erasure decoder over GF(256), matching the
encoder now running on the transmitter (common/mds_encoder.h -- verified
byte-for-byte against this exact decode logic before either was deployed).
"""

from typing import Optional
import numpy as np

N = 5
K = 3

_GF_EXP = [0] * 512
_GF_LOG = [0] * 256


def _init_gf_tables():
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        double = x << 1
        if double & 0x100:
            double ^= 0x11B
        x = double ^ x
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]


_init_gf_tables()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_pow(a: int, power: int) -> int:
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] * power) % 255]


def _gf_inv(a: int) -> int:
    return _GF_EXP[255 - _GF_LOG[a]]


def _build_generator() -> np.ndarray:
    gen = np.zeros((N, K), dtype=np.uint8)
    for i in range(K):
        gen[i, i] = 1
    for i in range(K, N):
        for j in range(K):
            gen[i, j] = _gf_pow(i - K + 2, j)
    return gen


_GENERATOR = _build_generator()


def reconstruct_telegram(fragment_payloads: dict, total_fragments: int) -> Optional[bytes]:
    if len(fragment_payloads) < K:
        return None

    indices = sorted(fragment_payloads.keys())[:K]
    frag_size = len(fragment_payloads[indices[0]])

    sub_matrix = _GENERATOR[indices, :]

    aug = sub_matrix.astype(np.int32).copy()
    inv = np.eye(K, dtype=np.int32)
    for col in range(K):
        pivot_row = None
        for r in range(col, K):
            if aug[r, col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            return None

        aug[[col, pivot_row]] = aug[[pivot_row, col]]
        inv[[col, pivot_row]] = inv[[pivot_row, col]]

        pivot_inv = _gf_inv(int(aug[col, col]))
        aug[col] = [_gf_mul(pivot_inv, int(v)) for v in aug[col]]
        inv[col] = [_gf_mul(pivot_inv, int(v)) for v in inv[col]]

        for r in range(K):
            if r != col and aug[r, col] != 0:
                factor = int(aug[r, col])
                aug[r] = [aug[r, c] ^ _gf_mul(factor, int(aug[col, c])) for c in range(K)]
                inv[r] = [inv[r, c] ^ _gf_mul(factor, int(inv[col, c])) for c in range(K)]

    frag_bytes = [np.frombuffer(fragment_payloads[i], dtype=np.uint8) for i in indices]

    chunks = np.zeros((K, frag_size), dtype=np.uint8)
    for out_row in range(K):
        acc = np.zeros(frag_size, dtype=np.uint8)
        for in_row in range(K):
            coef = int(inv[out_row, in_row])
            if coef != 0:
                for b in range(frag_size):
                    acc[b] ^= _gf_mul(coef, int(frag_bytes[in_row][b]))
        chunks[out_row] = acc

    message = chunks.tobytes()
    return message.rstrip(b"\x00")
