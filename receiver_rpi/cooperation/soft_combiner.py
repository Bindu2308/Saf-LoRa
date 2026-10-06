"""
receiver/cooperation/soft_combiner.py

NOT YET FUNCTIONAL. Documented here rather than silently omitted so the
gap is visible, per architecture doc sections 12/14 which describe
LLR-level combining (LLR_combined = w1*LLR1 + w2*LLR2, then re-run
Gray decode -> deinterleave -> Hamming -> dewhiten on the COMBINED
symbols).

That requires each gateway to forward a per-bit (or at minimum
per-symbol) LLR array, which the current gateway pipeline does not
produce -- common/gateway/synchronization/synchronizer.cpp and
common/lora_phy/lora_demodulator.cpp currently do a hard decode and report
only a single scalar avg_symbol_confidence per fragment (see
common/protocol/observation.h's comment on this exact point).

Building this out is real, separate work:
  1. common/lora_phy/chirp.h's demodulate_symbol() would need to return
     per-bit LLRs from the FFT magnitude spectrum, not just a hard peak-bin
     symbol + one scalar confidence.
  2. common/protocol/observation.h's wire format would need a variable-
     length LLR array field (protocol version bump).
  3. This module would then do the weighted LLR sum and re-run
     Gray decode/deinterleave/Hamming/dewhiten on the combined result,
     reusing common/lora_phy's existing functions on the COMBINED symbols
     rather than per-gateway hard-decoded bytes.

Until all three exist, hard_combiner.py is the real combining path in use.
Do not call soft_combine() -- it exists as a placeholder for that future
work, not a working implementation.
"""


def soft_combine(*args, **kwargs):
    raise NotImplementedError(
        "Soft (LLR-level) combining requires per-bit LLR output from the "
        "gateway PHY layer, which is not implemented yet. See this file's "
        "module docstring for what's needed. Use hard_combiner.py instead."
    )
