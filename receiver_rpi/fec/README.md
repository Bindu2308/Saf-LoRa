# fec/ — scope note

The architecture doc's `fec/` folder lists `hamming74.py`, `interleaver.py`,
`deinterleaver.py`, `whitening.py` as receiver-side modules. Those aren't
here (yet), and this is deliberate, not an oversight:

Today, PHY-level FEC (Gray decode → deinterleave → Hamming decode →
dewhiten) already happens **at the gateway**, in
`common/lora_phy/lora_demodulator.cpp`, using the exact same functions
Gateway 1 and Gateway 2 both link against. The gateway sends the RPi4 an
already-decoded 8-byte payload plus a `crc_ok` flag — not raw symbols or
LLRs.

The doc's `fec/` folder is designed for the **soft-combining path**
(section 14): combine LLRs from multiple gateways first, *then* run
Gray/deinterleave/Hamming/dewhiten once on the combined result. That path
needs per-bit LLRs from the gateways, which don't exist yet — see
`cooperation/soft_combiner.py`'s docstring for exactly what's missing.

Once gateways emit real LLR arrays, the right move is to reuse
`common/lora_phy`'s existing C++ functions (via a Python binding, or by
porting them) rather than writing a second, parallel Python FEC
implementation — that's exactly the Gateway1/Gateway2 divergence problem
this project already hit once, just at a different layer.

`mds.py` (the erasure/fragment-assembly step) is separate from this and
does exist — see that file's own docstring for its current scope.
