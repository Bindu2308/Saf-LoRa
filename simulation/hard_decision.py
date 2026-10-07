"""
isac_lora_sim/hard_decision.py

Makes the simulation's RECONSTRUCTION decision match real hardware
exactly: a fragment only enters the availability set if it is CRC-valid
("CLEAN"). CORRUPTED and LOST observations are both treated as "not
received" for reconstruction -- this is the real gateway's behavior
(receiver/network/protocol.py: crc_ok gates hard_combiner.py's fragment
pool) -- but CORRUPTED observations' quality_db still updates the HMM,
exactly matching the real gateway's CRC-fail-forwarding change.

Before this module: simulation strategies decided for THEMSELVES whether
to trust a CORRUPTED observation (soft combining), which is the
soft-decision idealization the paper's Proposition 1 explicitly
separates from hardware. After this module: the CRC gate is applied
BEFORE any strategy runs, so no combining rule -- state-blind, HMM-aware,
or genie -- can ever see a corrupted fragment as a candidate for
reconstruction. This is what makes the simulation's availability outcome
provably strategy-invariant, exactly as Proposition 1 proves
analytically for the real hardware.
"""

from dataclasses import dataclass


@dataclass
class HardDecisionObservation:
    node_id: int
    telegram_id: int
    fragment_id: int
    gateway_id: int
    crc_ok: bool          # True only if outcome == "CLEAN"
    quality_db: float      # still populated for CORRUPTED (HMM input); None for LOST


def apply_crc_gate(raw_observations):
    """Converts the simulator's raw (outcome, quality_db, state) stream
    into hard-decision observations. outcome == "CLEAN" -> crc_ok=True.
    outcome in {"CORRUPTED", "LOST"} -> crc_ok=False; CORRUPTED still
    carries a quality_db reading (the gateway measured something, it
    just didn't pass CRC), LOST carries None (nothing arrived at all).
    """
    gated = []
    for obs in raw_observations:
        crc_ok = (obs["outcome"] == "CLEAN")
        gated.append(HardDecisionObservation(
            node_id=obs["node_id"], telegram_id=obs["telegram_id"],
            fragment_id=obs["fragment_id"], gateway_id=obs["gateway_id"],
            crc_ok=crc_ok, quality_db=obs["quality_db"],
        ))
    return gated


def hard_decision_availability(observations, K):
    """The ONE reconstruction rule every strategy now shares: union of
    CRC-valid fragment IDs, reconstruct iff |union| >= K. No strategy
    parameter changes this function's output -- that is the point."""
    by_telegram = {}
    for obs in observations:
        if not obs.crc_ok:
            continue
        key = (obs.node_id, obs.telegram_id)
        by_telegram.setdefault(key, set()).add(obs.fragment_id)

    results = {}
    for key, fragment_set in by_telegram.items():
        results[key] = len(fragment_set) >= K
    return results


def assert_union_invariance(observations, K, strategy_names, strategy_fns):
    """Proves, for this observation set, that every listed strategy
    produces IDENTICAL per-telegram reconstruction outcomes once the CRC
    gate is applied -- the empirical counterpart to Proposition 1.
    strategy_fns: dict[name -> callable(observations, K) -> {key: bool}],
    each representing a DIFFERENT weighting/selection rule over the SAME
    gated observations.
    """
    reference_name = strategy_names[0]
    reference = strategy_fns[reference_name](observations, K)
    mismatches = {}
    for name in strategy_names[1:]:
        result = strategy_fns[name](observations, K)
        diffs = {k: (reference.get(k), result.get(k))
                 for k in set(reference) | set(result)
                 if reference.get(k) != result.get(k)}
        if diffs:
            mismatches[name] = diffs
    return mismatches
