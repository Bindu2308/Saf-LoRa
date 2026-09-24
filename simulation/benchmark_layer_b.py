"""
isac_lora_sim/benchmark_layer_b.py  (v2 -- corrected)

v1 of this script drew channel state independently each round, which
destroys the temporal persistence a real Gilbert-Elliott channel has --
and which EWMA-style learning specifically depends on. That produced a
misleading result (EWMA looked useless). This version uses the SAME
JammedGEChannel class already verified elsewhere in this codebase,
which correctly evolves state as a Markov chain across successive
rounds for each (node, gateway) link.

Implements "Benchmark Layer B" from the figure/benchmark plan: a fair,
same-testbed comparison against strong baselines, not a cross-paper
metric comparison.
"""

import random
import numpy as np
import statistics as st
from dataclasses import dataclass

from topology import make_topology
from gilbert_elliott import distance_scaled_params
from gilbert_elliott_jam_patch import JammedGEChannel
from hmm_estimator import ChannelHMM

POLICIES = ("random", "round-robin", "snr-greedy", "ewma-success",
            "belief-only", "state-aware", "oracle")


@dataclass
class Round:
    node_id: int
    gw_snr: dict
    gw_belief: dict
    gw_true_state: dict
    gw_true_delivery_prob: dict   # the FULL latent ground truth (state + severity), for Oracle
    gw_delivered: dict


def simulate_rounds(num_nodes=10, rounds_per_node=200, jammed_gateway=None,
                     jam_severity_db=0.0, seed=42, base_delivery_prob=0.75):
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)
    topo = make_topology(num_nodes, 2, 1300.0, seed=seed)

    channels, hmms = {}, {}
    channel_sev = {}
    for n in range(num_nodes):
        for g in (1, 2):
            dist = topo.distance(n, g - 1)
            params = distance_scaled_params(dist)
            sev = jam_severity_db if g == jammed_gateway else 0.0
            channels[(n, g)] = JammedGEChannel(params, rng, jam_severity_db=sev)
            hmms[(n, g)] = ChannelHMM()
            channel_sev[(n, g)] = sev

    rounds = []
    for n in range(num_nodes):
        for _ in range(rounds_per_node):
            gw_snr, gw_belief, gw_true_state, gw_true_prob, gw_delivered = {}, {}, {}, {}, {}
            for g in (1, 2):
                outcome, quality_db, state = channels[(n, g)].step()
                belief = hmms[(n, g)].update(quality_db)
                gw_snr[g] = quality_db if quality_db is not None else -99.0
                gw_belief[g] = belief
                gw_true_state[g] = state

                # Delivery probability reflects BOTH true channel state
                # AND jamming severity. This full latent value is what
                # Oracle must use -- picking on binary state alone was a
                # bug: a gateway can be in the GOOD state yet still be
                # the jammed one, so state alone does not identify the
                # actually-best gateway once severity also matters.
                delivery_p = base_delivery_prob if state == "GOOD" else max(0.05, base_delivery_prob - 0.5)
                delivery_p = max(0.0, min(1.0, delivery_p - channel_sev[(n, g)] * 0.04))
                gw_true_prob[g] = delivery_p
                gw_delivered[g] = py_rng.random() < delivery_p

            rounds.append(Round(n, gw_snr, gw_belief, gw_true_state, gw_true_prob, gw_delivered))
    return rounds


class EWMASuccessPolicy:
    def __init__(self, alpha=0.2, prior=0.5):
        self.alpha = alpha
        self.prior = prior
        self._belief = {}

    def score(self, node_id, gw):
        return self._belief.get((node_id, gw), self.prior)

    def learn(self, node_id, gw, delivered):
        key = (node_id, gw)
        cur = self._belief.get(key, self.prior)
        self._belief[key] = (1 - self.alpha) * cur + self.alpha * (1.0 if delivered else 0.0)


class FusedStateAwarePolicy:
    """State-aware = mostly live uplink belief, with a SMALL learned
    correction from genuine downlink outcomes (fusion_weight=0.2,
    verified as the best tested value -- higher weights on learned
    history hurt, since live belief remains the stronger signal).
    Distinct from belief-only, which uses uplink belief alone with no
    learning at all."""

    def __init__(self, alpha=0.2, prior=0.5, fusion_weight=0.2):
        self.alpha = alpha
        self.prior = prior
        self.fusion_weight = fusion_weight
        self._learned = {}

    def score(self, node_id, gw, belief):
        learned = self._learned.get((node_id, gw), self.prior)
        return self.fusion_weight * learned + (1 - self.fusion_weight) * belief

    def learn(self, node_id, gw, delivered):
        key = (node_id, gw)
        cur = self._learned.get(key, self.prior)
        self._learned[key] = (1 - self.alpha) * cur + self.alpha * (1.0 if delivered else 0.0)


def run_policy(policy, rounds, seed=42):
    rng = random.Random(seed)
    ewma = EWMASuccessPolicy()
    fused = FusedStateAwarePolicy()
    rr_next = {}
    delivered_count = 0
    gw_count = {1: 0, 2: 0}

    for r in rounds:
        if policy == "random":
            g = rng.choice((1, 2))
        elif policy == "round-robin":
            i = rr_next.get(r.node_id, 0)
            g = (1, 2)[i % 2]
            rr_next[r.node_id] = i + 1
        elif policy == "snr-greedy":
            g = max((1, 2), key=lambda gw: r.gw_snr[gw])
        elif policy == "ewma-success":
            g = max((1, 2), key=lambda gw: ewma.score(r.node_id, gw))
        elif policy == "belief-only":
            g = max((1, 2), key=lambda gw: r.gw_belief[gw])
        elif policy == "state-aware":
            g = max((1, 2), key=lambda gw: fused.score(r.node_id, gw, r.gw_belief[gw]))
        elif policy == "oracle":
            g = max((1, 2), key=lambda gw: r.gw_true_delivery_prob[gw])
        else:
            raise ValueError(policy)

        delivered = r.gw_delivered[g]
        delivered_count += int(delivered)
        gw_count[g] += 1
        ewma.learn(r.node_id, g, delivered)
        fused.learn(r.node_id, g, delivered)

    total = len(rounds)
    return delivered_count / total, 100.0 * gw_count[1] / total


def g_norm(r_policy, r_random, r_oracle):
    denom = r_oracle - r_random
    return 0.0 if abs(denom) < 1e-9 else (r_policy - r_random) / denom


def main():
    print("=" * 70)
    print("BENCHMARK LAYER B (v2, corrected): same-testbed policy comparison")
    print("=" * 70)

    severities = [0.0, 1.0, 2.0, 3.0]
    n_seeds = 15

    print(f"\n{'severity(dB)':<14}" + "".join(f"{p:<14}" for p in POLICIES))
    print("-" * (14 + 14 * len(POLICIES)))

    results = {}
    for sev in severities:
        jgw = 1 if sev > 0 else None
        rates = {p: [] for p in POLICIES}
        for seed in range(n_seeds):
            rounds = simulate_rounds(jammed_gateway=jgw, jam_severity_db=sev, seed=seed)
            for p in POLICIES:
                rate, _ = run_policy(p, rounds, seed=seed)
                rates[p].append(rate)

        means = {p: st.mean(rates[p]) for p in POLICIES}
        sds = {p: st.stdev(rates[p]) for p in POLICIES}
        gnorms = {p: g_norm(means[p], means["random"], means["oracle"]) for p in POLICIES}
        results[sev] = (means, sds, gnorms)

        print(f"{sev:<14.1f}" + "".join(f"{means[p]*100:<14.1f}" for p in POLICIES))

    print(f"\n{'severity(dB)':<14}" + "".join(f"{p:<14}" for p in POLICIES))
    print("G_norm (0=random, 1=oracle):")
    for sev in severities:
        _, _, gnorms = results[sev]
        print(f"{sev:<14.1f}" + "".join(f"{gnorms[p]:<14.2f}" for p in POLICIES))

    return results


if __name__ == "__main__":
    main()
