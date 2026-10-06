"""
SAF-LoRa gateway-asymmetry sweep (Experiment S1).

Validates Proposition 4: state-aware gain should grow with belief
asymmetry |b1 - b2| between two gateways, and vanish as asymmetry -> 0.

Two-gateway setup only, matching Corollary 3.1 / Proposition 4's
two-gateway specialization. Gateway 1 is the fixed baseline; Gateway 2's
true delivery probability is offset by -delta_p relative to Gateway 1,
sweeping delta_p across a grid. Both gateways share identical GOOD/BAD
loss parameters (epsilon_G, epsilon_B); only their GOOD-state share
differs, consistent with the Proposition 4 assumption.

Policies compared: Random, Round-robin, EWMA (confirmed-outcome based),
Belief-only (== State-aware per the paper's uplink-primed architecture),
Oracle (upper-bound reference only, not a real policy).

Run:
    python run_gateway_asymmetry.py
"""

import csv
import random
import statistics as st
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gilbert_elliott import GEParams
from gilbert_elliott_jam_patch import JammedGEChannel
from hmm_estimator import ChannelHMM


EWMA_ALPHA = 0.2   # fixed smoothing factor (sensitivity: try 0.05-0.5)

POLICIES = (
    "random",
    "round-robin",
    "ewma",
    "belief-only",
    "state-aware",
    "oracle",
)

DELTA_P_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NUM_SEEDS = 50
NUM_NODES = 10
ROUNDS_PER_NODE = 200
BASE_DELIVERY_PROB = 0.75

# Shared GOOD/BAD channel dynamics for both gateways (only the resulting
# GOOD-state SHARE differs between them, via p_good_to_bad below).
BASE_GE_PARAMS = GEParams(
    p_good_to_bad=0.05,
    p_bad_to_good=0.3,
    loss_prob_good=0.03,
    loss_prob_bad=0.5,
)


@dataclass
class Round:
    node_id: int
    gw_belief: dict
    gw_true_prob: dict
    gw_delivered: dict


def _gateway2_params(delta_p, base=BASE_GE_PARAMS):
    """
    Increase gateway 2's p_good_to_bad (and/or reduce p_bad_to_good) so
    its long-run GOOD-state share is lower than gateway 1's, producing a
    persistent belief/true-probability asymmetry of roughly delta_p.

    Stationary P(GOOD) = p_bad_to_good / (p_good_to_bad + p_bad_to_good).
    We solve for a new p_good_to_bad that shifts stationary P(GOOD) down
    by approximately delta_p, holding p_bad_to_good fixed.
    """
    stat_good_1 = base.p_bad_to_good / (base.p_good_to_bad + base.p_bad_to_good)
    target_good_2 = max(0.05, min(0.95, stat_good_1 - delta_p))

    # stat_good_2 = p_bad_to_good / (p_good_to_bad_2 + p_bad_to_good)
    # => p_good_to_bad_2 = p_bad_to_good * (1/stat_good_2 - 1)
    p_good_to_bad_2 = base.p_bad_to_good * (1.0 / target_good_2 - 1.0)

    return GEParams(
        p_good_to_bad=p_good_to_bad_2,
        p_bad_to_good=base.p_bad_to_good,
        loss_prob_good=base.loss_prob_good,
        loss_prob_bad=base.loss_prob_bad,
    )


def simulate_rounds(delta_p, num_nodes=NUM_NODES, rounds_per_node=ROUNDS_PER_NODE,
                     seed=42, base_delivery_prob=BASE_DELIVERY_PROB):
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    gw_params = {
        1: BASE_GE_PARAMS,
        2: _gateway2_params(delta_p),
    }

    channels = {}
    hmms = {}
    for n in range(num_nodes):
        for g in (1, 2):
            channels[(n, g)] = JammedGEChannel(gw_params[g], rng, jam_severity_db=0.0)
            hmms[(n, g)] = ChannelHMM()

    rounds = []
    for n in range(num_nodes):
        for _ in range(rounds_per_node):
            gw_belief = {}
            gw_true_prob = {}
            gw_delivered = {}

            for g in (1, 2):
                _, quality_db, state = channels[(n, g)].step()
                belief = hmms[(n, g)].update(quality_db)
                gw_belief[g] = belief

                delivery_p = (
                    base_delivery_prob if state == "GOOD"
                    else max(0.05, base_delivery_prob - 0.5)
                )
                gw_true_prob[g] = delivery_p
                gw_delivered[g] = py_rng.random() < delivery_p

            rounds.append(Round(
                node_id=n,
                gw_belief=gw_belief,
                gw_true_prob=gw_true_prob,
                gw_delivered=gw_delivered,
            ))

    return rounds


def run_policy(policy, rounds, seed):
    rng = random.Random(seed)
    rr_next = {}
    # EWMA state: per (node, gateway) running estimate + observation count.
    ewma_est = {}
    ewma_n = {}

    delivered_count = 0

    for r in rounds:
        gateways = (1, 2)

        if policy == "random":
            g = rng.choice(gateways)

        elif policy == "round-robin":
            i = rr_next.get(r.node_id, 0)
            g = gateways[i % 2]
            rr_next[r.node_id] = i + 1

        elif policy in ("belief-only", "state-aware"):
            g = max(gateways, key=lambda gw: r.gw_belief[gw])

        elif policy == "ewma":
            # Confirmed-outcome EWMA: rank by running observed success
            # rate per (node, gateway), defaulting to 0.5 with no history.
            def score(gw):
                return ewma_est.get((r.node_id, gw), 0.5)
            best = max(score(gw) for gw in gateways)
            g = rng.choice([gw for gw in gateways if score(gw) == best])

        elif policy == "oracle":
            g = max(gateways, key=lambda gw: r.gw_true_prob[gw])

        else:
            raise ValueError(policy)

        success = r.gw_delivered[g]
        delivered_count += int(success)

        if policy == "ewma":
            key = (r.node_id, g)
            prev = ewma_est.get(key, 0.5)
            ewma_est[key] = (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * float(success)

    return delivered_count / len(rounds)


def main():
    rows = []

    print("=" * 78)
    print("SAF-LoRa GATEWAY-ASYMMETRY SWEEP (Experiment S1)")
    print("=" * 78)
    print(f"Delta_p grid: {DELTA_P_GRID}")
    print(f"Seeds: {NUM_SEEDS}, Nodes: {NUM_NODES}, Rounds/node: {ROUNDS_PER_NODE}")
    print()

    for delta_p in DELTA_P_GRID:
        print(f"--- delta_p = {delta_p:.2f} ---")

        for policy in POLICIES:
            values = []
            for seed in range(NUM_SEEDS):
                rounds = simulate_rounds(delta_p, seed=seed)
                value = run_policy(policy, rounds, seed=seed + 100000)
                values.append(value)
                rows.append({
                    "delta_p": delta_p,
                    "policy": policy,
                    "seed": seed,
                    "delivery_rate": value,
                })

            mean = st.mean(values)
            sd = st.stdev(values) if len(values) > 1 else 0.0
            print(f"{policy:12s}: {100*mean:6.2f}% +/- {100*sd:5.2f}%")
        print()

    csv_path = Path("gateway_asymmetry_results.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["delta_p", "policy", "seed", "delivery_rate"])
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 78)
    print("STATE-AWARE GAIN OVER ROUND-ROBIN vs. DELTA_P")
    print("=" * 78)
    for delta_p in DELTA_P_GRID:
        sa = st.mean([r["delivery_rate"] for r in rows
                      if r["delta_p"] == delta_p and r["policy"] == "state-aware"])
        rr = st.mean([r["delivery_rate"] for r in rows
                      if r["delta_p"] == delta_p and r["policy"] == "round-robin"])
        gain = sa - rr
        print(f"delta_p={delta_p:.2f}  state-aware={100*sa:6.2f}%  round-robin={100*rr:6.2f}%  gain={100*gain:+6.2f}pp")

    print()
    print(f"Saved: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
