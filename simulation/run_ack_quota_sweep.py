"""
SAF-LoRa ACK-quota sweep, all five policies (Experiment S2).

Extends the hardware Table X result (which reports only the state-aware
policy) to a full cross-policy comparison: as the per-gateway ACK quota
tightens, does policy choice affect (a) the ACK-rejection rate and
(b) the confirmed-delivery rate among ACKs that ARE sent?

Mirrors Algorithm 2's admissible-set restriction: at each round the
policy may only choose from gateways whose quota is not yet exhausted
for the current window; if no gateway has remaining budget, the ACK
candidate is REJECTED (no ACK sent this round) rather than forced onto
an over-budget gateway.

Fixed at a moderate gateway asymmetry (delta_p = 0.20) reusing the same
channel construction as run_gateway_asymmetry.py, so scarcity effects
are evaluated under conditions where gateway choice actually matters.

Run:
    python run_ack_quota_sweep.py
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

POLICIES = ("random", "round-robin", "ewma", "belief-only", "state-aware")

# ACKs per window; "unrestricted" modeled as a very large number.
QUOTA_GRID = [10_000, 30, 15, 8, 4, 2]
QUOTA_LABELS = {10_000: "Unrestricted"}
WINDOW_ROUNDS = 60   # rounds per quota-reset window (abstracts the 60s window in Table X)

DELTA_P = 0.20
NUM_SEEDS = 50
NUM_NODES = 10
ROUNDS_PER_NODE = 200
BASE_DELIVERY_PROB = 0.75

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
    stat_good_1 = base.p_bad_to_good / (base.p_good_to_bad + base.p_bad_to_good)
    target_good_2 = max(0.05, min(0.95, stat_good_1 - delta_p))
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

    gw_params = {1: BASE_GE_PARAMS, 2: _gateway2_params(delta_p)}

    channels = {}
    hmms = {}
    for n in range(num_nodes):
        for g in (1, 2):
            channels[(n, g)] = JammedGEChannel(gw_params[g], rng, jam_severity_db=0.0)
            hmms[(n, g)] = ChannelHMM()

    # FIXED: build rounds TIME-INTERLEAVED across nodes (all nodes'
    # step-t round before any node's step-(t+1) round), not node-major.
    # The previous node-major ordering meant every WINDOW_ROUNDS-sized
    # quota window was dominated by a single node's traffic (since
    # ROUNDS_PER_NODE >> WINDOW_ROUNDS), so no real multi-node
    # contention for the shared per-window ACK budget ever occurred --
    # that is why reject rate came out bit-for-bit identical across
    # every policy with exactly 0.00% SD. Interleaving makes each
    # window genuinely reflect simultaneous demand from many nodes.
    rounds = []
    for _ in range(rounds_per_node):
        for n in range(num_nodes):
            gw_belief = {}
            gw_true_prob = {}
            gw_delivered = {}
            for g in (1, 2):
                _, quality_db, state = channels[(n, g)].step()
                belief = hmms[(n, g)].update(quality_db)
                gw_belief[g] = belief
                delivery_p = (base_delivery_prob if state == "GOOD"
                              else max(0.05, base_delivery_prob - 0.5))
                gw_true_prob[g] = delivery_p
                gw_delivered[g] = py_rng.random() < delivery_p
            rounds.append(Round(n, gw_belief, gw_true_prob, gw_delivered))
    return rounds


def run_policy_with_quota(policy, rounds, quota, seed):
    """
    Returns (reject_rate, confirmed_delivery_rate_among_sent).
    """
    rng = random.Random(seed)
    rr_next = {}
    ewma_est = {}
    ewma_n = {}

    sent = 0
    rejected = 0
    delivered = 0

    window_used = {1: 0, 2: 0}
    round_in_window = 0

    for r in rounds:
        if round_in_window >= WINDOW_ROUNDS:
            window_used = {1: 0, 2: 0}
            round_in_window = 0
        round_in_window += 1

        # Algorithm 2: admissible set = gateways with remaining budget.
        available = [g for g in (1, 2) if window_used[g] < quota]

        if not available:
            rejected += 1
            continue

        if policy == "random":
            g = rng.choice(available)

        elif policy == "round-robin":
            i = rr_next.get(r.node_id, 0)
            # Deterministic alternation restricted to the admissible set.
            ordered = available if (i % 2 == 0) else list(reversed(available))
            g = ordered[0]
            rr_next[r.node_id] = i + 1

        elif policy in ("belief-only", "state-aware"):
            g = max(available, key=lambda gw: r.gw_belief[gw])

        elif policy == "ewma":
            def score(gw):
                return ewma_est.get((r.node_id, gw), 0.5)
            best = max(score(gw) for gw in available)
            g = rng.choice([gw for gw in available if score(gw) == best])

        else:
            raise ValueError(policy)

        window_used[g] += 1
        sent += 1
        success = r.gw_delivered[g]
        delivered += int(success)

        if policy == "ewma":
            key = (r.node_id, g)
            prev = ewma_est.get(key, 0.5)
            ewma_est[key] = (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * float(success)

    total = sent + rejected
    reject_rate = rejected / total if total else 0.0
    delivery_rate = delivered / sent if sent else 0.0
    return reject_rate, delivery_rate


def main():
    rows = []

    print("=" * 78)
    print("SAF-LoRa ACK-QUOTA SWEEP, ALL POLICIES (Experiment S2)")
    print("=" * 78)
    print(f"Quota grid (ACKs/window): {QUOTA_GRID}")
    print(f"Window size: {WINDOW_ROUNDS} rounds")
    print(f"Fixed delta_p: {DELTA_P}")
    print(f"Seeds: {NUM_SEEDS}, Nodes: {NUM_NODES}, Rounds/node: {ROUNDS_PER_NODE}")
    print()

    for quota in QUOTA_GRID:
        label = QUOTA_LABELS.get(quota, str(quota))
        print(f"--- quota = {label} ---")

        for policy in POLICIES:
            reject_vals = []
            delivery_vals = []
            for seed in range(NUM_SEEDS):
                rounds = simulate_rounds(DELTA_P, seed=seed)
                reject_rate, delivery_rate = run_policy_with_quota(
                    policy, rounds, quota, seed=seed + 100000
                )
                reject_vals.append(reject_rate)
                delivery_vals.append(delivery_rate)
                rows.append({
                    "quota": quota,
                    "policy": policy,
                    "seed": seed,
                    "reject_rate": reject_rate,
                    "delivery_rate": delivery_rate,
                })

            rmean = st.mean(reject_vals)
            rsd = st.stdev(reject_vals) if len(reject_vals) > 1 else 0.0
            dmean = st.mean(delivery_vals)
            dsd = st.stdev(delivery_vals) if len(delivery_vals) > 1 else 0.0
            print(f"{policy:12s}: reject={100*rmean:6.2f}%+/-{100*rsd:4.2f}%  "
                  f"delivery(sent)={100*dmean:6.2f}%+/-{100*dsd:4.2f}%")
        print()

    csv_path = Path("ack_quota_sweep_results.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quota", "policy", "seed", "reject_rate", "delivery_rate"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
