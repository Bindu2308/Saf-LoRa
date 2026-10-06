"""
SAF-LoRa gateway-diversity scaling experiment.
"""

import csv
import random
import statistics as st
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from topology import make_topology
from gilbert_elliott import distance_scaled_params
from gilbert_elliott_jam_patch import JammedGEChannel
from hmm_estimator import ChannelHMM


POLICIES = (
    "random",
    "round-robin",
    "belief-only",
    "state-aware",
    "oracle",
)

GATEWAY_COUNTS = [1, 2, 3, 4, 5, 6]
SEVERITY_DB = 1.5
NUM_SEEDS = 50
NUM_NODES = 10
ROUNDS_PER_NODE = 200
AREA_SIZE_M = 1300.0


@dataclass
class Round:
    node_id: int
    gw_snr: dict
    gw_belief: dict
    gw_true_prob: dict
    gw_delivered: dict


def simulate_rounds(
    num_nodes=NUM_NODES,
    num_gateways=2,
    rounds_per_node=ROUNDS_PER_NODE,
    jammed_gateway=1,
    jam_severity_db=SEVERITY_DB,
    seed=42,
    base_delivery_prob=0.75,
):
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    topo = make_topology(
        num_nodes,
        num_gateways,
        AREA_SIZE_M,
        seed=seed,
    )

    channels = {}
    hmms = {}
    channel_sev = {}

    for n in range(num_nodes):
        for g in range(1, num_gateways + 1):
            dist = topo.distance(n, g - 1)
            params = distance_scaled_params(dist)

            sev = jam_severity_db if g == jammed_gateway else 0.0

            channels[(n, g)] = JammedGEChannel(
                params,
                rng,
                jam_severity_db=sev,
            )
            hmms[(n, g)] = ChannelHMM()
            channel_sev[(n, g)] = sev

    rounds = []

    for n in range(num_nodes):
        for _ in range(rounds_per_node):
            gw_snr = {}
            gw_belief = {}
            gw_true_prob = {}
            gw_delivered = {}

            for g in range(1, num_gateways + 1):
                _, quality_db, state = channels[(n, g)].step()

                belief = hmms[(n, g)].update(quality_db)

                gw_snr[g] = (
                    quality_db if quality_db is not None else -99.0
                )
                gw_belief[g] = belief

                delivery_p = (
                    base_delivery_prob
                    if state == "GOOD"
                    else max(0.05, base_delivery_prob - 0.5)
                )

                delivery_p = max(
                    0.0,
                    min(
                        1.0,
                        delivery_p - channel_sev[(n, g)] * 0.04,
                    ),
                )

                gw_true_prob[g] = delivery_p
                gw_delivered[g] = py_rng.random() < delivery_p

            rounds.append(
                Round(
                    node_id=n,
                    gw_snr=gw_snr,
                    gw_belief=gw_belief,
                    gw_true_prob=gw_true_prob,
                    gw_delivered=gw_delivered,
                )
            )

    return rounds


def run_policy(policy, rounds, num_gateways, seed):
    rng = random.Random(seed)
    rr_next = {}
    delivered_count = 0

    for r in rounds:
        gateways = list(range(1, num_gateways + 1))

        if policy == "random":
            g = rng.choice(gateways)

        elif policy == "round-robin":
            i = rr_next.get(r.node_id, 0)
            g = gateways[i % num_gateways]
            rr_next[r.node_id] = i + 1

        elif policy in ("belief-only", "state-aware"):
            g = max(gateways, key=lambda gw: r.gw_belief[gw])

        elif policy == "oracle":
            g = max(gateways, key=lambda gw: r.gw_true_prob[gw])

        else:
            raise ValueError(policy)

        delivered_count += int(r.gw_delivered[g])

    return delivered_count / len(rounds)


def main():
    rows = []

    print("=" * 78)
    print("SAF-LoRa GATEWAY-DIVERSITY SCALING EXPERIMENT")
    print("=" * 78)
    print(f"Gateways: {GATEWAY_COUNTS}")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"Nodes: {NUM_NODES}")
    print(f"Rounds/node: {ROUNDS_PER_NODE}")
    print(f"Interference severity: {SEVERITY_DB:.1f} dB")
    print()

    for n_gw in GATEWAY_COUNTS:
        print(f"--- {n_gw} gateway(s) ---")

        for policy in POLICIES:
            values = []

            for seed in range(NUM_SEEDS):
                jammed_gateway = 1

                rounds = simulate_rounds(
                    num_gateways=n_gw,
                    jammed_gateway=jammed_gateway,
                    seed=seed,
                )

                value = run_policy(
                    policy,
                    rounds,
                    n_gw,
                    seed=seed + 100000,
                )
                values.append(value)

                rows.append(
                    {
                        "gateways": n_gw,
                        "policy": policy,
                        "seed": seed,
                        "delivery_rate": value,
                    }
                )

            mean = st.mean(values)
            sd = st.stdev(values) if len(values) > 1 else 0.0

            print(
                f"{policy:12s}: "
                f"{100*mean:6.2f}% +/- {100*sd:5.2f}%"
            )

        print()

    csv_path = Path("gateway_scaling_results.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gateways",
                "policy",
                "seed",
                "delivery_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 78)
    print("MEAN DELIVERY RATE")
    print("=" * 78)
    print(
        f"{'GW':>3}  "
        f"{'Random':>10}  "
        f"{'RR':>10}  "
        f"{'Belief':>10}  "
        f"{'SAF-LoRa':>10}  "
        f"{'Oracle':>10}"
    )

    for n_gw in GATEWAY_COUNTS:
        vals = {}
        for policy in POLICIES:
            x = [
                r["delivery_rate"]
                for r in rows
                if r["gateways"] == n_gw
                and r["policy"] == policy
            ]
            vals[policy] = st.mean(x)

        print(
            f"{n_gw:3d}  "
            f"{100*vals['random']:9.2f}%  "
            f"{100*vals['round-robin']:9.2f}%  "
            f"{100*vals['belief-only']:9.2f}%  "
            f"{100*vals['state-aware']:9.2f}%  "
            f"{100*vals['oracle']:9.2f}%"
        )

    print()
    print(f"Saved: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
