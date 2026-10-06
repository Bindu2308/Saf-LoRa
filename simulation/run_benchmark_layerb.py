"""
SAF-LoRa Benchmark Layer B: local quantitative policy comparison
(Fig. 12 per the Figure/Benchmark Plan).

Compares all Table-5 policies under an interference-severity sweep on
IDENTICAL channel realizations per (seed, severity):
    Random, Round-robin, RSSI/SNR-greedy, EWMA, Belief-only,
    SAF-LoRa (state-aware), Oracle.

Also computes the normalized oracle-gap recovery:
    G_norm = (R_policy - R_random) / (R_oracle - R_random)

Per the plan: this is an internal-baseline comparison on OUR OWN
simulator/testbed, not a claim of beating any external Transactions
paper's reported number.

Run:
    python run_benchmark_layerb.py
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
    "snr-greedy",
    "ewma",
    "belief-only",
    "state-aware",
    "oracle",
)

SEVERITY_DB_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
NUM_SEEDS = 50
NUM_NODES = 10
ROUNDS_PER_NODE = 200
BASE_DELIVERY_PROB = 0.75
JAMMED_GATEWAY = 1  # gateway 1 carries the interference severity

BASE_GE_PARAMS = GEParams(
    p_good_to_bad=0.05,
    p_bad_to_good=0.3,
    loss_prob_good=0.03,
    loss_prob_bad=0.5,
)


@dataclass
class Round:
    node_id: int
    gw_snr: dict          # raw quality_db reading this round (or a floor value if LOST)
    gw_belief: dict        # HMM belief
    gw_true_prob: dict     # latent true delivery probability (oracle-only)
    gw_delivered: dict     # actual delivery outcome this round


def simulate_rounds(severity_db, num_nodes=NUM_NODES, rounds_per_node=ROUNDS_PER_NODE,
                     seed=42, base_delivery_prob=BASE_DELIVERY_PROB):
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    channels = {}
    hmms = {}
    channel_sev = {}
    for n in range(num_nodes):
        for g in (1, 2):
            sev = severity_db if g == JAMMED_GATEWAY else 0.0
            channels[(n, g)] = JammedGEChannel(BASE_GE_PARAMS, rng, jam_severity_db=sev)
            hmms[(n, g)] = ChannelHMM()
            channel_sev[(n, g)] = sev

    rounds = []
    for n in range(num_nodes):
        for _ in range(rounds_per_node):
            gw_snr = {}
            gw_belief = {}
            gw_true_prob = {}
            gw_delivered = {}

            for g in (1, 2):
                _, quality_db, state = channels[(n, g)].step()
                belief = hmms[(n, g)].update(quality_db)

                # Raw SNR reading for the snr-greedy baseline: use a low
                # floor value when nothing arrived (LOST), since a
                # SNR-greedy policy literally has no reading to rank on
                # in that case -- this makes LOST look unattractive,
                # which is the correct greedy behavior.
                gw_snr[g] = quality_db if quality_db is not None else -99.0
                gw_belief[g] = belief

                delivery_p = (base_delivery_prob if state == "GOOD"
                              else max(0.05, base_delivery_prob - 0.5))
                delivery_p = max(0.0, min(1.0, delivery_p - channel_sev[(n, g)] * 0.04))
                gw_true_prob[g] = delivery_p
                gw_delivered[g] = py_rng.random() < delivery_p

            rounds.append(Round(n, gw_snr, gw_belief, gw_true_prob, gw_delivered))
    return rounds


def run_policy(policy, rounds, seed):
    rng = random.Random(seed)
    rr_next = {}
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

        elif policy == "snr-greedy":
            # Raw instantaneous link statistic only -- no HMM belief.
            g = max(gateways, key=lambda gw: r.gw_snr[gw])

        elif policy in ("belief-only", "state-aware"):
            g = max(gateways, key=lambda gw: r.gw_belief[gw])

        elif policy == "ewma":
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
    print("SAF-LoRa BENCHMARK LAYER B (Fig. 12): LOCAL POLICY COMPARISON")
    print("=" * 78)
    print(f"Severity grid (dB): {SEVERITY_DB_GRID}")
    print(f"Seeds: {NUM_SEEDS}, Nodes: {NUM_NODES}, Rounds/node: {ROUNDS_PER_NODE}")
    print()

    for severity in SEVERITY_DB_GRID:
        print(f"--- severity = {severity:.1f} dB ---")

        for policy in POLICIES:
            values = []
            for seed in range(NUM_SEEDS):
                rounds = simulate_rounds(severity, seed=seed)
                value = run_policy(policy, rounds, seed=seed + 100000)
                values.append(value)
                rows.append({
                    "severity_db": severity,
                    "policy": policy,
                    "seed": seed,
                    "delivery_rate": value,
                })

            mean = st.mean(values)
            sd = st.stdev(values) if len(values) > 1 else 0.0
            print(f"{policy:12s}: {100*mean:6.2f}% +/- {100*sd:5.2f}%")
        print()

    csv_path = Path("benchmark_layerb_results.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["severity_db", "policy", "seed", "delivery_rate"])
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 78)
    print("NORMALIZED ORACLE-GAP RECOVERY: G_norm = (R_policy - R_random) / (R_oracle - R_random)")
    print("=" * 78)
    gnorm_rows = []
    for severity in SEVERITY_DB_GRID:
        r_random = st.mean([r["delivery_rate"] for r in rows
                             if r["severity_db"] == severity and r["policy"] == "random"])
        r_oracle = st.mean([r["delivery_rate"] for r in rows
                             if r["severity_db"] == severity and r["policy"] == "oracle"])
        denom = r_oracle - r_random

        print(f"severity={severity:.1f}dB  (R_random={100*r_random:.2f}%, R_oracle={100*r_oracle:.2f}%)")
        for policy in POLICIES:
            r_policy = st.mean([r["delivery_rate"] for r in rows
                                 if r["severity_db"] == severity and r["policy"] == policy])
            g_norm = (r_policy - r_random) / denom if denom != 0 else float("nan")
            print(f"    {policy:12s}: G_norm = {g_norm:6.3f}")
            gnorm_rows.append({
                "severity_db": severity,
                "policy": policy,
                "g_norm": g_norm,
            })

    gnorm_path = Path("benchmark_layerb_gnorm.csv")
    with gnorm_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["severity_db", "policy", "g_norm"])
        writer.writeheader()
        writer.writerows(gnorm_rows)

    print()
    print(f"Saved: {csv_path.resolve()}")
    print(f"Saved: {gnorm_path.resolve()}")


if __name__ == "__main__":
    main()
