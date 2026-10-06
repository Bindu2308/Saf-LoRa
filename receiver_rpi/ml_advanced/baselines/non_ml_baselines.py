"""
receiver/ml_advanced/baselines/non_ml_baselines.py

The essential comparison points every reviewer will want to see: does
the complex ML actually beat simple heuristics? These operate directly
on graph_observations.csv rows -- no training needed, pure Python.
"""

import random


def single_gateway_strategy(edges: list, gateway_id: int = 1) -> set:
    return {int(e["fragment_id"]) for e in edges if int(e["gateway_id"]) == gateway_id}


def random_gateway_strategy(edges: list, seed: int = None) -> set:
    rng = random.Random(seed)
    by_fragment = {}
    for e in edges:
        by_fragment.setdefault(int(e["fragment_id"]), []).append(e)
    recovered = set()
    for frag_id, candidates in by_fragment.items():
        rng.choice(candidates)
        recovered.add(frag_id)
    return recovered


def best_metric_strategy(edges: list, metric_key: str) -> set:
    by_fragment = {}
    for e in edges:
        by_fragment.setdefault(int(e["fragment_id"]), []).append(e)
    recovered = set()
    for frag_id, candidates in by_fragment.items():
        max(candidates, key=lambda e: float(e[metric_key]))
        recovered.add(frag_id)
    return recovered


def best_rssi_strategy(edges: list) -> set:
    return best_metric_strategy(edges, "rssi_db")


def best_snr_strategy(edges: list) -> set:
    return best_metric_strategy(edges, "snr_db")


def state_blind_hard_combine_strategy(edges: list) -> set:
    """What your system does TODAY with no ML at all (cooperation/hard_combiner.py
    with reliability_multiplier=1.0): use whichever gateway reported each
    fragment, cooperatively across all gateways -- the real 'state-blind'
    baseline from your abstract, not a strawman."""
    return {int(e["fragment_id"]) for e in edges}


STRATEGIES = {
    "single_gateway_gw1": lambda edges: single_gateway_strategy(edges, gateway_id=1),
    "single_gateway_gw2": lambda edges: single_gateway_strategy(edges, gateway_id=2),
    "random_gateway": random_gateway_strategy,
    "best_rssi": best_rssi_strategy,
    "best_snr": best_snr_strategy,
    "state_blind_cooperative": state_blind_hard_combine_strategy,
}


def evaluate_strategy(telegram_groups: dict, strategy_fn, total_fragments_key: str = "total_fragments") -> dict:
    """telegram_groups: {(node_id, telegram_id): [edge_dict, ...]}"""
    reconstructed = 0
    frac_sum = 0.0
    for key, edges in telegram_groups.items():
        total_fragments = int(edges[0][total_fragments_key])
        recovered = strategy_fn(edges)
        frac = len(recovered) / max(total_fragments, 1)
        frac_sum += min(frac, 1.0)
        if len(recovered) >= total_fragments:
            reconstructed += 1

    n = len(telegram_groups)
    return {
        "telegram_reconstruction_rate": reconstructed / n if n else 0.0,
        "avg_fragment_recovery_frac": frac_sum / n if n else 0.0,
        "num_telegrams": n,
    }
