"""
isac_lora_sim/ack_scheduler_sim.py

Real integration: ACK-scheduling simulation driven by the ACTUAL
Gilbert-Elliott channel model and HMM belief estimator (gilbert_elliott.py,
hmm_estimator.py), not a synthetic stand-in. This is what should
generate any numbers that go in the paper.
"""

import random
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from topology import make_topology
from gilbert_elliott import GEChannel, GEParams, distance_scaled_params
from hmm_estimator import ChannelHMM

ACK_AIRTIME_S = 0.04634
POLICIES = ("broadcast", "random", "round-robin", "ewma", "state-aware")


class DownlinkBelief:
    def __init__(self, alpha: float = 0.2, prior: float = 0.5):
        self.alpha = alpha
        self.prior = prior
        self._belief = {}

    def get(self, node_id, gateway_id, uplink_prior=None):
        key = (node_id, gateway_id)
        if key in self._belief:
            return self._belief[key]
        return uplink_prior if uplink_prior is not None else self.prior

    def record_outcome(self, node_id, gateway_id, good: bool):
        key = (node_id, gateway_id)
        cur = self._belief.get(key, self.prior)
        self._belief[key] = (1 - self.alpha) * cur + self.alpha * (1.0 if good else 0.0)


class AirtimeBudget:
    def __init__(self, duty_limit: float = 0.10, window_s: float = 3600.0):
        self.duty_limit = duty_limit
        self.window_s = window_s
        self._sent = defaultdict(deque)

    def _prune(self, gateway_id, now):
        dq = self._sent[gateway_id]
        cutoff = now - self.window_s
        while dq and dq[0] < cutoff:
            dq.popleft()

    def can_send(self, gateway_id, now):
        self._prune(gateway_id, now)
        used = len(self._sent[gateway_id]) * ACK_AIRTIME_S
        return (used + ACK_AIRTIME_S) <= (self.duty_limit * self.window_s)

    def record_send(self, gateway_id, now):
        self._sent[gateway_id].append(now)


class DownlinkScheduler:
    def __init__(self, policy, gateway_ids=(1, 2), duty_limit=0.10, seed=None):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.policy = policy
        self.gateway_ids = tuple(gateway_ids)
        self.belief = DownlinkBelief()
        self.budget = AirtimeBudget(duty_limit=duty_limit)
        self._rng = random.Random(seed)
        self._rr_next = defaultdict(int)
        self.stats = defaultdict(int)

    def _argmax_random_ties(self, candidates, score_fn, tol=1e-9):
        scored = [(score_fn(c), c) for c in candidates]
        best = max(s for s, _ in scored)
        tied = [c for s, c in scored if abs(s - best) <= tol]
        if len(tied) > 1:
            self.stats["ties_broken_randomly"] += 1
        return self._rng.choice(tied)

    def select(self, node_id, available, now, uplink_prior_fn=None):
        if not available:
            self.stats["no_gateway_available"] += 1
            return []
        affordable = [g for g in available if self.budget.can_send(g, now)]
        if not affordable:
            self.stats["budget_blocked"] += 1
            return []
        if self.policy == "broadcast":
            chosen = affordable
        elif self.policy == "random":
            chosen = [self._rng.choice(affordable)]
        elif self.policy == "round-robin":
            idx = self._rr_next[node_id] % len(affordable)
            self._rr_next[node_id] += 1
            chosen = [affordable[idx]]
        elif self.policy == "ewma":
            chosen = [self._argmax_random_ties(affordable, lambda g: self.belief.get(node_id, g))]
        else:
            def score(g):
                prior = uplink_prior_fn(node_id, g) if uplink_prior_fn else None
                return self.belief.get(node_id, g, uplink_prior=prior)
            chosen = [self._argmax_random_ties(affordable, score)]
        for g in chosen:
            self.budget.record_send(g, now)
        self.stats["acks_scheduled"] += 1
        return chosen


@dataclass
class RealTelegramAckEvent:
    node_id: int
    timestamp: float
    gw_belief: dict   # {gateway_id: ChannelHMM.p_good at this moment} -- REAL HMM belief
    gw_true_state: dict  # {gateway_id: "GOOD"/"BAD"} -- ground truth, for scoring the belief


def real_telegram_stream(num_nodes=10, num_gateways=2, area_size_m=1300.0,
                          telegrams_per_node=200, jammed_gateway=None,
                          jam_severity=0.0, seed=42):
    """Drives the ACTUAL GEChannel + ChannelHMM per (node, gateway) link,
    exactly as strategies.py's evaluate_hmm_state_aware() does, and
    yields one ACK-decision event per telegram with the real HMM belief
    at that moment as the state-aware/ewma policies' input signal.

    jammed_gateway / jam_severity: temporarily raises that gateway's
    loss_prob_bad and loss_prob_good for ALL nodes, modeling a nearby
    interference source -- the simulation analogue of the hardware
    campaign's controlled interferer near one gateway.
    """
    rng = np.random.RandomState(seed)
    topo = make_topology(num_nodes, num_gateways, area_size_m, seed=seed)

    channels, hmms = {}, {}
    for node_id in range(num_nodes):
        for gw_id in range(1, num_gateways + 1):
            dist = topo.distance(node_id, gw_id - 1)
            params = distance_scaled_params(dist)
            if gw_id == jammed_gateway:
                params = GEParams(
                    p_good_to_bad=params.p_good_to_bad,
                    p_bad_to_good=params.p_bad_to_good,
                    loss_prob_good=min(params.loss_prob_good + jam_severity, 0.95),
                    loss_prob_bad=min(params.loss_prob_bad + jam_severity, 0.98),
                )
            channels[(node_id, gw_id)] = GEChannel(params, rng)
            hmms[(node_id, gw_id)] = ChannelHMM()

    t = 0.0
    for node_id in range(num_nodes):
        for _ in range(telegrams_per_node):
            gw_belief, gw_true = {}, {}
            for gw_id in range(1, num_gateways + 1):
                outcome, quality_db, state = channels[(node_id, gw_id)].step()
                hmm = hmms[(node_id, gw_id)]
                gw_belief[gw_id] = hmm.update(quality_db)
                gw_true[gw_id] = state
            yield RealTelegramAckEvent(node_id=node_id, timestamp=t,
                                        gw_belief=gw_belief, gw_true_state=gw_true)
            t += rng.uniform(3.0, 8.0)


def run_ack_scheduling_sweep(policy, telegram_stream, duty_limit=0.10, seed=42):
    """Runs one policy over a REAL telegram stream. gw_counts is the
    acknowledgment allocation; also returns HMM-vs-truth agreement stats
    (the validation the reviewers asked for, computable here because the
    simulation -- unlike the lost hardware log -- has ground truth by
    construction)."""
    scheduler = DownlinkScheduler(policy=policy, duty_limit=duty_limit, seed=seed)
    gw_counts = defaultdict(int)
    hmm_correct, hmm_total = 0, 0

    for event in telegram_stream:
        available = list(event.gw_belief.keys())
        uplink_prior_fn = lambda n, g: event.gw_belief[g]
        chosen = scheduler.select(event.node_id, available, event.timestamp,
                                   uplink_prior_fn=uplink_prior_fn)
        for g in chosen:
            gw_counts[g] += 1

        for g in event.gw_belief:
            belief_says_good = event.gw_belief[g] >= 0.5
            truth_is_good = event.gw_true_state[g] == "GOOD"
            hmm_correct += int(belief_says_good == truth_is_good)
            hmm_total += 1
            scheduler.belief.record_outcome(event.node_id, g, belief_says_good)

    hmm_accuracy = hmm_correct / hmm_total if hmm_total else 0.0
    return dict(gw_counts), dict(scheduler.stats), hmm_accuracy


def gateway_share(gw_counts, reference_gateway):
    total = sum(gw_counts.values())
    return 100.0 * gw_counts.get(reference_gateway, 0) / total if total else 0.0
