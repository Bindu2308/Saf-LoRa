"""
receiver/scheduling/downlink_scheduler.py

Replaces the broadcast-to-both-gateways ACK path with a policy-driven
choice of ONE gateway per acknowledgment, subject to a rolling airtime
budget.

Why this exists
---------------
The previous path called send_downlink() on both gateways for every ACK,
so each acknowledgment consumed downlink airtime twice. That is not just
wasteful: it means no gateway-selection decision existed, so no state
information could influence anything on the downlink.

Measured justification for state-aware selection: on 5,372 joined
telegrams, uplink HMM belief predicts downlink ACK success with
AUC = 0.619 (avg SNR 0.610, min SNR 0.565, avg RSSI 0.564). Modest, but
well clear of chance, so uplink observations are a usable cold-start
prior for a downlink belief.

IMPORTANT -- what the belief can and cannot learn from
------------------------------------------------------
GatewayServer.send_downlink() returns True when the TCP write to the
gateway succeeded. It does NOT mean the gateway transmitted over LoRa,
and it certainly does not mean the node received the ACK. So a True
return is not evidence of downlink success and must never be fed to the
belief as a positive outcome.

Until nodes report ACK receipt upstream, the belief therefore runs on
the uplink prior alone. record_receipt_report() is the hook for real
outcomes once that firmware exists; nothing else should update the
belief.
"""

import random
import time
from collections import defaultdict, deque

POLICIES = ("state-aware", "ewma", "round-robin", "random", "broadcast")

# Airtime of one ACK at SF7/BW125, 15-byte implicit-header packet.
ACK_AIRTIME_S = 0.04634


class DownlinkBelief:
    """Per (node, gateway) estimate of P(ACK will be delivered).

    Seeded from the uplink HMM belief, then updated only by genuine
    receipt reports from the node. An EWMA rather than a filter, because
    with no observation model for the downlink there is nothing
    principled to filter with -- and claiming otherwise would overstate
    what this is.
    """

    def __init__(self, alpha: float = 0.2, prior: float = 0.5):
        self.alpha = alpha
        self.prior = prior
        self._belief = {}
        self._observations = defaultdict(int)

    def get(self, node_id: int, gateway_id: int, uplink_prior: float = None) -> float:
        key = (node_id, gateway_id)
        if key in self._belief:
            return self._belief[key]
        # Cold start: uplink belief if available (AUC 0.619), else neutral.
        return uplink_prior if uplink_prior is not None else self.prior

    def record_receipt_report(self, node_id: int, gateway_id: int, delivered: bool):
        """Call ONLY with a genuine node-side receipt report. Do not call
        with send_downlink()'s return value -- that reports a TCP write,
        not a radio delivery."""
        key = (node_id, gateway_id)
        cur = self._belief.get(key, self.prior)
        self._belief[key] = (1 - self.alpha) * cur + self.alpha * (1.0 if delivered else 0.0)
        self._observations[key] += 1

    def observation_count(self, node_id: int, gateway_id: int) -> int:
        return self._observations[(node_id, gateway_id)]


class AirtimeBudget:
    """Rolling per-gateway transmit-airtime cap over a sliding window.

    Enforces a duty-cycle ceiling without needing discrete scheduling
    rounds: each candidate ACK is admitted only if the gateway's airtime
    over the trailing window stays under the limit.
    """

    def __init__(self, duty_limit: float = 0.10, window_s: float = 3600.0):
        self.duty_limit = duty_limit
        self.window_s = window_s
        self._sent = defaultdict(deque)   # gateway_id -> deque of timestamps

    def _prune(self, gateway_id: int, now: float):
        dq = self._sent[gateway_id]
        cutoff = now - self.window_s
        while dq and dq[0] < cutoff:
            dq.popleft()

    def can_send(self, gateway_id: int, now: float = None) -> bool:
        now = now or time.time()
        self._prune(gateway_id, now)
        used = len(self._sent[gateway_id]) * ACK_AIRTIME_S
        return (used + ACK_AIRTIME_S) <= (self.duty_limit * self.window_s)

    def record_send(self, gateway_id: int, now: float = None):
        self._sent[gateway_id].append(now or time.time())

    def utilization(self, gateway_id: int, now: float = None) -> float:
        now = now or time.time()
        self._prune(gateway_id, now)
        return (len(self._sent[gateway_id]) * ACK_AIRTIME_S) / self.window_s


class DownlinkScheduler:
    """Chooses which gateway acknowledges a given node.

    policy:
      state-aware  highest downlink belief (uplink-primed)
      ewma         highest recent ACK-delivery EWMA, ignoring uplink
      round-robin  alternate gateways per node
      random       uniform over connected gateways
      broadcast    both gateways -- reproduces the previous behavior,
                   retained as the experimental baseline
    """

    def __init__(self, policy: str = "state-aware", gateway_ids=(1, 2),
                 duty_limit: float = 0.10, seed: int = None):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.policy = policy
        self.gateway_ids = tuple(gateway_ids)
        self.belief = DownlinkBelief()
        self.budget = AirtimeBudget(duty_limit=duty_limit)
        self._rng = random.Random(seed)
        self._rr_next = defaultdict(int)
        self.stats = defaultdict(int)

    def _argmax_random_ties(self, candidates, score_fn, tol: float = 1e-9):
        """argmax that breaks ties uniformly at random.

        Python's max() returns the FIRST maximal element, so any policy
        whose scores are equal -- which is every policy before it has
        observations -- silently collapses onto the lowest-numbered
        gateway. That happened in the first campaign: the ewma policy
        sent 352 consecutive ACKs through gateway 1 and none through
        gateway 2, which looks like a decision but is just list order.
        """
        scored = [(score_fn(c), c) for c in candidates]
        best = max(s for s, _ in scored)
        tied = [c for s, c in scored if abs(s - best) <= tol]
        if len(tied) > 1:
            self.stats["ties_broken_randomly"] += 1
        return self._rng.choice(tied)

    def select(self, node_id: int, available: list, uplink_prior_fn=None) -> list:
        """Returns the gateway ids that should transmit this ACK.

        available: gateway ids with a live connection right now.
        uplink_prior_fn: optional (node_id, gateway_id) -> float, used
            only to cold-start a (node, gateway) pair with no receipt
            history.
        """
        if not available:
            self.stats["no_gateway_available"] += 1
            return []

        affordable = [g for g in available if self.budget.can_send(g)]
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
            # Deliberately ignores the uplink prior: this is the baseline
            # that answers "does uplink state add anything over simply
            # tracking recent downlink outcomes?"
            chosen = [self._argmax_random_ties(
                affordable, lambda g: self.belief.get(node_id, g))]
        else:  # state-aware
            def score(g):
                prior = uplink_prior_fn(node_id, g) if uplink_prior_fn else None
                return self.belief.get(node_id, g, uplink_prior=prior)
            chosen = [self._argmax_random_ties(affordable, score)]

        for g in chosen:
            self.budget.record_send(g)
            self.stats[f"sent_gw{g}"] += 1
        self.stats["acks_scheduled"] += 1
        return chosen

    def summary(self) -> dict:
        out = dict(self.stats)
        out["policy"] = self.policy
        for g in self.gateway_ids:
            out[f"utilization_gw{g}"] = round(self.budget.utilization(g), 5)
        return out
