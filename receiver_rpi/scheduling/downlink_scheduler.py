"""
receiver/scheduling/downlink_scheduler.py  (H1-H4 + snr-greedy + belief-only + logging)

Existing policies are unchanged. Added:
  snr-greedy   highest latest uplink SNR (needs uplink_snr_fn from caller)
  belief-only  highest uplink HMM belief (missing -> 0.5)
Optional per-decision CSV log via log_path.
"""
import os
import random
import time
from collections import defaultdict, deque

POLICIES = ("state-aware", "ewma", "round-robin", "random", "broadcast",
            "snr-greedy", "belief-only")
ACK_AIRTIME_S = 0.04634   # SF7/BW125, 15 B implicit header


class DownlinkBelief:
    def __init__(self, alpha=0.2, prior=0.5, k=5.0):
        self.alpha, self.prior, self.k = alpha, prior, k
        self._ewma = {}
        self._n = defaultdict(int)

    def ewma(self, node, gw):
        return self._ewma.get((node, gw), self.prior)

    def blended(self, node, gw, uplink_prior):
        n = self._n[(node, gw)]
        if uplink_prior is None:
            return self.ewma(node, gw)
        w = n / (n + self.k)
        return w * self.ewma(node, gw) + (1 - w) * uplink_prior

    def record_receipt_report(self, node, gw, delivered: bool):
        key = (node, gw)
        cur = self._ewma.get(key, self.prior)
        self._ewma[key] = (1 - self.alpha) * cur + self.alpha * (1.0 if delivered else 0.0)
        self._n[key] += 1

    def observation_count(self, node, gw):
        return self._n[(node, gw)]


class WindowCounter:
    def __init__(self, window_s):
        self.window_s = window_s
        self._t = defaultdict(deque)

    def count(self, gw, now):
        dq = self._t[gw]
        while dq and dq[0] < now - self.window_s:
            dq.popleft()
        return len(dq)

    def add(self, gw, now):
        self._t[gw].append(now)


class DownlinkScheduler:
    def __init__(self, policy="state-aware", gateway_ids=(1, 2), duty_limit=0.10,
                 seed=None, ack_quota=0, quota_window_s=60.0, airtime_window_s=3600.0,
                 log_path=None):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.policy = policy
        self.gateway_ids = tuple(gateway_ids)
        self.duty_limit = duty_limit
        self.ack_quota = ack_quota
        self.belief = DownlinkBelief()
        self._airtime = WindowCounter(airtime_window_s)
        self._quota = WindowCounter(quota_window_s)
        self._rng = random.Random(seed)
        self._rr = defaultdict(int)
        self.stats = defaultdict(int)
        self.last_reject_reason = None
        self.last_scores = {}
        self._log = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            new = not os.path.exists(log_path)
            self._log = open(log_path, "a", buffering=1)
            if new:
                self._log.write("t,policy,node,available,chosen,reject,scores,snr,belief\n")

    def _airtime_ok(self, g, now):
        used = (self._airtime.count(g, now) + 1) * ACK_AIRTIME_S
        return used <= self.duty_limit * self._airtime.window_s

    def _quota_ok(self, g, now):
        return self.ack_quota <= 0 or self._quota.count(g, now) < self.ack_quota

    def _argmax(self, cands, score):
        s = {g: score(g) for g in cands}
        self.last_scores = s
        best = max(s.values())
        tied = [g for g in cands if abs(s[g] - best) <= 1e-9]
        if len(tied) > 1:
            self.stats["ties_random"] += 1
        return self._rng.choice(tied)

    def _write_log(self, now, node, available, chosen, snr_fn, prior_fn):
        if not self._log:
            return
        def fmt(fn):
            out = []
            for g in available:
                try:
                    v = fn(node, g) if fn else None
                except Exception:
                    v = None
                out.append(f"{g}:{'' if v is None else format(float(v), '.3f')}")
            return ";".join(out)
        sc = ";".join(f"{g}:{v:.4f}" for g, v in sorted(self.last_scores.items()))
        self._log.write(f"{now:.3f},{self.policy},{node},"
                        f"{'|'.join(map(str, available))},{'|'.join(map(str, chosen))},"
                        f"{self.last_reject_reason or ''},{sc},"
                        f"{fmt(snr_fn)},{fmt(prior_fn)}\n")

    def select(self, node_id, available, uplink_prior_fn=None, uplink_snr_fn=None):
        now = time.time()
        self.stats["candidates"] += 1
        self.last_scores = {}
        if not available:
            self.last_reject_reason = "no_gateway"
            self.stats["rej_no_gateway"] += 1
            self._write_log(now, node_id, [], [], uplink_snr_fn, uplink_prior_fn)
            return []
        ok_air = [g for g in available if self._airtime_ok(g, now)]
        if not ok_air:
            self.last_reject_reason = "airtime"
            self.stats["rej_airtime"] += 1
            self._write_log(now, node_id, available, [], uplink_snr_fn, uplink_prior_fn)
            return []
        feasible = [g for g in ok_air if self._quota_ok(g, now)]
        if not feasible:
            self.last_reject_reason = "quota"
            self.stats["rej_quota"] += 1
            self._write_log(now, node_id, available, [], uplink_snr_fn, uplink_prior_fn)
            return []

        if self.policy == "broadcast":
            chosen = feasible
        elif self.policy == "random":
            chosen = [self._rng.choice(feasible)]
        elif self.policy == "round-robin":
            chosen = [feasible[self._rr[node_id] % len(feasible)]]
            self._rr[node_id] += 1
        elif self.policy == "ewma":
            chosen = [self._argmax(feasible, lambda g: self.belief.ewma(node_id, g))]
        elif self.policy == "belief-only":
            def sc(g):
                p = uplink_prior_fn(node_id, g) if uplink_prior_fn else None
                return 0.5 if p is None else p
            chosen = [self._argmax(feasible, sc)]
        elif self.policy == "snr-greedy":
            if uplink_snr_fn is None:
                raise RuntimeError("snr-greedy needs uplink_snr_fn from the caller")
            def sc(g):
                v = uplink_snr_fn(node_id, g)
                if v is None:
                    self.stats["snr_missing"] += 1
                    return -999.0
                return float(v)
            chosen = [self._argmax(feasible, sc)]
        else:  # state-aware (SAF-LoRa, unchanged)
            chosen = [self._argmax(feasible, lambda g: self.belief.blended(
                node_id, g, uplink_prior_fn(node_id, g) if uplink_prior_fn else None))]

        for g in chosen:
            self._airtime.add(g, now)
            self._quota.add(g, now)
            self.stats[f"sent_gw{g}"] += 1
        self.last_reject_reason = None
        self._write_log(now, node_id, available, chosen, uplink_snr_fn, uplink_prior_fn)
        return chosen

    def used_capacity(self):
        now = time.time()
        return {g: self._quota.count(g, now) for g in self.gateway_ids}
