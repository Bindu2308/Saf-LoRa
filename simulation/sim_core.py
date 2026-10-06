"""
sim_core.py -- shared core for the extended SAF-LoRa simulations (E1-E6).

Reuses the existing gilbert_elliott.py / gilbert_elliott_jam_patch.py /
hmm_estimator.py modules unchanged. Channel construction and delivery model
are IDENTICAL to run_gateway_asymmetry.py / run_benchmark_layerb.py:
    delivery prob = 0.75 (GOOD state), 0.25 (BAD state)
so results here are directly comparable with earlier tables.

Tie-breaking: uniform at random by default (TIES=random), as stated in the paper;
TIES=first reproduces the legacy lowest-index rule.

Data layout: arrays of shape [N nodes, T steps, 2 gateways]; decisions are
taken TIME-INTERLEAVED (all nodes at step t before any node at t+1), so ACK
quota windows see genuinely simultaneous multi-node demand.
"""

import os
import random
from collections import deque

import numpy as np

from gilbert_elliott import GEParams
from gilbert_elliott_jam_patch import JammedGEChannel
from hmm_estimator import ChannelHMM, HMMParams

P_DELIV_GOOD = 0.75
P_DELIV_BAD = 0.25
BASE_P_GB, BASE_P_BG = 0.05, 0.3
PI_GOOD = BASE_P_BG / (BASE_P_GB + BASE_P_BG)      # 0.857 stationary GOOD share
DEFAULT_L_BAD = 1.0 / BASE_P_BG                    # mean BAD burst (steps)


def gw_params(pi_good, l_bad):
    """GE params with stationary GOOD share pi_good and mean BAD burst l_bad."""
    p_bg = 1.0 / l_bad
    p_gb = p_bg * (1.0 / pi_good - 1.0)
    if p_gb > 1.0:
        raise ValueError(f"infeasible: pi_good={pi_good:.3f}, l_bad={l_bad}")
    return GEParams(p_good_to_bad=p_gb, p_bad_to_good=p_bg,
                    loss_prob_good=0.03, loss_prob_bad=0.5)


def generate(n_nodes, T, seed, dp=0.0, l_bad=DEFAULT_L_BAD,
             obs_noise_db=0.0, matched_hmm=False):
    """One common channel realization. Gateway 0 has stationary GOOD share
    PI_GOOD; gateway 1 has PI_GOOD - dp (same burst length)."""
    rng = np.random.RandomState(seed)
    py = random.Random(seed)
    nrng = np.random.RandomState(seed + 424242)   # measurement-noise stream (does not disturb channel stream)
    pi2 = max(0.05, min(0.95, PI_GOOD - dp))
    pars = (gw_params(PI_GOOD, l_bad), gw_params(pi2, l_bad))

    shape = (n_nodes, T, 2)
    snr = np.empty(shape)
    bel = np.empty(shape)
    belm = np.empty(shape) if matched_hmm else None
    tp = np.empty(shape)
    dv = np.zeros(shape, dtype=bool)

    for n in range(n_nodes):
        ch = [JammedGEChannel(pars[g], rng, jam_severity_db=0.0) for g in (0, 1)]
        h = [ChannelHMM() for _ in (0, 1)]
        hm = ([ChannelHMM(HMMParams(p_good_to_good=1.0 - pars[g].p_good_to_bad,
                                    p_bad_to_bad=1.0 - pars[g].p_bad_to_good))
               for g in (0, 1)] if matched_hmm else None)
        for t in range(T):
            for g in (0, 1):
                _, q, state = ch[g].step()
                if q is not None:
                    q = q + obs_noise_db * nrng.standard_normal()
                snr[n, t, g] = q if q is not None else -99.0
                bel[n, t, g] = h[g].update(q)
                if matched_hmm:
                    belm[n, t, g] = hm[g].update(q)
                p = P_DELIV_GOOD if state == "GOOD" else P_DELIV_BAD
                tp[n, t, g] = p
                dv[n, t, g] = py.random() < p
    return {"snr": snr, "bel": bel, "belm": belm, "tp": tp, "dv": dv}


# Tie-breaking. The paper states "ties are broken uniformly at random".
#   TIES=random (default): uniform choice among tied gateways (matches the paper text)
#   TIES=first           : legacy behaviour, lowest index wins (reproduces the old tables)
# Ties matter: a lost packet is stored as SNR = -99 dB, so whenever BOTH uplinks of a
# node are lost, SNR-greedy sees a tie; with "first" it then always picks gateway 0,
# which is the JAMMED gateway in sim_core2 (jam_gw=0).
TIES = os.environ.get("TIES", "random").lower()
TIE_STATS = {}                                   # policy -> [ties, decisions]


def _pick(avail, score, rng, policy=None):
    vals = [score(x) for x in avail]
    best = max(vals)
    cands = [x for x, v in zip(avail, vals) if v == best]
    if policy is not None:
        st = TIE_STATS.setdefault(policy, [0, 0])
        st[0] += len(cands) > 1
        st[1] += 1
    if len(cands) == 1 or TIES == "first":
        return cands[0]
    return rng.choice(cands)


def _pick_first(avail, score):                  # kept for backward compatibility
    return max(avail, key=score)


def evaluate(data, policy, tau=0, quota=None, window_steps=6,
             alpha=0.2, lam=0.5, seed=0):
    """
    Evaluate one policy on a realization.
    tau   : staleness. snr/belief seen by the policy are from step t-tau;
            confirmed outcomes (EWMA) become available tau extra steps late.
    quota : per-gateway ACK cap per window of `window_steps` time steps
            (None = unrestricted). If no gateway has budget, candidate REJECTED.
    Policies: random, round-robin, snr-greedy, belief, belief-matched, ewma,
              blend (fixed lam), blend-paper (w=n/(n+5)), oracle.
    Returns dict(delivery_sent, reject, per_cand, agree).
    """
    N, T, _ = data["tp"].shape
    rng = random.Random(seed)
    order_rng = random.Random(seed * 7919 + 13)   # same arrival order for every policy
    rr = [0] * N
    ew = [[0.5, 0.5] for _ in range(N)]
    cnt = [[0, 0] for _ in range(N)]
    pend = [deque() for _ in range(N)]
    used = [0, 0]
    sent = rej = dlv = agree = ninf = 0

    snr, bel, belm, tp, dv = (data[k] for k in ("snr", "bel", "belm", "tp", "dv"))

    for t in range(T):
        if quota is not None and t % window_steps == 0:
            used = [0, 0]
        ts = t - tau if t >= tau else 0
        order = list(range(N))
        order_rng.shuffle(order)
        for n in order:
            pq = pend[n]
            while pq and pq[0][0] <= t:
                _, g, s = pq.popleft()
                ew[n][g] = (1 - alpha) * ew[n][g] + alpha * s
                cnt[n][g] += 1

            avail = [g for g in (0, 1) if quota is None or used[g] < quota]
            if not avail:
                rej += 1
                continue

            if policy == "random":
                g = rng.choice(avail)
            elif policy == "round-robin":
                pref = rr[n] % 2
                g = pref if pref in avail else avail[0]
                rr[n] += 1
            elif policy == "snr-greedy":
                g = _pick(avail, lambda x: snr[n, ts, x], rng, policy)
            elif policy == "belief":
                g = _pick(avail, lambda x: bel[n, ts, x], rng, policy)
            elif policy == "belief-matched":
                g = _pick(avail, lambda x: belm[n, ts, x], rng, policy)
            elif policy == "ewma":
                best = max(ew[n][x] for x in avail)
                g = rng.choice([x for x in avail if ew[n][x] == best])
            elif policy == "blend":
                g = _pick(avail, lambda x: (1 - lam) * bel[n, ts, x] + lam * ew[n][x], rng, policy)
            elif policy == "blend-paper":
                def sc(x):
                    w = cnt[n][x] / (cnt[n][x] + 5.0)
                    return (1 - w) * bel[n, ts, x] + w * ew[n][x]
                g = _pick(avail, sc, rng, policy)
            elif policy == "oracle":
                best = max(tp[n, t, x] for x in avail)
                g = rng.choice([x for x in avail if tp[n, t, x] == best])
            else:
                raise ValueError(policy)

            used[g] += 1
            sent += 1
            s = bool(dv[n, t, g])
            dlv += s
            if len(avail) == 2 and tp[n, t, 0] != tp[n, t, 1]:
                ninf += 1
                if tp[n, t, g] == max(tp[n, t, 0], tp[n, t, 1]):
                    agree += 1
            pq.append((t + 1 + tau, g, float(s)))

    total = sent + rej
    return {
        "delivery_sent": dlv / sent if sent else 0.0,
        "reject": rej / total if total else 0.0,
        "per_cand": dlv / total if total else 0.0,
        "agree": agree / ninf if ninf else 0.0,          # oracle-agreement on informative decisions
    }
