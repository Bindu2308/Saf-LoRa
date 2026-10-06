"""
run_clean_baseline.py -- how much can the gateway-1 ACK share of a policy that "should give 50%" vary
in a SYMMETRIC (dp = 0, no interference) setting?  Answers: could 56.2% (random) / 61.3% (belief-only)
in the hardware clean runs, or the 57.2% interference result, arise from short-run randomness or
from unequal gateway budgets?

Scenarios (per policy: random, round-robin, belief, snr-greedy):
  A  equal budgets, no quota                       -> pure sampling + node-clustering noise
  B  equal per-window quotas (40 per 6-step window, window demand 60) -> fallback effects only
  C  UNEQUAL budgets: gateway 2 gets `ratio` x the quota of gateway 1 (40) (ratio 0.8, 0.6, 0.4)
     -> shows how much budget exhaustion at one gateway shifts the share of policies that
        fall back to the other gateway (this is the 'budget fallback' hypothesis in the paper)
Run length: SHORT = 10 nodes x 40 rounds (~400 ACK decisions), LONG = 10 nodes x 200 rounds.
Reports mean, SD, 2.5/97.5 percentiles of the GW1 share and P(|share-50| >= 6.2 pp) (the 56.2%
gap) and >= 11.3 pp (the 61.3% gap).
Usage: NREP=300 python run_clean_baseline.py       (quick test: NREP=20)
"""
import os, random, statistics as st
import numpy as np
from sim_core import generate

NREP = int(os.environ.get("NREP", 300))
N = 10
POLS = ("random", "round-robin", "belief", "snr-greedy")


def share(data, policy, quotas, window_steps, seed):
    """Fraction of sent ACKs that went through gateway 0 ("GW1")."""
    n_nodes, T, _ = data["tp"].shape
    rng = random.Random(seed); order_rng = random.Random(seed * 7919 + 13)
    rr = [0] * n_nodes; used = [0, 0]; sent = [0, 0]
    snr, bel = data["snr"], data["bel"]
    for t in range(T):
        if quotas is not None and t % window_steps == 0:
            used = [0, 0]
        order = list(range(n_nodes)); order_rng.shuffle(order)
        for n in order:
            avail = [g for g in (0, 1) if quotas is None or used[g] < quotas[g]]
            if not avail:
                continue
            if policy == "random": g = rng.choice(avail)
            elif policy == "round-robin":
                pref = rr[n] % 2; g = pref if pref in avail else avail[0]; rr[n] += 1
            elif policy == "belief": g = max(avail, key=lambda x: bel[n, t, x])
            else: g = max(avail, key=lambda x: snr[n, t, x])
            used[g] += 1; sent[g] += 1
    tot = sent[0] + sent[1]
    return 100.0 * sent[0] / tot if tot else float("nan")


def summarize(vals, ref_gaps=(6.2, 11.3)):
    v = np.array(vals); dev = np.abs(v - 50.0)
    q = np.percentile(v, [2.5, 97.5])
    return (f"mean {v.mean():5.1f}  SD {v.std(ddof=1):4.1f}  95% range [{q[0]:5.1f},{q[1]:5.1f}]  "
            f"P(dev>=6.2pp) {np.mean(dev >= ref_gaps[0]):.3f}  P(dev>=11.3pp) {np.mean(dev >= ref_gaps[1]):.3f}")


worlds = {}
def get(T, rep):
    k = (T, rep)
    if k not in worlds:
        worlds[k] = generate(N, T, rep, dp=0.0)
    return worlds[k]


for label, T in (("SHORT (10 nodes x 40 rounds)", 40), ("LONG (10 nodes x 200 rounds)", 200)):
    print(f"\n################ {label}, {NREP} replications, dp = 0 ################")
    scen = [("A equal budgets, no quota", None, None)]
    W = 6
    q1 = 40                      # window demand = 10 nodes x 6 steps = 60 ACK candidates, so 40 is NOT always binding
    scen.append(("B equal quotas (40/window)", (q1, q1), W))
    for ratio in (0.8, 0.6, 0.4):
        scen.append((f"C GW2 quota = {ratio:.1f} x GW1 (GW1=40)", (q1, max(1, round(q1 * ratio))), W))
    for name, quotas, w in scen:
        print(f"\n-- scenario {name}")
        for pol in POLS:
            vals = [share(get(T, r), pol, quotas, w, r + 100000) for r in range(NREP)]
            print(f"   {pol:12s} {summarize(vals)}")
