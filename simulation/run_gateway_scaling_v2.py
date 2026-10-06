"""
run_gateway_scaling_v2.py -- explains/removes the 2-GW discrepancy (68% here vs ~72% in the benchmark).

Root cause in run_gateway_scaling.py: channels use distance_scaled_params(), i.e. an
out-of-coverage packet-LOSS floor of 0.88-0.98 depending on node-gateway distance, but the
true ACK-delivery probability (0.75/0.25 by GE state) does NOT depend on distance. The HMM
belief is dragged down by coverage losses that do not change delivery, so ranking by belief
is partly misled. Two modes are run on identical seeds:
  coverage : original behaviour (topology + distance-scaled loss)
  flat     : all gateways use the same GE parameters as the benchmark (no coverage effect);
             gateway 1 carries the 1.5 dB jam
Prints delivery and G_norm=(D-D_RR)/(D_oracle-D_RR) for both, so the two numbers can be
reported side by side. Usage: NSEEDS=50 python run_gateway_scaling_v2.py
"""
import os, random, statistics as st
import numpy as np
from topology import make_topology
from gilbert_elliott import GEParams, distance_scaled_params
from gilbert_elliott_jam_patch import JammedGEChannel
from hmm_estimator import ChannelHMM

NSEEDS = int(os.environ.get("NSEEDS", 50))
N_NODES, ROUNDS, SEV, AREA = 10, 200, 1.5, 1300.0
GWS = [1, 2, 3, 4, 5, 6]
BASE = GEParams(p_good_to_bad=0.05, p_bad_to_good=0.3, loss_prob_good=0.03, loss_prob_bad=0.5)
POLS = ("random", "round-robin", "snr-greedy", "belief", "oracle")


def simulate(mode, n_gw, seed):
    rng = np.random.RandomState(seed); py = random.Random(seed)
    topo = make_topology(N_NODES, n_gw, AREA, seed=seed)
    ch, hm, sev = {}, {}, {}
    for n in range(N_NODES):
        for g in range(1, n_gw + 1):
            par = distance_scaled_params(topo.distance(n, g - 1)) if mode == "coverage" else BASE
            sev[(n, g)] = SEV if g == 1 else 0.0
            ch[(n, g)] = JammedGEChannel(par, rng, jam_severity_db=sev[(n, g)])
            hm[(n, g)] = ChannelHMM()
    rounds = []
    for n in range(N_NODES):
        for _ in range(ROUNDS):
            snr, bel, tp, dv = {}, {}, {}, {}
            for g in range(1, n_gw + 1):
                _, q, state = ch[(n, g)].step()
                bel[g] = hm[(n, g)].update(q)
                snr[g] = q if q is not None else -99.0
                p = 0.75 if state == "GOOD" else 0.25
                p = max(0.0, min(1.0, p - 0.04 * sev[(n, g)]))
                tp[g] = p; dv[g] = py.random() < p
            rounds.append((n, snr, bel, tp, dv))
    return rounds


def run(pol, rounds, n_gw, seed):
    rng = random.Random(seed); rr = {}; gws = list(range(1, n_gw + 1)); k = 0
    for n, snr, bel, tp, dv in rounds:
        if pol == "random": g = rng.choice(gws)
        elif pol == "round-robin":
            i = rr.get(n, 0); g = gws[i % n_gw]; rr[n] = i + 1
        elif pol == "snr-greedy": g = max(gws, key=lambda x: snr[x])
        elif pol == "belief": g = max(gws, key=lambda x: bel[x])
        else: g = max(gws, key=lambda x: tp[x])
        k += dv[g]
    return 100.0 * k / len(rounds)


for mode in ("coverage", "flat"):
    print(f"\n=== mode = {mode} (1.5 dB jam on gateway 1; {NSEEDS} seeds) ===")
    print("GW |  Random     RR     SNR   Belief  Oracle | Gn(belief) Gn(SNR) | Oracle-Belief (pp)")
    for n_gw in GWS:
        V = {p: [] for p in POLS}
        for s in range(NSEEDS):
            r = simulate(mode, n_gw, s)
            for p in POLS: V[p].append(run(p, r, n_gw, s + 100000))
        m = {p: st.mean(v) for p, v in V.items()}
        den = m["oracle"] - m["round-robin"]
        gn = lambda p: (m[p] - m["round-robin"]) / den if den > 1e-9 else float("nan")
        print(f"{n_gw:2d} | {m['random']:6.2f} {m['round-robin']:6.2f} {m['snr-greedy']:6.2f} {m['belief']:6.2f} "
              f"{m['oracle']:6.2f} | {gn('belief'):8.3f} {gn('snr-greedy'):8.3f} | {m['oracle']-m['belief']:6.2f}")
