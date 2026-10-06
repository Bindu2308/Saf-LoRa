"""
run_gateway_scaling_v3.py -- gateway-diversity scaling (1..6 gateways), flat and coverage modes.

Same channel generation as run_gateway_scaling_v2.py (identical random streams), plus:
  1. Ties broken uniformly at random (TIES=random, default; TIES=first = legacy v2 rule).
     In v2, max() gave ties to the LOWEST gateway index = gateway 1 = the JAMMED gateway.
     A lost packet is stored as SNR = -99 dB, so whenever all of a node's uplinks are lost
     SNR-greedy "tied" and always chose the jammed gateway. In coverage mode, where distant
     links lose most packets, this can push SNR-greedy below random for reasons that have
     nothing to do with SNR. The TIE-RATE and JAM-SHARE diagnostics below show how often.
  2. Extra baseline snr-last: last RECEIVED SNR per link (the paper's definition of
     SNR-greedy, "most recent uplink SNR observed"); snr-greedy keeps the v2 behaviour.
  3. Expected delivery (mean of the true delivery probability of the chosen gateway)
     is reported next to realized delivery: same mean, lower variance.
  4. Paired 95% CIs for SAF(belief) - SNR-greedy and SAF - snr-last.
  5. MODEL DESCRIPTION printout (topology, distance_scaled_params at sample distances,
     measured uplink loss per gateway) -> use it to write the paper's coverage definition.
Usage:  NSEEDS=50 python run_gateway_scaling_v3.py | tee out_scaling_final.txt
        TIES=first NSEEDS=50 python run_gateway_scaling_v3.py   (reproduces v2 numbers)
Output: gateway_scaling_results.csv
"""
import csv, os, random, statistics as st
import numpy as np
from topology import make_topology
from gilbert_elliott import GEParams, distance_scaled_params
from gilbert_elliott_jam_patch import JammedGEChannel
from hmm_estimator import ChannelHMM

NSEEDS = int(os.environ.get("NSEEDS", 50))
TIES = os.environ.get("TIES", "random").lower()
N_NODES, ROUNDS, SEV, AREA = 10, 200, 1.5, 1300.0
GWS = [1, 2, 3, 4, 5, 6]
BASE = GEParams(p_good_to_bad=0.05, p_bad_to_good=0.3, loss_prob_good=0.03, loss_prob_bad=0.5)
POLS = ("random", "round-robin", "snr-greedy", "snr-last", "belief", "oracle")
T_CRIT = 2.01 if NSEEDS >= 50 else 2.09 if NSEEDS >= 20 else 2.26


def simulate(mode, n_gw, seed):
    """Identical random-number use to v2; additionally records loss flags."""
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
        last = {g: -99.0 for g in range(1, n_gw + 1)}
        for _ in range(ROUNDS):
            snr, bel, tp, dv, lost = {}, {}, {}, {}, {}
            for g in range(1, n_gw + 1):
                _, q, state = ch[(n, g)].step()
                bel[g] = hm[(n, g)].update(q)
                snr[g] = q if q is not None else -99.0
                lost[g] = q is None
                if q is not None:
                    last[g] = q
                p = 0.75 if state == "GOOD" else 0.25
                p = max(0.0, min(1.0, p - 0.04 * sev[(n, g)]))
                tp[g] = p; dv[g] = py.random() < p
            rounds.append((n, snr, dict(last), bel, tp, dv, lost))
    return rounds, topo


def argmax(gws, score, rng):
    best = max(score(x) for x in gws)
    cands = [x for x in gws if score(x) == best]
    if len(cands) == 1 or TIES == "first":
        return cands[0], len(cands) > 1
    return rng.choice(cands), True


def run(pol, rounds, n_gw, seed):
    rng = random.Random(seed); rr = {}; gws = list(range(1, n_gw + 1))
    k = e = ties = jam = 0
    for n, snr, last, bel, tp, dv, _ in rounds:
        tie = False
        if pol == "random": g = rng.choice(gws)
        elif pol == "round-robin":
            i = rr.get(n, 0); g = gws[i % n_gw]; rr[n] = i + 1
        elif pol == "snr-greedy": g, tie = argmax(gws, lambda x: snr[x], rng)
        elif pol == "snr-last": g, tie = argmax(gws, lambda x: last[x], rng)
        elif pol == "belief": g, tie = argmax(gws, lambda x: bel[x], rng)
        else: g, tie = argmax(gws, lambda x: tp[x], rng)
        k += dv[g]; e += tp[g]; ties += tie; jam += (g == 1)
    L = len(rounds)
    return dict(real=100 * k / L, exp=100 * e / L, tie=100 * ties / L, jam=100 * jam / L)


def describe_model():
    print("=== MODEL DESCRIPTION (for the paper's coverage-configuration sentence) ===")
    print(f"area {AREA:.0f} m x {AREA:.0f} m, {N_NODES} nodes uniform at random (re-drawn per seed), "
          f"{ROUNDS} rounds, {SEV} dB jam + delivery -{0.04*SEV:.2f} on gateway 1 only")
    print("flat: every link uses BASE =", BASE)
    print("coverage: link params = distance_scaled_params(node-gateway distance):")
    for d in (0, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1838):
        print(f"   d = {d:5d} m -> {distance_scaled_params(float(d))}")
    for n_gw in GWS:
        t = make_topology(N_NODES, n_gw, AREA, seed=0)
        dist = [t.distance(n, g) for n in range(N_NODES) for g in range(n_gw)]
        pos = "; ".join(f"({x:.0f},{y:.0f})" for x, y in t.gateway_positions)
        print(f"   {n_gw} GW (seed 0): positions {pos}; node-GW distance mean {st.mean(dist):.0f} m "
              f"[{min(dist):.0f}-{max(dist):.0f}]")
    print("delivery probability: 0.75 GOOD / 0.25 BAD in BOTH modes (independent of distance);")
    print("only the uplink observations (loss, SNR) depend on distance in coverage mode.\n")


describe_model()
rows = []
for mode in ("coverage", "flat"):
    print(f"\n=== mode = {mode} (1.5 dB jam on gateway 1; {NSEEDS} seeds; ties = {TIES}) ===")
    print("GW |  Random     RR    SNR  SNRlast  Belief  Oracle | Gn(bel) Gn(SNR) | Bel-SNR (pp)     Bel-SNRlast (pp) "
          "| SNR tie% | GW1(jam) share %: SNR Bel RR | uplink loss % GW1..")
    for n_gw in GWS:
        V = {p: [] for p in POLS}; loss = np.zeros(n_gw); nobs = 0
        for s in range(NSEEDS):
            r, _ = simulate(mode, n_gw, s)
            for rec in r:
                loss += [rec[6][g] for g in range(1, n_gw + 1)]; nobs += 1
            for p in POLS:
                res = run(p, r, n_gw, s + 100000); V[p].append(res)
                rows.append(dict(mode=mode, n_gw=n_gw, seed=s, policy=p, ties=TIES,
                                 **{k: f"{v:.4f}" for k, v in res.items()}))
        m = {p: st.mean(x["real"] for x in V[p]) for p in POLS}
        den = m["oracle"] - m["round-robin"]
        gn = lambda p: (m[p] - m["round-robin"]) / den if den > 1e-9 else float("nan")
        def pdiff(a, b):
            d = [x["exp"] - y["exp"] for x, y in zip(V[a], V[b])]
            return st.mean(d), (T_CRIT * st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0)
        d1, c1 = pdiff("belief", "snr-greedy"); d2, c2 = pdiff("belief", "snr-last")
        tie = st.mean(x["tie"] for x in V["snr-greedy"])
        js = lambda p: st.mean(x["jam"] for x in V[p])
        print(f"{n_gw:2d} | {m['random']:6.2f} {m['round-robin']:6.2f} {m['snr-greedy']:6.2f} {m['snr-last']:6.2f} "
              f"{m['belief']:6.2f} {m['oracle']:6.2f} | {gn('belief'):6.3f} {gn('snr-greedy'):6.3f} | "
              f"{d1:+5.2f} +/- {c1:.2f}   {d2:+5.2f} +/- {c2:.2f}  | {tie:6.1f}   | "
              f"{js('snr-greedy'):5.1f} {js('belief'):5.1f} {js('round-robin'):5.1f} | "
              + " ".join(f"{100*l/nobs:.0f}" for l in loss))
    print("(delivery = realized; paired differences use expected delivery, same mean, lower variance)")

with open("gateway_scaling_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "n_gw", "seed", "policy", "ties", "real", "exp", "tie", "jam"])
    w.writeheader(); w.writerows(rows)
print("\nSaved gateway_scaling_results.csv")
