"""
validate_geom2_duty.py -- more faithful check of the second-geometry gain
model than validate_geom2_gain.py. That script used a CONSTANT structural
asymmetry (dp) present every round, which gave gains ~10x the hardware
result. The real interferer is INTERMITTENT (duty 30.8% per the paper), so
this script reproduces that: both gateways run symmetric clean channels,
and an ON/OFF process (mean duty matching the hardware interferer) degrades
one gateway only while ON. Severity is calibrated so the LONG-RUN AVERAGE
round-robin gap still matches the hardware's measured 12.6 pp, but now via
bursts instead of a permanent tilt -- so policies only benefit on the
fraction of rounds the interferer is active, and only once their estimator
has caught up to the transition, which is the ingredient the first
validation omitted.

Round-to-second mapping is illustrative, not literal: MEAN_ON_ROUNDS is
chosen for a resolvable burst length in the simulator's round granularity,
with MEAN_OFF_ROUNDS set to reproduce the measured 30.8% duty cycle exactly.

Usage:  NSEEDS=1000 python3 validate_geom2_duty.py
"""
import csv
import os
import random
import statistics as st

import numpy as np

from gilbert_elliott_jam_patch import JammedGEChannel
from hmm_estimator import ChannelHMM
from sim_core import evaluate, gw_params, PI_GOOD, P_DELIV_GOOD, P_DELIV_BAD, DEFAULT_L_BAD

NSEEDS = int(os.environ.get("NSEEDS", 1000))
N, T = 10, 200
T_CRIT = 1.96

DUTY = 0.308                               # measured interferer duty cycle (paper: 30.8%)
MEAN_ON_ROUNDS = 3.0                       # illustrative round-granularity burst length
MEAN_OFF_ROUNDS = MEAN_ON_ROUNDS * (1 - DUTY) / DUTY
P_ON_TO_OFF = 1.0 / MEAN_ON_ROUNDS
P_OFF_TO_ON = 1.0 / MEAN_OFF_ROUNDS
JAM_GW = 0                                 # degraded while ON (matches final12's GW1)
TARGET_GAP_PP = 12.6


def interferer_trace(rng, T):
    on = rng.random() < DUTY
    trace = []
    for _ in range(T):
        trace.append(on)
        on = (not (rng.random() < P_ON_TO_OFF)) if on else (rng.random() < P_OFF_TO_ON)
    return trace


def generate(n_nodes, T, seed, sev_pp):
    rng = np.random.RandomState(seed)
    py = random.Random(seed)
    pars = gw_params(PI_GOOD, DEFAULT_L_BAD)   # SAME params both gateways: symmetric base
    shape = (n_nodes, T, 2)
    snr = np.empty(shape); bel = np.empty(shape)
    tp = np.empty(shape); dv = np.zeros(shape, dtype=bool)
    for n in range(n_nodes):
        ch = [JammedGEChannel(pars, rng, jam_severity_db=0.0) for _ in (0, 1)]
        h = [ChannelHMM() for _ in (0, 1)]
        interf = interferer_trace(rng, T)
        for t in range(T):
            for g in (0, 1):
                _, q, state = ch[g].step()
                snr[n, t, g] = q if q is not None else -99.0
                bel[n, t, g] = h[g].update(q)
                p = P_DELIV_GOOD if state == "GOOD" else P_DELIV_BAD
                if g == JAM_GW and interf[t]:
                    p = max(0.0, min(1.0, p - sev_pp / 100.0))
                tp[n, t, g] = p
                dv[n, t, g] = py.random() < p
    return {"snr": snr, "bel": bel, "belm": None, "tp": tp, "dv": dv}


def rr_gap_pp(sev_pp, seeds):
    g1, g2 = [], []
    for s in seeds:
        w = generate(N, T, s, sev_pp)
        g1.append(w["tp"][:, :, 0].mean())
        g2.append(w["tp"][:, :, 1].mean())
    return 100 * (st.mean(g2) - st.mean(g1))


def calibrate_sev(seeds_for_calib=60):
    print("Calibrating ON-period severity to match hardware gap of %.1f pp..." % TARGET_GAP_PP)
    best_sev, best_err = None, 1e9
    for sev in (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70):
        gap = rr_gap_pp(sev, range(seeds_for_calib))
        err = abs(gap - TARGET_GAP_PP)
        print(f"  sev={sev:4.0f}pp-while-ON  simulated avg gap={gap:+.2f} pp  |err|={err:.2f}")
        if err < best_err:
            best_sev, best_err = sev, err
    print(f"-> using sev={best_sev:.0f} pp while ON (gap error {best_err:.2f} pp)\n")
    return best_sev


def col(sev, pol, seeds):
    return [100 * evaluate(generate(N, T, s, sev), pol, seed=s + 100000)["delivery_sent"] for s in seeds]


def paired_stats(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d); m = st.mean(d); ci = T_CRIT * st.stdev(d) / (n ** 0.5)
    return m, ci, n


def main():
    seeds = range(NSEEDS)
    sev = calibrate_sev()
    policies = {"round-robin": "round-robin", "snr-greedy": "snr-greedy",
                "belief": "belief", "blend-paper": "blend-paper"}
    print(f"Running {NSEEDS} seeds per policy, ON-severity={sev:.0f}pp, duty={DUTY:.3f} ...")
    data = {name: col(sev, pol, seeds) for name, pol in policies.items()}
    rows = []
    print(f"\n{'policy':<14}{'mean (%)':>10}")
    for name, vals in data.items():
        m = st.mean(vals)
        print(f"{name:<14}{m:>10.3f}")
        for s, v in zip(seeds, vals):
            rows.append(dict(policy=name, seed=s, delivery_pct=f"{v:.4f}"))
    print(f"\n{'contrast':<28}{'gain (pp)':>12}{'95% CI':>12}{'n':>8}")
    hw_observed = {"snr-greedy": -0.21, "blend-paper": -0.04, "belief": -0.04}
    for name in ("snr-greedy", "blend-paper"):
        m, ci, n = paired_stats(data[name], data["round-robin"])
        tag = "excludes 0" if abs(m) > ci else "includes 0"
        hw = hw_observed.get(name)
        print(f"{name + ' - round-robin':<28}{m:>+12.3f}{ci:>12.3f}{n:>8d}   ({tag}; hardware: {hw:+.2f} pp, n=12)")
    with open("geom2_duty_validation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "seed", "delivery_pct"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved geom2_duty_validation_results.csv ({len(rows)} rows), sev={sev:.0f}pp, duty={DUTY:.3f}, NSEEDS={NSEEDS}")


if __name__ == "__main__":
    main()
