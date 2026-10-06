"""
run_extended.py -- extended SAF-LoRa simulations E1-E6.

  E1  stale information (tau)          -- how robust is state-aware selection to old info?
  E2  temporal correlation (L_bad)     -- does the HMM add value beyond instantaneous SNR?
  E3  asymmetry x ACK quota (2-D)      -- does selection keep its benefit under scarcity?
  E4  ACK-demand scaling with N nodes  -- shared quota; unrestricted control is flat by design
  E5  blend ablation (lambda + paper)  -- the confirmed-outcome policy of Sec. IV-G, never simulated before
  E6  observation noise (sigma_obs)    -- estimator quality -> selection -> delivery

All policies see IDENTICAL channel realizations per (config, seed).
Usage:  python run_extended.py [E1 E2 ...]     (default: all)   NSEEDS=50 env var
Output: extended_results.csv  (appended per experiment; delete to start fresh)
"""

import csv
import os
import sys
import time
from pathlib import Path

from sim_core import generate, evaluate, DEFAULT_L_BAD

NSEEDS = int(os.environ.get("NSEEDS", 50))
N_NODES, T = 10, 200
OUT = Path("extended_results.csv")
FIELDS = ["exp", "dp", "l_bad", "tau", "quota", "n_nodes", "lam", "obs_noise",
          "policy", "seed", "delivery_sent", "reject", "per_cand", "agree"]

BASE_POLICIES = ["random", "round-robin", "snr-greedy", "belief", "ewma", "oracle"]


def row(exp, policy, seed, res, **kw):
    r = {k: "" for k in FIELDS}
    r.update(exp=exp, policy=policy, seed=seed, **kw)
    r.update({k: res[k] for k in ("delivery_sent", "reject", "per_cand", "agree")})
    return r


def e1(rows):
    for dp in (0.0, 0.2):
        for seed in range(NSEEDS):
            d = generate(N_NODES, T, seed, dp=dp)
            for tau in (0, 1, 2, 4, 8, 16):
                for p in BASE_POLICIES:
                    rows.append(row("E1", p, seed, evaluate(d, p, tau=tau, seed=seed + 100000),
                                    dp=dp, tau=tau))


def e2(rows):
    for dp in (0.0, 0.2):
        for l_bad in (1, 2, 4, 8, 16):
            for seed in range(NSEEDS):
                d = generate(N_NODES, T, seed, dp=dp, l_bad=l_bad, matched_hmm=True)
                for p in BASE_POLICIES + ["belief-matched"]:
                    rows.append(row("E2", p, seed, evaluate(d, p, seed=seed + 100000),
                                    dp=dp, l_bad=l_bad))


def e3(rows):
    for dp in (0.0, 0.2, 0.4):
        for seed in range(NSEEDS):
            d = generate(N_NODES, T, seed, dp=dp)
            for q in (None, 15, 8, 4, 2):
                for p in BASE_POLICIES:
                    rows.append(row("E3", p, seed, evaluate(d, p, quota=q, seed=seed + 100000),
                                    dp=dp, quota=-1 if q is None else q))


def e4(rows):
    for n in (2, 4, 6, 8, 10, 20, 30, 50):
        for seed in range(NSEEDS):
            d = generate(n, T, seed, dp=0.2)
            for q in (None, 8):
                for p in ["random", "round-robin", "snr-greedy", "belief", "ewma"]:
                    rows.append(row("E4", p, seed, evaluate(d, p, quota=q, seed=seed + 100000),
                                    dp=0.2, n_nodes=n, quota=-1 if q is None else q))


def e5(rows):
    for dp in (0.0, 0.2, 0.4):
        for seed in range(NSEEDS):
            d = generate(N_NODES, T, seed, dp=dp)
            for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
                rows.append(row("E5", "blend", seed,
                                evaluate(d, "blend", lam=lam, seed=seed + 100000), dp=dp, lam=lam))
            for p in ("blend-paper", "snr-greedy", "random", "oracle"):
                rows.append(row("E5", p, seed, evaluate(d, p, seed=seed + 100000), dp=dp))


def e6(rows):
    for sigma in (0.0, 1.0, 2.0, 3.0, 5.0):
        for seed in range(NSEEDS):
            d = generate(N_NODES, T, seed, dp=0.2, obs_noise_db=sigma)
            for p in ("random", "snr-greedy", "belief", "ewma", "oracle"):
                rows.append(row("E6", p, seed, evaluate(d, p, seed=seed + 100000),
                                dp=0.2, obs_noise=sigma))


EXPERIMENTS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6}


def main():
    wanted = [a.upper() for a in sys.argv[1:]] or list(EXPERIMENTS)
    existing = []
    if OUT.exists():
        with OUT.open(newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["exp"] not in wanted]
    new = []
    for name in wanted:
        t0 = time.time()
        EXPERIMENTS[name](new)
        print(f"{name} done in {time.time() - t0:.1f}s ({NSEEDS} seeds)", flush=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(existing)
        w.writerows(new)
    print(f"Saved {OUT.resolve()}")


if __name__ == "__main__":
    main()
