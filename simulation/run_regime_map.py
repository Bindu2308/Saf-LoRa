"""
run_regime_map.py -- regime map of outcome-aware adaptive fusion (saf-adaptive) over
    rho  : uplink-downlink coupling (1 = identical, 0 = independent)
    tau  : information age in rounds (uplink info AND confirmed outcomes delayed by tau)
    L    : probability that an ACK-receipt report is LOST (random loss; 0 = complete feedback)
Compared: SNR-greedy (+loss), belief (fixed HMM), EWMA, fusion, oracle.  dp = 0.2, 10 nodes x 200 rounds.
Prints, for each L, paired tables (pp, mean +/- 95% CI):
   fusion - SNR-greedy ;  fusion - EWMA ;  fusion - max(SNR, EWMA)  (regret vs. the better baseline)
and saves regime_map.csv and regime_map_L*.png heat maps.
Reuses the world/estimator code of sim_extensions.py (same model as Section V-J6).
Usage: NSEEDS=20 python run_regime_map.py         (quick test: NSEEDS=2)
"""
import csv, os, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_extensions import (make_world, hmm_filter, info_tables, pick, N_NODES, N_ROUNDS, PI_G, LAM,
                            Q_G, Q_B, EWMA_ALPHA, BMA_FORGET)

NSEEDS = int(os.environ.get("NSEEDS", 20))
RHOS = (1.0, 0.75, 0.5, 0.25, 0.0)
TAUS = (0, 1, 2, 4, 8)
LOSSES = (0.0, 0.2, 0.5)
POLS = ("snr-greedy-loss", "belief", "ewma", "saf-adaptive", "oracle")


def run(world, tau, miss, seed):
    status, snr, q, y_u = world["status"], world["snr"], world["q"], world["y_u"]
    belief = hmm_filter(status, snr); tabs = info_tables(status, snr, belief)
    rep_u = np.random.RandomState(seed + 9091).random((N_ROUNDS, N_NODES))
    idx = np.arange(N_NODES); out = {}
    for pol in POLS:
        rng = np.random.RandomState(seed + 7919)
        ew = np.full((N_NODES, 2), 0.5); logw = np.array([np.log(0.9), np.log(0.1)]); hist = []; got = []
        for t in range(N_ROUNDS):
            j = t - 1 - tau
            if j >= 0:
                cj, xj, yj, okj = hist[j]
                if pol == "saf-adaptive":
                    pj = np.clip(xj, 1e-3, 1 - 1e-3)
                    ll = yj[:, None] * np.log(pj) + (1 - yj[:, None]) * np.log(1 - pj)
                    logw = BMA_FORGET * logw + (ll * okj[:, None]).sum(0)
                ew[idx, cj] = np.where(okj > 0, (1 - EWMA_ALPHA) * ew[idx, cj] + EWMA_ALPHA * yj, ew[idx, cj])
            k = t - tau
            if k >= 0:
                b_k = belief[k]; age = t - tabs["last_t"][k]; m = tabs["m"][k]
                b_pred = np.where(tabs["last_t"][k] >= 0, m + LAM ** age * (tabs["b_obs"][k] - m), PI_G)
                last_rd = tabs["last_rd"][k]
            else:
                b_k = np.full((N_NODES, 2), PI_G); b_pred = b_k.copy(); last_rd = np.full((N_NODES, 2), np.nan)
            feats = np.stack([Q_B + (Q_G - Q_B) * b_pred, ew], -1)
            if pol == "snr-greedy-loss": c = pick(last_rd, rng)
            elif pol == "belief": c = pick(b_k, rng)
            elif pol == "ewma": c = pick(ew, rng)
            elif pol == "oracle": c = pick(q[t], rng)
            else:
                wts = np.exp(logw - logw.max()); wts /= wts.sum(); c = pick(feats @ wts, rng)
            qc = q[t, idx, c]; yt = (y_u[t, idx, c] < qc).astype(float)
            okt = (rep_u[t] >= miss).astype(float)              # report arrives with prob 1 - miss
            hist.append((c, feats[idx, c], yt, okt)); got.append(qc)
        out[pol] = 100 * float(np.mean(got))
    return out


rows, R = [], {}
for L in LOSSES:
    for rho in RHOS:
        worlds = [make_world(s, dp=0.2, rho=rho) for s in range(NSEEDS)]
        for tau in TAUS:
            for s in range(NSEEDS):
                r = run(worlds[s], tau, L, s)
                R[(L, rho, tau, s)] = r
                for p, v in r.items():
                    rows.append(dict(L=L, rho=rho, tau=tau, seed=s, policy=p, delivery=f"{v:.4f}"))
        print(f"done L={L} rho={rho}", flush=True)

with open("regime_map.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["L", "rho", "tau", "seed", "policy", "delivery"])
    w.writeheader(); w.writerows(rows)

tc = 2.01 if NSEEDS >= 50 else (2.09 if NSEEDS >= 20 else 2.5)
def paired(L, rho, tau, f):
    x = [f(R[(L, rho, tau, s)]) for s in range(NSEEDS)]
    return st.mean(x), (tc * st.stdev(x) / NSEEDS ** 0.5 if NSEEDS > 1 else 0.0)

defs = {"fusion - SNR-greedy": lambda r: r["saf-adaptive"] - r["snr-greedy-loss"],
        "fusion - EWMA": lambda r: r["saf-adaptive"] - r["ewma"],
        "fusion - best(SNR,EWMA)": lambda r: r["saf-adaptive"] - max(r["snr-greedy-loss"], r["ewma"])}
for L in LOSSES:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for ax, (name, f) in zip(axes, defs.items()):
        M = np.array([[paired(L, rho, tau, f)[0] for tau in TAUS] for rho in RHOS])
        print(f"\n=== report loss L={L}: {name} (pp; rows rho, cols tau {TAUS}) ===")
        for rho, row in zip(RHOS, M):
            print(f"rho={rho:4.2f} " + " ".join(f"{v:+6.2f}" for v in row))
        vm = max(1.0, np.abs(M).max())
        im = ax.imshow(M, cmap="RdBu", vmin=-vm, vmax=vm, aspect="auto")
        ax.set_xticks(range(len(TAUS))); ax.set_xticklabels(TAUS); ax.set_yticks(range(len(RHOS))); ax.set_yticklabels(RHOS)
        ax.set_xlabel(r"information age $\tau$ (rounds)"); ax.set_ylabel(r"coupling $\rho$"); ax.set_title(f"{name} (pp), L={L}")
        for a in range(len(RHOS)):
            for b in range(len(TAUS)):
                ax.text(b, a, f"{M[a, b]:+.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(f"regime_map_L{int(L*100)}.png", dpi=200); plt.close(fig)
print("\nSaved regime_map.csv and regime_map_L*.png")
