"""
analyze_extensions.py -- tables (paired 95% CIs) and figures for E7-E11.
Reads extensions_results.csv (from sim_extensions.py).
Run: python analyze_extensions.py [csv]
"""
import csv
import sys
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "extensions_results.csv"
KEY = ("exp", "dp", "rho", "tau", "tx_prob", "obs_noise", "dp_dn", "flip")
COL = {"random": "tab:gray", "round-robin": "tab:blue", "oracle": "black", "snr-greedy": "tab:green", "snr-greedy-loss": "tab:cyan",
       "snr-mean": "tab:olive", "belief": "tab:red", "belief-pred": "darkred", "belief-matched": "tab:brown",
       "ewma": "tab:orange", "blend-paper": "tab:pink", "saf-adaptive": "tab:purple"}
LAB = {"belief": "SAF-LoRa (belief)", "belief-pred": "SAF-LoRa (predictive belief)",
       "belief-matched": "SAF-LoRa (matched HMM)", "saf-adaptive": "SAF-LoRa (adaptive fusion)",
       "snr-greedy": "SNR-greedy", "snr-greedy-loss": "SNR-greedy (+loss)", "snr-mean": "SNR-mean", "ewma": "EWMA",
       "blend-paper": "Fixed blend n/(n+5)", "round-robin": "Round-robin", "random": "Random",
       "oracle": "Oracle"}

data = defaultdict(dict)  # (cfg, seed) -> {policy: delivery%}
for r in csv.DictReader(open(SRC)):
    data[(tuple(r[k] for k in KEY), int(r["seed"]))][r["policy"]] = 100 * float(r["delivery"])


def cfgs(exp, **flt):
    i = {k: n for n, k in enumerate(KEY)}
    out = {c for (c, _s) in data if c[0] == exp and all(float(c[i[k]]) == v for k, v in flt.items())}
    return sorted(out, key=lambda c: tuple(float(x) for x in c[1:]))


def mean(c, p):
    return st.mean(d[p] for (cc, s), d in data.items() if cc == c and p in d)


def paired(c, a, b):
    x = [d[a] - d[b] for (cc, s), d in data.items() if cc == c and a in d and b in d]
    return st.mean(x), 2.01 * st.stdev(x) / len(x) ** 0.5


def table(title, exp, xkey, xlab, pols, comps, **flt):
    i = KEY.index(xkey)
    cs = cfgs(exp, **flt)
    print(f"\n=== {title} ===")
    print(f"{xlab:>8} " + " ".join(f"{p[:12]:>12}" for p in pols) + "  " +
          "  ".join(f"{a[:10]}-{b[:8]:>8}" for a, b in comps))
    for c in cs:
        row = f"{float(c[i]):8.2f} " + " ".join(f"{mean(c, p):12.2f}" for p in pols)
        for a, b in comps:
            g, ci = paired(c, a, b)
            row += f"  {g:+8.2f}±{ci:4.2f}   "
        print(row)
    return cs


def plot(ax, cs, xkey, pols, xlab, title):
    i = KEY.index(xkey)
    xs = [float(c[i]) for c in cs]
    for p in pols:
        ax.plot(range(len(xs)), [mean(c, p) for c in cs], marker="o", color=COL[p], label=LAB[p],
                ls="--" if p in ("snr-greedy", "snr-mean", "ewma", "blend-paper") else "-")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel(xlab); ax.set_ylabel("Simulated ACK delivery (%)"); ax.set_title(title); ax.grid(alpha=.25)


# E7 stale information
P7 = ["random", "snr-greedy-loss", "snr-mean", "ewma", "belief", "belief-pred", "saf-adaptive", "oracle"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, dp in zip(axes, (0.0, 0.2)):
    cs = table(f"E7 stale information, dp={dp}", "E7", "tau", "tau", P7,
               [("belief-pred", "snr-greedy-loss"), ("belief-pred", "snr-mean"), ("belief-pred", "belief")], dp=dp)
    plot(ax, cs, "tau", P7, r"Information age $\tau$ (rounds)", rf"$\Delta p={dp}$")
axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("e7_stale_predictive.png", dpi=250); plt.close(fig)

# E8 sparse observations
P8 = ["random", "snr-greedy-loss", "snr-mean", "ewma", "belief", "belief-pred", "saf-adaptive", "oracle"]
cs = table("E8 sparse uplink observations (dp=0.2)", "E8", "tx_prob", "tx_prob", P8,
           [("belief-pred", "snr-greedy-loss"), ("belief-pred", "snr-mean")])
fig, ax = plt.subplots(figsize=(6.6, 4.4))
plot(ax, cs, "tx_prob", P8, "Uplink transmission probability per round", r"Sparse observations, $\Delta p=0.2$")
ax.invert_xaxis(); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("e8_sparse.png", dpi=250); plt.close(fig)

# E9 observation noise with matched HMM
P9 = ["snr-greedy", "snr-greedy-loss", "belief", "belief-matched", "belief-pred", "oracle"]
cs = table("E9 observation noise (dp=0.2)", "E9", "obs_noise", "sigma", P9,
           [("belief-matched", "snr-greedy-loss"), ("belief", "snr-greedy-loss")])
fig, ax = plt.subplots(figsize=(6.6, 4.4))
plot(ax, cs, "obs_noise", P9, r"Measurement noise $\sigma$ (dB)", r"Observation noise, $\Delta p=0.2$")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("e9_noise_matched.png", dpi=250); plt.close(fig)

# E10 uplink/downlink mismatch
P10 = ["random", "snr-greedy-loss", "belief", "ewma", "blend-paper", "saf-adaptive", "oracle"]
cs = table("E10 uplink/downlink coupling rho (dp=0.2)", "E10", "rho", "rho", P10,
           [("saf-adaptive", "snr-greedy-loss"), ("saf-adaptive", "ewma"), ("saf-adaptive", "blend-paper")])
fig, ax = plt.subplots(figsize=(6.6, 4.4))
plot(ax, cs, "rho", P10, r"Uplink--downlink coupling $\rho$", r"Mismatch, $\Delta p=0.2$")
ax.invert_xaxis(); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("e10_mismatch.png", dpi=250); plt.close(fig)

# E11 opposite downlink asymmetry
cs = table("E11 opposite downlink asymmetry (uplink dp=0.2 on GW1, downlink dp_dn on GW0, rho=0)",
           "E11", "dp_dn", "dp_dn", P10,
           [("saf-adaptive", "snr-greedy-loss"), ("saf-adaptive", "ewma"), ("saf-adaptive", "blend-paper")])
fig, ax = plt.subplots(figsize=(6.6, 4.4))
plot(ax, cs, "dp_dn", P10, r"Downlink asymmetry $\Delta p_{\mathrm{dn}}$ (opposite gateway)",
     "Uplink and downlink disagree")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("e11_opposite.png", dpi=250); plt.close(fig)
print("\nFigures: e7_stale_predictive e8_sparse e9_noise_matched e10_mismatch e11_opposite")
