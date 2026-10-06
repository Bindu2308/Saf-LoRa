"""
analyze_extended.py -- tables (paired 95% CIs) + figures for E1-E6.
Reads extended_results.csv, prints summary tables, saves e1..e6 PNGs.
Paired difference = same seed / same channel realization for both policies.
"""

import csv
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"random": "tab:gray", "round-robin": "tab:blue", "snr-greedy": "tab:green",
       "ewma": "tab:orange", "belief": "tab:red", "belief-matched": "tab:brown",
       "blend": "tab:purple", "blend-paper": "tab:pink", "oracle": "black"}
LAB = {"belief": "SAF-LoRa (belief)", "belief-matched": "SAF-LoRa (matched HMM)",
       "snr-greedy": "RSSI/SNR-greedy", "ewma": "EWMA", "round-robin": "Round-robin",
       "random": "Random", "oracle": "Oracle", "blend-paper": "Paper blend w=n/(n+5)"}

data = defaultdict(dict)      # (exp, cfg-tuple, seed) -> {policy: metrics}
for r in csv.DictReader(open("extended_results.csv")):
    cfg = (r["exp"], r["dp"], r["l_bad"], r["tau"], r["quota"], r["n_nodes"], r["lam"], r["obs_noise"])
    data[(cfg, int(r["seed"]))][r["policy"] + ("@" + r["lam"] if r["policy"] == "blend" else "")] = \
        {k: float(r[k]) for k in ("delivery_sent", "reject", "per_cand", "agree")}


def cfgs(exp, **flt):
    idx = {"dp": 1, "l_bad": 2, "tau": 3, "quota": 4, "n_nodes": 5, "lam": 6, "obs_noise": 7}
    out = set()
    for (cfg, _s) in data:
        if cfg[0] != exp:
            continue
        if all(cfg[idx[k]] == str(v) for k, v in flt.items()):
            out.add(cfg)
    return sorted(out, key=lambda c: tuple(float(x) if x not in ("",) else -9 for x in c[1:]))


def vals(cfg, pol, metric="delivery_sent"):
    return [100 * d[pol][metric] for (c, s), d in sorted(data.items()) if c == cfg and pol in d]


def mean(cfg, pol, metric="delivery_sent"):
    return st.mean(vals(cfg, pol, metric))


def sd(cfg, pol, metric="delivery_sent"):
    return st.stdev(vals(cfg, pol, metric))


def paired(cfg, a, b, metric="delivery_sent"):
    x = [100 * (d[a][metric] - d[b][metric]) for (c, s), d in sorted(data.items())
         if c == cfg and a in d and b in d]
    return st.mean(x), 2.01 * st.stdev(x) / len(x) ** 0.5


def f(v):  # float from cfg field
    return float(v)


# ---------------------------------------------------------------- E1
print("\n=== E1: stale information (delivery %, paired diff vs SNR-greedy) ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, dp in zip(axes, ("0.0", "0.2")):
    cs = cfgs("E1", dp=dp)
    taus = [f(c[3]) for c in cs]
    print(f"-- dp={dp}")
    print("tau   " + "  ".join(f"{p:>10}" for p in ("random", "rr", "snr", "belief", "ewma", "oracle")) +
          "   belief-random   belief-snr")
    for c in cs:
        m = [mean(c, p) for p in ("random", "round-robin", "snr-greedy", "belief", "ewma", "oracle")]
        g1, c1 = paired(c, "belief", "random"); g2, c2 = paired(c, "belief", "snr-greedy")
        print(f"{f(c[3]):3.0f}   " + "  ".join(f"{x:10.2f}" for x in m) + f"   {g1:+6.2f}±{c1:.2f}   {g2:+6.2f}±{c2:.2f}")
    for p in ("random", "round-robin", "snr-greedy", "ewma", "belief", "oracle"):
        ax.errorbar(range(len(cs)), [mean(c, p) for c in cs], yerr=[sd(c, p) for c in cs],
                    marker="o", capsize=2, color=COL[p], label=LAB.get(p, p))
    ax.set_xticks(range(len(cs))); ax.set_xticklabels([f"{t:.0f}" for t in taus])
    ax.set_xlabel(r"Information age $\tau$ (steps)"); ax.set_ylabel("Simulated ACK-delivery probability (%)")
    ax.set_title(rf"$\Delta p={dp}$"); ax.grid(alpha=.25)
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("e1_stale_information.png", dpi=250); plt.close(fig)

# ---------------------------------------------------------------- E2
print("\n=== E2: temporal correlation (mean BAD burst length) ===")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for j, dp in enumerate(("0.0", "0.2")):
    cs = cfgs("E2", dp=dp)
    print(f"-- dp={dp}")
    print("L_bad  snr-greedy  belief(fixed)  belief(matched)   matched-snr        ewma   oracle   agree(snr/belief/matched)")
    for c in cs:
        g, ci = paired(c, "belief-matched", "snr-greedy")
        print(f"{f(c[2]):5.0f}  {mean(c,'snr-greedy'):10.2f}  {mean(c,'belief'):13.2f}  {mean(c,'belief-matched'):15.2f}   "
              f"{g:+6.2f}±{ci:.2f}  {mean(c,'ewma'):8.2f} {mean(c,'oracle'):7.2f}   "
              f"{mean(c,'snr-greedy','agree'):.1f}/{mean(c,'belief','agree'):.1f}/{mean(c,'belief-matched','agree'):.1f}")
    ax = axes[j]
    for p in ("random", "snr-greedy", "belief", "belief-matched", "ewma", "oracle"):
        ax.errorbar(range(len(cs)), [mean(c, p) for c in cs], yerr=[sd(c, p) for c in cs],
                    marker="o", capsize=2, color=COL[p], label=LAB.get(p, p))
    ax.set_xticks(range(len(cs))); ax.set_xticklabels([f"{f(c[2]):.0f}" for c in cs])
    ax.set_xlabel("Mean BAD-burst length (steps)"); ax.set_ylabel("Simulated ACK-delivery probability (%)")
    ax.set_title(rf"$\Delta p={dp}$"); ax.grid(alpha=.25)
axes[0].legend(fontsize=7)
ax = axes[2]
for dp, ls in (("0.0", "-"), ("0.2", "--")):
    cs = cfgs("E2", dp=dp)
    ax.plot(range(len(cs)), [paired(c, "belief-matched", "snr-greedy")[0] for c in cs], ls,
            marker="o", color="tab:brown", label=rf"matched HMM − SNR-greedy, $\Delta p={dp}$")
    ax.plot(range(len(cs)), [paired(c, "belief", "snr-greedy")[0] for c in cs], ls,
            marker="s", color="tab:red", label=rf"fixed HMM − SNR-greedy, $\Delta p={dp}$")
ax.axhline(0, color="gray", lw=1)
ax.set_xticks(range(len(cs))); ax.set_xticklabels([f"{f(c[2]):.0f}" for c in cs])
ax.set_xlabel("Mean BAD-burst length (steps)"); ax.set_ylabel("Paired gain over SNR-greedy (pp)")
ax.set_title("Does temporal filtering add value?"); ax.grid(alpha=.25); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig("e2_temporal_correlation.png", dpi=250); plt.close(fig)

# ---------------------------------------------------------------- E3
print("\n=== E3: asymmetry x quota (delivery among sent; gain SAF-LoRa - policy, paired) ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, dp in zip(axes, ("0.0", "0.2", "0.4")):
    cs = cfgs("E3", dp=dp)
    print(f"-- dp={dp}")
    print("quota  reject%   random     rr     snr   belief    ewma  oracle | belief-RR       belief-snr    succ-ACK/100cand(belief, RR)")
    for c in cs:
        q = "inf" if c[4] == "-1" else c[4]
        g, ci = paired(c, "belief", "round-robin"); g2, ci2 = paired(c, "belief", "snr-greedy")
        print(f"{q:>5}  {mean(c,'belief','reject'):6.1f}  " +
              "  ".join(f"{mean(c,p):6.2f}" for p in ("random", "round-robin", "snr-greedy", "belief", "ewma", "oracle")) +
              f" | {g:+5.2f}±{ci:.2f}  {g2:+5.2f}±{ci2:.2f}    {mean(c,'belief','per_cand'):5.1f}, {mean(c,'round-robin','per_cand'):5.1f}")
    for p in ("random", "round-robin", "snr-greedy", "ewma", "belief", "oracle"):
        ax.errorbar(range(len(cs)), [mean(c, p) for c in cs], yerr=[sd(c, p) for c in cs],
                    marker="o", capsize=2, color=COL[p], label=LAB.get(p, p))
    ax.set_xticks(range(len(cs))); ax.set_xticklabels(["∞" if c[4] == "-1" else c[4] for c in cs])
    ax.set_xlabel("Per-gateway quota per window"); ax.set_ylabel("Delivery among sent ACKs (%)")
    ax.set_title(rf"$\Delta p={dp}$"); ax.grid(alpha=.25)
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("e3_asymmetry_x_quota.png", dpi=250); plt.close(fig)

# ---------------------------------------------------------------- E4
print("\n=== E4: ACK-demand scaling (dp=0.2; window = 6 time steps; quota 8/gateway) ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
print("  N   reject%(Q=8)  delivery(sent) belief/RR   succ-ACK per 100 candidates belief/RR   | Q=inf delivery belief/RR")
ns = sorted({f(c[5]) for c in cfgs("E4")})
for n in ns:
    cq = [c for c in cfgs("E4", quota=8) if f(c[5]) == n][0]
    cu = [c for c in cfgs("E4", quota=-1) if f(c[5]) == n][0]
    print(f"{n:4.0f}   {mean(cq,'belief','reject'):6.1f}       {mean(cq,'belief'):6.2f}/{mean(cq,'round-robin'):6.2f}"
          f"              {mean(cq,'belief','per_cand'):5.1f}/{mean(cq,'round-robin','per_cand'):5.1f}"
          f"                      | {mean(cu,'belief'):6.2f}/{mean(cu,'round-robin'):6.2f}")
cq_list = [[c for c in cfgs("E4", quota=8) if f(c[5]) == n][0] for n in ns]
cu_list = [[c for c in cfgs("E4", quota=-1) if f(c[5]) == n][0] for n in ns]
axes[0].plot(range(len(ns)), [mean(c, "belief", "reject") for c in cq_list], marker="o", color="tab:red")
axes[0].set_ylabel("ACK reject rate (%), Q=8"); axes[0].set_title("(a) Reject rate vs. node count")
for p in ("random", "round-robin", "snr-greedy", "ewma", "belief"):
    axes[1].plot(range(len(ns)), [mean(c, p) for c in cq_list], marker="o", color=COL[p], label=LAB.get(p, p))
axes[1].plot(range(len(ns)), [mean(c, "belief") for c in cu_list], ":", color="gray", label="SAF-LoRa, unrestricted (flat by construction)")
axes[1].set_ylabel("Delivery among sent ACKs (%)"); axes[1].set_title("(b) Delivery vs. node count"); axes[1].legend(fontsize=7)
for ax in axes:
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"{n:.0f}" for n in ns]); ax.set_xlabel("Nodes N"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("e4_ack_demand_scaling.png", dpi=250); plt.close(fig)

# ---------------------------------------------------------------- E5
print("\n=== E5: blend ablation  score=(1-lam)*belief + lam*EWMA  (delivery %) ===")
fig, ax = plt.subplots(figsize=(6.6, 4.4))
lams = [0.0, 0.25, 0.5, 0.75, 1.0]
for dp, mk in (("0.0", "o"), ("0.2", "s"), ("0.4", "^")):
    print(f"-- dp={dp}")
    for lam in lams:
        cl = [x for x in cfgs("E5", dp=dp) if x[6] == str(lam)][0]
        print(f"   lam={lam:4.2f}: {mean(cl,'blend@'+str(lam)):6.2f}")
    cl0 = [x for x in cfgs("E5", dp=dp) if x[6] == ""][0]
    print(f"   paper rule w=n/(n+5): {mean(cl0,'blend-paper'):6.2f}   snr-greedy {mean(cl0,'snr-greedy'):6.2f}   "
          f"random {mean(cl0,'random'):6.2f}   oracle {mean(cl0,'oracle'):6.2f}")
    cl_0 = [x for x in cfgs("E5", dp=dp) if x[6] == "0.0"][0]
    # blend@0.0 lives under a different cfg key (lam=0.0): pair by seed manually
    b0 = {s: d["blend@0.0"]["delivery_sent"] for (cc, s), d in data.items() if cc == cl_0 and "blend@0.0" in d}
    bp = {s: d["blend-paper"]["delivery_sent"] for (cc, s), d in data.items() if cc == cl0 and "blend-paper" in d}
    diffs = [100 * (bp[s] - b0[s]) for s in bp if s in b0]
    print(f"   paper rule − belief-only: {st.mean(diffs):+.2f}±{2.01*st.stdev(diffs)/len(diffs)**.5:.2f} pp (paired)")
    ax.plot(lams, [mean([x for x in cfgs("E5", dp=dp) if x[6] == str(l)][0], "blend@" + str(l)) for l in lams],
            marker=mk, label=rf"fixed $\lambda$, $\Delta p={dp}$")
    ax.axhline(mean(cl0, "blend-paper"), ls=":", color=ax.lines[-1].get_color(), lw=1)
ax.set_xlabel(r"EWMA weight $\lambda$ (0 = belief-only, 1 = EWMA-only)")
ax.set_ylabel("Simulated ACK-delivery probability (%)"); ax.grid(alpha=.25); ax.legend(fontsize=7)
ax.set_title("Blend ablation (dotted = paper rule $w=n/(n+5)$)")
fig.tight_layout(); fig.savefig("e5_blend_ablation.png", dpi=250); plt.close(fig)

# ---------------------------------------------------------------- E6
print("\n=== E6: observation noise on the SNR reading fed to HMM and SNR-greedy (dp=0.2) ===")
fig, ax = plt.subplots(figsize=(6.6, 4.4))
cs = cfgs("E6")
print("sigma   snr-greedy  belief   ewma   random  oracle | belief-snr (paired)   agree snr/belief")
for c in cs:
    g, ci = paired(c, "belief", "snr-greedy")
    print(f"{f(c[7]):4.1f}    {mean(c,'snr-greedy'):7.2f}  {mean(c,'belief'):7.2f} {mean(c,'ewma'):6.2f} {mean(c,'random'):7.2f} {mean(c,'oracle'):7.2f} |"
          f"   {g:+5.2f}±{ci:.2f}          {mean(c,'snr-greedy','agree'):.1f}/{mean(c,'belief','agree'):.1f}")
for p in ("random", "snr-greedy", "belief", "ewma", "oracle"):
    ax.errorbar([f(c[7]) for c in cs], [mean(c, p) for c in cs], yerr=[sd(c, p) for c in cs],
                marker="o", capsize=2, color=COL[p], label=LAB.get(p, p))
ax.set_xlabel(r"Measurement noise $\sigma_{obs}$ (dB)"); ax.set_ylabel("Simulated ACK-delivery probability (%)")
ax.grid(alpha=.25); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("e6_observation_noise.png", dpi=250); plt.close(fig)
print("\nFigures: e1_stale_information e2_temporal_correlation e3_asymmetry_x_quota e4_ack_demand_scaling e5_blend_ablation e6_observation_noise")
