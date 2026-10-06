#!/usr/bin/env python3
"""Redraw the E10 uplink-downlink mismatch figure, styled to match the
paper's other simulation figures (plain serif, not heavily bold).

Usage:  python3 redraw_e10.py [extensions_results.csv] [outbase]
Writes <outbase>.pdf and <outbase>.png   (default outbase: e10_mismatch)
"""
import csv, sys, statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "extensions_results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "e10_mismatch"

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.linewidth": 0.9, "lines.linewidth": 1.8, "lines.markersize": 6,
    "legend.fontsize": 10.5,
})

STYLE = [
    ("random",          "Random",                      "tab:gray",  "v", "-"),
    ("snr-greedy-loss",  "SNR-greedy (+loss)",          "tab:green", "^", "-"),
    ("belief",           "SAF-LoRa (belief)",           "tab:red",   "s", "-"),
    ("ewma",              "EWMA",                        "tab:orange","D", "--"),
    ("blend-paper",       "Fixed blend $n/(n{+}5)$",     "tab:pink",  "P", "--"),
    ("saf-adaptive",      "SAF-LoRa (adaptive fusion)",  "tab:blue",  "o", "-"),
    ("oracle",            "Oracle",                      "black",     "o", "-"),
]

rows = [r for r in csv.DictReader(open(SRC)) if r["exp"] == "E10"]
if not rows:
    sys.exit("no E10 rows found in " + SRC)

vals = defaultdict(list)
for r in rows:
    vals[(float(r["rho"]), r["policy"])].append(100 * float(r["delivery"]))

rho = sorted({k[0] for k in vals}, reverse=True)

fig, ax = plt.subplots(figsize=(6.6, 4.6))
print(f"{'rho':>6} " + " ".join(f"{p[:11]:>12}" for p, *_ in STYLE))
for pol, lab, col, mk, ls in STYLE:
    y = [st.mean(vals[(r, pol)]) for r in rho if (r, pol) in vals]
    if len(y) != len(rho):
        print(f"  (skipping {pol}: {len(y)}/{len(rho)} points)")
        continue
    ax.plot(rho, y, marker=mk, color=col, linestyle=ls, label=lab)
for r in rho:
    print(f"{r:6.2f} " + " ".join(
        f"{st.mean(vals[(r,p)]):12.2f}" if (r, p) in vals else f"{'-':>12}"
        for p, *_ in STYLE))

ax.set_xlim(max(rho) + 0.03, min(rho) - 0.03)
ax.set_xticks(rho)
ax.set_xlabel(r"Uplink--downlink coupling $\rho$")
ax.set_ylabel("Simulated ACK delivery (%)")
ax.set_title(r"Mismatch, $\Delta p = 0.2$")
ax.grid(alpha=.25)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True)

fig.tight_layout()
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=250, bbox_inches="tight")
print("\nwrote", OUT + ".pdf", "and", OUT + ".png")
