import csv, statistics as st
from collections import defaultdict
import matplotlib.pyplot as plt

d = defaultdict(dict)
for r in csv.DictReader(open("gateway_asymmetry_results.csv")):
    d[(float(r["delta_p"]), int(r["seed"]))][r["policy"]] = float(r["delivery_rate"])
dps = sorted({k[0] for k in d})

def gain(pol, dp):
    x = [100*(v[pol]-v["round-robin"]) for (s,_),v in d.items() if s == dp]
    return st.mean(x), 2.01*st.stdev(x)/len(x)**0.5

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for pol, lab, c in [("state-aware","SAF-LoRa","tab:red"), ("oracle","Oracle","black")]:
    b, bci = gain(pol, 0.0)
    ys, es = [], []
    for dp in dps:
        m, ci = gain(pol, dp)
        ys.append(m-b); es.append((ci**2 + bci**2)**0.5)
    ax.errorbar(dps, ys, yerr=es, marker="o", capsize=3, color=c, label=lab)
ax.plot(dps, [25*x for x in dps], "--", color="gray", label=r"Prop. 4: $(\epsilon_B-\epsilon_G)\Delta p/2$")
ax.set_xlabel(r"Gateway-asymmetry parameter $\Delta p$")
ax.set_ylabel("Gain over round-robin above $\\Delta p{=}0$ (pp)")
ax.set_title("Gain vs. asymmetry compared with Prop. 4 prediction")
ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("s1_incremental_gain.png", dpi=300)
print("saved s1_incremental_gain.png")
