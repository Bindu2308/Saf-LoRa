"""
Plot the ACK-quota sweep (Experiment S2): reject rate and delivery-among-sent
vs. quota, all five policies.

Run:
    python make_ack_quota_figure.py
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

INPUT = Path("ack_quota_sweep_results.csv")
OUTPUT = Path("ack_quota_sweep.png")

ORDER = ["random", "round-robin", "ewma", "belief-only", "state-aware"]
LABELS = {
    "random": "Random",
    "round-robin": "Round-robin",
    "ewma": "EWMA",
    "belief-only": "Belief-only",
    "state-aware": "SAF-LoRa",
}


def main():
    reject = defaultdict(list)
    delivery = defaultdict(list)
    with INPUT.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["policy"], int(row["quota"]))
            reject[key].append(float(row["reject_rate"]))
            delivery[key].append(float(row["delivery_rate"]))

    quotas = sorted({q for (_, q) in reject.keys()})
    xlabels = ["Unres." if q >= 10_000 else str(q) for q in quotas]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax = axes[0]
    N, W = 2000, 60   # total rounds, window size
    def closed_form(q):
        full, rem = divmod(N, W)
        sent = full*min(W, 2*q) + min(rem, 2*q)
        return 100.0*(1 - sent/N)
    ax.plot(range(len(quotas)), [closed_form(q) for q in quotas], marker="o",
            linewidth=1.8, color="tab:purple",
            label="All policies (identical by construction)")
    ax.set_xticks(range(len(quotas))); ax.set_xticklabels(xlabels)
    ax.set_xlabel("Per-gateway ACK quota / window")
    ax.set_ylabel("Reject rate (%)")
    ax.set_title("(a) ACK-rejection vs. quota (closed form)")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8)

    ax2 = axes[1]
    for policy in ORDER:
        means = [100.0 * st.mean(delivery[(policy, q)]) for q in quotas]
        sds = [100.0 * (st.stdev(delivery[(policy, q)]) if len(delivery[(policy, q)]) > 1 else 0.0)
               for q in quotas]
        ax2.errorbar(range(len(quotas)), means, yerr=sds, marker="o", capsize=3,
                     linewidth=1.6, label=LABELS[policy], linestyle="--" if policy=="belief-only" else "-")
    ax2.set_xticks(range(len(quotas)))
    ax2.set_xticklabels(xlabels)
    ax2.set_xlabel("Per-gateway ACK quota / window")
    ax2.set_ylabel("Delivery rate among sent ACKs (%)")
    ax2.set_title("(b) Delivery quality vs. quota")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
