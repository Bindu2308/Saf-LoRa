"""
Fig. 12: Nearest-work benchmark + normalized baseline comparison
(per SAF_LoRa_Figure_and_Transaction_Benchmark_Plan, Table 6).

(a) Policy response: delivery rate vs. interference severity, all 7 policies.
(b) Normalized oracle-gap recovery G_norm vs. severity, practical policies only.

Run:
    python make_fig12_benchmark.py
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

DELIVERY_INPUT = Path("benchmark_layerb_results.csv")
GNORM_INPUT = Path("benchmark_layerb_gnorm.csv")
OUTPUT = Path("fig12_benchmark.png")

ORDER = ["random", "round-robin", "snr-greedy", "ewma", "belief-only", "state-aware", "oracle"]
LABELS = {
    "random": "Random",
    "round-robin": "Round-robin",
    "snr-greedy": "RSSI/SNR-greedy",
    "ewma": "EWMA-success",
    "belief-only": "Belief-only",
    "state-aware": "SAF-LoRa",
    "oracle": "Oracle",
}
COLORS = {"random":"tab:gray","round-robin":"tab:blue","snr-greedy":"tab:green",
          "ewma":"tab:orange","belief-only":"tab:purple","state-aware":"tab:red","oracle":"black"}
STYLES = {k: "-" for k in COLORS}; STYLES["belief-only"] = "--"
GNORM_ORDER = ["round-robin", "snr-greedy", "ewma", "belief-only", "state-aware"]


def main():
    delivery = defaultdict(list)
    with DELIVERY_INPUT.open(newline="") as f:
        for row in csv.DictReader(f):
            delivery[(row["policy"], float(row["severity_db"]))].append(float(row["delivery_rate"]))

    gnorm = {}
    with GNORM_INPUT.open(newline="") as f:
        for row in csv.DictReader(f):
            gnorm[(row["policy"], float(row["severity_db"]))] = float(row["g_norm"])

    severities = sorted({sev for (_, sev) in delivery.keys()})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    for policy in ORDER:
        means = [100.0 * st.mean(delivery[(policy, s)]) for s in severities]
        sds = [100.0 * (st.stdev(delivery[(policy, s)]) if len(delivery[(policy, s)]) > 1 else 0.0)
               for s in severities]
        ax.errorbar(severities, means, yerr=sds, marker="o", capsize=3,
                    linewidth=1.6, label=LABELS[policy], color=COLORS[policy], linestyle=STYLES[policy])
    ax.set_xlabel("Interference severity (dB)")
    ax.set_ylabel("Confirmed delivery probability (%)")
    ax.set_title("(a) Policy response to interference severity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5)

    ax2 = axes[1]
    for policy in GNORM_ORDER:
        vals = [gnorm[(policy, s)] for s in severities]
        ax2.plot(severities, vals, marker="o", linewidth=1.6, label=LABELS[policy], color=COLORS[policy], linestyle=STYLES[policy])
    ax2.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
    ax2.axhline(1.0, linestyle="--", color="gray", linewidth=1.0)
    ax2.set_xlabel("Interference severity (dB)")
    ax2.set_ylabel(r"$G_{norm}$ (0 = random, 1 = oracle)")
    ax2.set_title("(b) Normalized oracle-gap recovery")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
