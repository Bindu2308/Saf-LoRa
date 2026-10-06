"""
Plot the gateway-asymmetry sweep (Experiment S1).

Two panels:
  (a) Delivery rate vs. delta_p, all six policies.
  (b) State-aware gain over round-robin vs. delta_p (the Proposition 4
      validation curve), with the delta_p=0 baseline gain also marked
      so the INCREMENTAL gain above baseline is visually clear -- this
      is the quantity Proposition 4 actually predicts grows linearly
      with asymmetry, since even at delta_p=0 a belief-based policy can
      exploit instantaneous state fluctuations that round-robin cannot.

Run:
    python make_gateway_asymmetry_figure.py
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

INPUT = Path("gateway_asymmetry_results.csv")
OUTPUT = Path("gateway_asymmetry_sweep.png")

ORDER = ["random", "round-robin", "ewma", "belief-only", "state-aware", "oracle"]
LABELS = {
    "random": "Random",
    "round-robin": "Round-robin",
    "ewma": "EWMA",
    "belief-only": "Belief-only",
    "state-aware": "SAF-LoRa (state-aware)",
    "oracle": "Oracle",
}


def main():
    data = defaultdict(list)
    with INPUT.open(newline="") as f:
        for row in csv.DictReader(f):
            data[(row["policy"], float(row["delta_p"]))].append(float(row["delivery_rate"]))

    delta_ps = sorted({dp for (_, dp) in data.keys()})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    for policy in ORDER:
        means = [100.0 * st.mean(data[(policy, dp)]) for dp in delta_ps]
        sds = [100.0 * (st.stdev(data[(policy, dp)]) if len(data[(policy, dp)]) > 1 else 0.0)
               for dp in delta_ps]
        ax.errorbar(delta_ps, means, yerr=sds, marker="o", capsize=3,
                    linewidth=1.6, label=LABELS[policy])
    ax.set_xlabel(r"Gateway-asymmetry parameter $\Delta p$")
    ax.set_ylabel("Delivery rate (%)")
    ax.set_title("(a) Delivery rate vs. asymmetry")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    gains = []
    baseline_gain = None
    for dp in delta_ps:
        sa = st.mean(data[("state-aware", dp)])
        rr = st.mean(data[("round-robin", dp)])
        gain = 100.0 * (sa - rr)
        gains.append(gain)
        if dp == 0.0:
            baseline_gain = gain

    ax2.plot(delta_ps, gains, marker="o", color="tab:red", linewidth=1.8,
              label="State-aware gain over round-robin")
    ax2.axhline(baseline_gain, linestyle="--", color="gray", linewidth=1.2,
                label=f"$\\Delta p{{=}}0$ baseline ({baseline_gain:.1f}pp)")
    incremental = [g - baseline_gain for g in gains]
    ax2b = ax2.twinx()
    ax2b.plot(delta_ps, incremental, marker="s", color="tab:blue", linewidth=1.4,
              linestyle=":", label="Incremental gain above baseline")
    ax2.set_xlabel(r"Gateway-asymmetry parameter $\Delta p$")
    ax2.set_ylabel("Absolute gain (pp)", color="tab:red")
    ax2b.set_ylabel("Incremental gain above baseline (pp)", color="tab:blue")
    ax2.set_title("(b) Proposition 4 validation")
    ax2.grid(True, alpha=0.25)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
