"""
auc_uplink_predicts_downlink.py

THE PREMISE TEST for SAF-LoRa.

The revised direction assumes a state-aware feedback scheduler can pick
which node-gateway pair should receive a rationed ACK. That only works if
something the server already observes predicts whether an ACK will
actually arrive.

The server observes the UPLINK (node -> gateway): SNR, RSSI, CRC outcome,
and the HMM belief derived from them. The ACK travels the other way
(gateway -> node). The two directions may be correlated -- or may not be.

This script measures that correlation directly:

    AUC ~ 0.5   the uplink belief carries no information about downlink
                success. A state-aware scheduler built on uplink belief
                would be scheduling on noise. Build a separate downlink
                observation process first.

    AUC > 0.6   there is usable signal. The scheduler premise holds and
                the firmware work is justified.

Run from the folder holding both CSVs:
    python auc_uplink_predicts_downlink.py

Inputs
    graph_observations_clean.csv : telegram_id, node_id, gateway_id,
                                   hmm_p_good_at_time, snr_db, rssi_db, ...
    dfrag_ack_log_clean.csv      : timestamp, node_id, telegram_id, success

Join key is (node_id, telegram_id).

IMPORTANT CAVEAT, read before trusting the number
-------------------------------------------------
dfrag_ack_log records whether the NODE received an ACK, but not which
gateway sent it. So a per-(node, gateway) AUC is not directly computable
from these files. This script reports the per-NODE version: does the
uplink belief for a node (aggregated across gateways) predict whether
that node's ACK arrived? That is a strictly weaker test than the one the
scheduler needs, but if it comes back at chance, the stronger version is
very unlikely to be better -- so it is a valid early kill-switch, not a
substitute for the real experiment.
"""

import csv
from collections import defaultdict


def load_uplink(path="graph_observations_clean.csv"):
    """Per (node, telegram): mean HMM belief, mean/min SNR, mean RSSI,
    and the number of gateways that reported anything."""
    acc = defaultdict(lambda: {"belief": [], "snr": [], "rssi": [], "gws": set()})
    skipped = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                key = (int(r["node_id"]), int(r["telegram_id"]))
                acc[key]["belief"].append(float(r["hmm_p_good_at_time"]))
                acc[key]["snr"].append(float(r["snr_db"]))
                acc[key]["rssi"].append(float(r["rssi_db"]))
                acc[key]["gws"].add(int(r["gateway_id"]))
            except (ValueError, KeyError):
                skipped += 1
    out = {}
    for k, v in acc.items():
        if not v["belief"]:
            continue
        out[k] = {
            "belief": sum(v["belief"]) / len(v["belief"]),
            "avg_snr": sum(v["snr"]) / len(v["snr"]),
            "min_snr": min(v["snr"]),
            "avg_rssi": sum(v["rssi"]) / len(v["rssi"]),
            "n_gateways": len(v["gws"]),
        }
    return out, skipped


def load_ack(path="dfrag_ack_log_clean.csv"):
    """Per (node, telegram): did the node receive an ACK reporting success?"""
    out, skipped = {}, 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[(int(r["node_id"]), int(r["telegram_id"]))] = int(r["success"])
            except (ValueError, KeyError):
                skipped += 1
    return out, skipped


def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney U), ties handled by average rank.
    No sklearn dependency."""
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    n_pos = sum(l for _, l in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sum_pos = sum(rk for rk, (_, l) in zip(ranks, pairs) if l == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    uplink, sk_u = load_uplink()
    acks, sk_a = load_ack()

    print(f"uplink telegrams:  {len(uplink):,}  (skipped {sk_u} bad rows)")
    print(f"ACK outcomes:      {len(acks):,}  (skipped {sk_a} bad rows)")

    joined = [(uplink[k], acks[k]) for k in uplink.keys() & acks.keys()]
    print(f"joined on (node, telegram): {len(joined):,}")

    if len(joined) < 100:
        print("\nToo few joined records to draw any conclusion.")
        print("The two logs may not share a telegram_id space -- check whether")
        print("telegram_id is per-node-sequential in both files.")
        return

    n_pos = sum(l for _, l in joined)
    print(f"  ACK succeeded: {n_pos:,} ({100*n_pos/len(joined):.1f}%)")
    print(f"  ACK failed:    {len(joined)-n_pos:,}")

    if n_pos == 0 or n_pos == len(joined):
        print("\nAll outcomes identical -- AUC undefined. No discrimination possible.")
        return

    print()
    print(f"{'predictor':<14}{'AUC':>8}   interpretation")
    print("-" * 52)
    results = {}
    for name in ("belief", "avg_snr", "min_snr", "avg_rssi", "n_gateways"):
        a = auc([u[name] for u, _ in joined], [l for _, l in joined])
        results[name] = a
        # AUC below 0.5 means the predictor is informative but inverted,
        # so report distance from chance in either direction.
        strength = abs(a - 0.5)
        verdict = ("no signal" if strength < 0.03 else
                   "weak" if strength < 0.08 else
                   "moderate" if strength < 0.15 else "strong")
        arrow = "" if a >= 0.5 else "  (inverted)"
        print(f"{name:<14}{a:>8.4f}   {verdict}{arrow}")

    best = max(results.values(), key=lambda x: abs(x - 0.5))
    print()
    print("=" * 52)
    if abs(best - 0.5) < 0.03:
        print("VERDICT: no usable signal.")
        print()
        print("Uplink observations do not predict downlink ACK success here.")
        print("A state-aware scheduler driven by uplink belief would be")
        print("scheduling on noise. Before building it, the system needs a")
        print("separate downlink observation process -- e.g. per-(node,")
        print("gateway) ACK receipt reports accumulated into their own belief.")
    elif abs(best - 0.5) < 0.08:
        print("VERDICT: weak signal.")
        print()
        print("Some information is present but thin. A state-aware policy")
        print("may beat RANDOM only marginally. Worth measuring the effect")
        print("size before committing to the full firmware effort.")
    else:
        print("VERDICT: usable signal.")
        print()
        print("Uplink observations do predict downlink ACK success. The")
        print("state-aware scheduling premise holds and the firmware work")
        print("is justified.")
    print("=" * 52)
    print()
    print("Reminder: this is the per-NODE test. The scheduler needs the")
    print("per-(node, gateway) version, which requires logging which gateway")
    print("relayed each ACK -- not currently recorded.")


if __name__ == "__main__":
    main()
