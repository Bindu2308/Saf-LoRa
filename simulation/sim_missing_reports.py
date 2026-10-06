"""
sim_missing_reports.py -- effect of non-random ACK-receipt-report loss (paper: 785 of 4299 = 18% unreported).

A report for the ACK sent in round j reaches the server with probability
    REP_GOOD  if the downlink state D was GOOD in that round,   REP_BAD if it was BAD
(interference that kills the ACK also kills the next uplink that carries the report).
Settings compared (dp = 0.2):  complete (1,1) | random loss (0.82,0.82) | informative loss (0.97,0.55)
and rho in {1.0, 0.0} (uplink predicts downlink / does not).

Outputs:
  - missing_reports_detail.csv  : one row per (rho, setting, seed, policy) with
                                   deliv / real / rep_est / rep_frac
  - missing_reports_summary.csv : the aggregated table (mean over seeds),
                                   matching what is printed to the terminal

Usage: NSEEDS=500 python3 sim_missing_reports.py
"""
import csv, os, statistics as st
import numpy as np
import sim_extensions as se
from sim_extensions import (make_world, hmm_filter, info_tables, pick, N_NODES, N_ROUNDS, PI_G, LAM,
                            Q_G, Q_B, EWMA_ALPHA, BMA_FORGET)

NSEEDS = int(os.environ.get("NSEEDS", 50))
POLS = ["random", "oracle", "snr-greedy-loss", "belief-pred", "ewma", "blend-paper", "saf-adaptive"]
SETTINGS = [("complete", 1.0, 1.0), ("random-missing", 0.82, 0.82), ("informative-missing", 0.97, 0.55)]


def run(world, rep_good, rep_bad, seed):
    status, snr, q, y_u = world["status"], world["snr"], world["q"], world["y_u"]
    belief = hmm_filter(status, snr); tabs = info_tables(status, snr, belief)
    rep_u = np.random.RandomState(seed + 9091).random((N_ROUNDS, N_NODES))   # same draws for every policy
    idx = np.arange(N_NODES); out = {}
    for pol in POLS:
        rng = np.random.RandomState(seed + 7919)
        ew = np.full((N_NODES, 2), 0.5); cnt = np.zeros((N_NODES, 2))
        logw = np.array([np.log(0.9), np.log(0.1)]); hist = []
        all_y = all_n = rep_y = rep_n = 0.0; got_q = []
        for t in range(N_ROUNDS):
            j = t - 1
            if j >= 0:
                cj, xj, yj, okj = hist[j]
                if pol == "saf-adaptive":
                    pj = np.clip(xj, 1e-3, 1 - 1e-3)
                    ll = yj[:, None] * np.log(pj) + (1 - yj[:, None]) * np.log(1 - pj)
                    logw = BMA_FORGET * logw + (ll * okj[:, None]).sum(0)
                ew[idx, cj] = np.where(okj > 0, (1 - EWMA_ALPHA) * ew[idx, cj] + EWMA_ALPHA * yj, ew[idx, cj])
                cnt[idx, cj] += okj
            b_k = belief[t]; age = t - tabs["last_t"][t]; m = tabs["m"][t]
            b_pred = np.where(tabs["last_t"][t] >= 0, m + LAM ** age * (tabs["b_obs"][t] - m), PI_G)
            p_bel = Q_B + (Q_G - Q_B) * b_pred; feats = np.stack([p_bel, ew], -1)
            if pol == "random": c = rng.randint(0, 2, N_NODES)
            elif pol == "oracle": c = pick(q[t], rng)
            elif pol == "snr-greedy-loss": c = pick(tabs["last_rd"][t], rng)
            elif pol == "belief-pred": c = pick(b_pred, rng)
            elif pol == "ewma": c = pick(ew, rng)
            elif pol == "blend-paper":
                wt = cnt / (cnt + 5); c = pick((1 - wt) * b_k + wt * ew, rng)
            else:
                wts = np.exp(logw - logw.max()); wts /= wts.sum(); c = pick(feats @ wts, rng)
            qc = q[t, idx, c]; yt = (y_u[t, idx, c] < qc).astype(float)
            okt = (rep_u[t] < np.where(qc == Q_G, rep_good, rep_bad)).astype(float)
            hist.append((c, feats[idx, c], yt, okt))
            all_y += yt.sum(); all_n += N_NODES; rep_y += (yt * okt).sum(); rep_n += okt.sum()
            got_q.append(qc)
        out[pol] = dict(deliv=100 * float(np.mean(got_q)), real=100 * all_y / all_n,
                        rep_est=100 * rep_y / max(rep_n, 1), rep_frac=100 * rep_n / all_n)
    return out


def main():
    detail_rows = []
    summary_rows = []

    for rho in (1.0, 0.0):
        print(f"\n=== rho = {rho}, dp = 0.2, {NSEEDS} seeds ===")
        print("setting              rep%  | random: true / reported-est / BIAS | delivery (%): "
              "SNRloss  belief-pred  EWMA  blend  adaptive  oracle | adaptive-EWMA (paired)")
        for name, rg, rb in SETTINGS:
            R = [run(make_world(s, dp=0.2, rho=rho), rg, rb, s) for s in range(NSEEDS)]

            for s, r in enumerate(R):
                for pol, d in r.items():
                    detail_rows.append(dict(rho=rho, setting=name, seed=s, policy=pol,
                                            deliv=f"{d['deliv']:.4f}", real=f"{d['real']:.4f}",
                                            rep_est=f"{d['rep_est']:.4f}", rep_frac=f"{d['rep_frac']:.4f}"))

            g = lambda p, k: st.mean(r[p][k] for r in R)
            d_pair = [r["saf-adaptive"]["deliv"] - r["ewma"]["deliv"] for r in R]
            ci = (1.96 if NSEEDS >= 200 else 2.01) * st.stdev(d_pair) / len(d_pair) ** 0.5

            bias = g('random', 'rep_est') - g('random', 'real')
            print(f"{name:20s} {g('random','rep_frac'):4.1f}  |  {g('random','real'):5.1f} / {g('random','rep_est'):5.1f} / "
                  f"{bias:+5.1f}      | "
                  f"{g('snr-greedy-loss','deliv'):6.2f} {g('belief-pred','deliv'):9.2f} {g('ewma','deliv'):7.2f} "
                  f"{g('blend-paper','deliv'):6.2f} {g('saf-adaptive','deliv'):8.2f} {g('oracle','deliv'):7.2f} | {st.mean(d_pair):+.2f}+/-{ci:.2f}")

            summary_rows.append(dict(
                rho=rho, setting=name, n_seeds=NSEEDS,
                report_pct=f"{g('random','rep_frac'):.2f}",
                true_delivery_pct=f"{g('random','real'):.3f}",
                reported_estimate_pct=f"{g('random','rep_est'):.3f}",
                bias_pp=f"{bias:+.3f}",
                snr_greedy_loss_pct=f"{g('snr-greedy-loss','deliv'):.3f}",
                belief_pred_pct=f"{g('belief-pred','deliv'):.3f}",
                ewma_pct=f"{g('ewma','deliv'):.3f}",
                blend_paper_pct=f"{g('blend-paper','deliv'):.3f}",
                saf_adaptive_pct=f"{g('saf-adaptive','deliv'):.3f}",
                oracle_pct=f"{g('oracle','deliv'):.3f}",
                adaptive_minus_ewma_pp=f"{st.mean(d_pair):+.3f}",
                adaptive_minus_ewma_ci95=f"{ci:.3f}",
            ))

    with open("missing_reports_detail.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader(); w.writerows(detail_rows)

    with open("missing_reports_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    print(f"\nSaved missing_reports_detail.csv ({len(detail_rows)} rows)")
    print(f"Saved missing_reports_summary.csv ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
