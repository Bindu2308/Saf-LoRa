"""
run_power_analysis.py -- power of the hardware 8-block Wilcoxon comparison.

Model (all constants come from the paper; edit at will):
  - 8 blocks, each policy run once per block, ~134 ACKs per (block, policy) run (4299 ACKs / 32 runs)
  - true confirmed-delivery probability of the reference policy P_BASE = 0.49 (Table VII)
  - candidate policy has true probability P_BASE + delta
  - per-run noise = binomial sampling + extra run-level SD RUN_SD (chosen so that the total SD
    matches the observed 0.055-0.076 in Table VII) + optional block-shared drift SHARED_SD
  - test: two-sided Wilcoxon signed-rank on the 8 paired block differences, alpha = 0.05
    (the smallest attainable two-sided p with n=8 is 2/256 = 0.0078)
Usage: python run_power_analysis.py
"""
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

REPS, BLOCKS, N_ACK, P_BASE, ALPHA = 4000, 8, 134, 0.49, 0.05
BINOM_SD = (P_BASE * (1 - P_BASE) / N_ACK) ** 0.5
TOTAL_SD = 0.062                                           # mean of the SDs in Table VII
RUN_SD = max(0.0, (TOTAL_SD ** 2 - BINOM_SD ** 2)) ** 0.5
DELTAS = (0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15)
rng = np.random.RandomState(1)
print(f"binomial SD per run = {100*BINOM_SD:.1f} pp, extra run-level SD = {100*RUN_SD:.1f} pp "
      f"(total {100*TOTAL_SD:.1f} pp)")

def sim(delta, shared_sd):
    shared = rng.normal(0, shared_sd, (REPS, BLOCKS, 1))
    a = rng.binomial(N_ACK, np.clip(P_BASE + delta, 0, 1), (REPS, BLOCKS)) / N_ACK
    b = rng.binomial(N_ACK, P_BASE, (REPS, BLOCKS)) / N_ACK
    a = a + rng.normal(0, RUN_SD, (REPS, BLOCKS)) + shared[:, :, 0]
    b = b + rng.normal(0, RUN_SD, (REPS, BLOCKS)) + shared[:, :, 0]
    return a, b

for shared_sd, label in ((0.0, "runs independent within a block"), (0.03, "block-shared drift SD = 3 pp")):
    print(f"\n--- {label} ---")
    print("true gain | power Wilcoxon | power paired-t | mean |observed diff| (pp)")
    mde = None
    for d in DELTAS:
        a, b = sim(d, shared_sd)
        pw = np.mean([wilcoxon(a[i], b[i]).pvalue < ALPHA for i in range(REPS)])
        pt = np.mean([ttest_rel(a[i], b[i]).pvalue < ALPHA for i in range(REPS)])
        print(f"  {100*d:4.1f} pp |     {pw:5.2f}      |     {pt:5.2f}      | {100*np.mean(np.abs((a-b).mean(1))):5.2f}")
        if mde is None and pw >= 0.8 and d > 0: mde = d
    print("smallest tested gain with >= 80% Wilcoxon power:", "none in grid" if mde is None else f"{100*mde:.0f} pp")
