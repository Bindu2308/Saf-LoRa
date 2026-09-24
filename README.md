# SAF-LoRa: State-Aware Feedback Scheduling for Split-Telegram LoRa Networks

This repository contains the paper source, simulation code, hardware firmware,
receiver software, and raw experimental data for SAF-LoRa.

## Structure

```
paper/          Final, compile-verified IEEEtran LaTeX source and figures
simulation/     Python discrete-event simulator: channel model, HMM,
                hard-decision verification, ACK scheduler, benchmark comparisons
firmware/       ESP32 firmware: transmitters (SX1262/SX1276), gateways,
                D-FRAG bandit, controlled-interference source
receiver/       Raspberry Pi 4 network server (main.py)
data/           Raw captured hardware data (interferer transition logs,
                per-observation belief series) used for the HMM ground-truth
                correlation analysis
docs/           Hardware setup and campaign procedure documentation
```

## Paper status

`paper/main.tex` compiles cleanly under `IEEEtran` (journal class) with zero
errors, zero undefined references/citations, and zero duplicate labels, as of
the last verification pass in this repository's history.

## Key results reproduced in `simulation/`

- **Proposition 1 (hard-decision availability limit):** `hard_decision.py` +
  `run_hard_decision_verification.py` -- zero strategy-outcome mismatches
  across 10-60 simulated nodes, independently confirmed with a separate
  10,000-observation run.
- **Scheduling mechanism:** `ack_scheduler_sim.py` -- the same 5-policy
  scheduler logic validated on hardware, driven by the same Gilbert-Elliott
  channel and HMM belief model.
- **Premise validation:** `auc_uplink_predicts_downlink.py` -- AUC = 0.619,
  uplink belief predicting downlink acknowledgment delivery.
- **Internal policy ablation:** `benchmark_layer_b.py` -- SAF-LoRa vs.
  Random / Round-Robin / SNR-Greedy / EWMA-Success / Belief-Only / Oracle.
  **Not included in the paper** -- see note below.

## Important note on `benchmark_layer_b.py`

This script's `state-aware` policy uses `FusedStateAwarePolicy`, which learns
from genuine simulated outcomes. This is a **hypothetical future-work
mechanism**, not the system described in the paper's Algorithm 2, Fig. 3, and
Limitations -- those correctly state that the deployed downlink belief is
uplink-primed only and does not learn from confirmed delivery, since nodes do
not currently report acknowledgment receipt upstream. Do not present this
script's results as describing the implemented system without that
distinction stated explicitly.

The paper's actual "vs. other work" comparison is the qualitative,
fully-sourced Table I (`\label{tab:related-work-comparison}` in `main.tex`),
not this internal ablation.

## Reproducing the simulation results

```bash
cd simulation
python3 run_hard_decision_verification.py
python3 benchmark_layer_b.py   # internal ablation only, see note above
```

Dependencies: `numpy`. No other external packages required for the
simulation code.
