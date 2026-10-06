# ISAC-LoRa TGAT + PPO (Advanced ML Layer)

This is the complex ML system (Temporal Graph Attention Network + PPO,
plus XGBoost/LSTM/DQN/non-ML baselines) discussed as a scaled-up
alternative to the working HMM+Gradient Boosting system in `receiver/ml/`.

**This does NOT replace `receiver/ml/`.** Both can coexist; `ml/`
continues running live on the RPi4 as-is. This folder is a separate,
larger undertaking that trains offline (in Colab) and deploys for
inference only once trained.

## What's genuinely tested vs. not

Being upfront about this, since it matters for how much to trust each
piece before a long training run:

**Tested, verified correct (pure Python/numpy, no GPU needed):**
- `graph_logger.py` -- verified with a synthetic multi-gateway telegram, duplicate detection confirmed correct
- `graph_dataset.py` -- verified graph construction against real logged data structure
- `baselines/non_ml_baselines.py` -- verified against real data, results match expected ground truth

**Written carefully but NOT executable in the dev sandbox this project
was built in (no internet access there to install PyTorch/stable-baselines3):**
- `tgat_model.py`, `tgat_pretrain.py`, `ppo_env.py`, `ppo_train.py`,
  `baselines/xgboost_baseline.py`, `baselines/lstm_baseline.py`, `baselines/dqn_baseline.py`

For the untested pieces: **run `smoke_test.py` FIRST in Colab, always**,
before any real training run. It builds tiny synthetic graphs, runs a
forward+backward pass through the full TGAT model, and checks gradients
actually flow -- catches any real bug in under a minute, before you
waste 30+ minutes discovering the same bug mid-training-run.

## Step-by-step: what to actually do

### 1. Collect data (already happening)
Your RPi4's `receiver/main.py` has been logging `reliability_examples.csv`
via `ml/training_logger.py` this whole time. It does NOT yet log the
finer-grained `graph_observations.csv` that TGAT needs -- for that, add
`graph_logger.py`'s integration (3 lines, documented at the bottom of
that file) to `main.py`, then let the receiver run to accumulate real
dual-gateway telegram data.

**Minimum data needed:** the pretraining/PPO scripts refuse to run below
~50 telegrams (configurable via `--min-graphs`) -- below that, results
would be unreliable. More is always better; aim for several hundred if
you can, matching what worked well for the GB baseline (597 examples,
90% test accuracy).

### 2. Pull the data off the RPi4
```bash
scp isac-rx@<RPI4_IP>:~/isac-lora-receiver/receiver/data/training/graph_observations.csv .
scp isac-rx@<RPI4_IP>:~/isac-lora-receiver/receiver/data/training/reliability_examples.csv .
```

### 3. Train in Google Colab
Open `ISAC_LoRa_TGAT_PPO_Training.ipynb` in
[colab.research.google.com](https://colab.research.google.com), set
**Runtime > Change runtime type > T4 GPU** (free tier), and run the
cells in order. Upload your two CSVs and a zip of this `ml_advanced/`
folder when prompted.

### 4. Deploy back to the RPi4
Download `tgat_encoder.pt` and `ppo_policy.zip` from Colab at the end,
copy both into `receiver/ml_advanced/models/` on the RPi4:
```bash
mkdir -p ~/isac-lora-receiver/receiver/ml_advanced/models
scp tgat_encoder.pt ppo_policy.zip isac-rx@<RPI4_IP>:~/isac-lora-receiver/receiver/ml_advanced/models/
```
Then install the (small, inference-only) dependencies on the RPi4:
```bash
pip3 install torch stable-baselines3 --break-system-packages
```
`inference.py` provides `TGATPPOInference`, ready to wire into
`main.py` similarly to how `ml/gradient_boosting.py` is already wired in
-- not done automatically here, since `main.py` has been edited enough
times this session that a deliberate, reviewed change is safer than
another automated rewrite.

## Honest scope notes

- **PPO's action space assumes exactly 2 gateways** (GW1/GW2/both).
  Scaling to more gateways needs a different action space (the original
  doc's own section 8 covers this) -- not built here since your current
  hardware has 2.
- **The offline replay environment uses REAL logged telegrams**, not a
  synthetic channel simulator -- deliberately, to avoid a sim-to-real
  gap. This means training diversity is limited to whatever conditions
  you've actually captured; if you only ever tested at close range, the
  policy won't have seen weak-signal scenarios to learn from.
- **The PPO/DQN encoder is frozen during RL training** (not jointly
  fine-tuned with the policy) -- a reasonable simplification for a first
  version; unfreezing it is a natural extension if results look
  promising and you want to push further.
