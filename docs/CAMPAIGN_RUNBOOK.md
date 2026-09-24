# SAF-LoRa Campaign Runbook

One hardware session. Everything needed, in order.

---

## 1. Hardware

| Item | Count | Change from last time |
|---|---|---|
| Transmitter nodes | 12 | reflash (D-FRAG policy) |
| Gateways | 2 | none |
| Raspberry Pi 4 | 1 | scheduler already installed |
| **Interference source** | **1** | **new — flash `interferer_main.cpp`** |

**Interferer placement matters.** Put it close to Gateway 1 only, not
equidistant between the gateways. The experiment depends on one
gateway's links degrading while the other's stay clean. Equidistant
placement degrades both together and measures nothing about
discrimination.

---

## 2. Firmware to flash

### Interferer (1 board)
`interferer_main.cpp` — no other files needed. Open the Serial Monitor
at 115200 before the session starts and **record the wall-clock time**
when `INTERFERER,0,STARTED` appears. `millis()` is boot-relative, so
without that anchor the transitions cannot be aligned to server
timestamps.

### Transmitters (12 boards)
Replace `dfrag_bandit.h` with the new version. Your `main.cpp` files are
unchanged.

The D-FRAG policy is a compile-time constant. At the top of
`dfrag_bandit.h`:

```cpp
#define DFRAG_POLICY DFRAG_EXP3     // or DFRAG_FIXED / DFRAG_RANDOM / DFRAG_RR
```

Each node prints its policy at startup — check the Serial Monitor on at
least one board after flashing. A mis-flashed board is otherwise
invisible until analysis.

### Gateways (2 boards)
**No changes.** Already correct from the last campaign.

---

## 3. Before connecting: archive the old ACK log

```bash
cd ~/isac-lora-receiver/receiver
mv data/training/dfrag_ack_log.csv data/training/dfrag_ack_log_prescheduler.csv
```

The log gained `gateway_id` and `policy` columns. Without archiving, new
6-field rows land under the old 4-field header and the file becomes
unparseable.

---

## 4. Experiment A — policy comparison (interferer OFF)

All 12 nodes on `DFRAG_EXP3`. Interferer powered down.

```bash
cd ~/isac-lora-receiver/receiver

for pol in broadcast state-aware ewma round-robin random; do
    echo "=== $pol ==="
    python3 main.py --policy $pol --scheduler-seed 42 2>&1 | tee runA_$pol.log
    # Ctrl+C after ~45 min
    mkdir -p campaign2/A_$pol && cp data/training/*.csv campaign2/A_$pol/
done
```

Run each the same duration. `broadcast` reproduces the pre-scheduler
behaviour and is the control.

**After the first run**, confirm the log format:

```bash
head -3 data/training/dfrag_ack_log.csv
```

Expect `timestamp,node_id,telegram_id,success,gateway_id,policy`, with
**two rows per ACK under broadcast** — one per gateway. That doubling is
the airtime cost the old single-row log was hiding.

---

## 5. Experiment B — feedback-budget sweep (interferer OFF)

Best policy from A (likely `state-aware`), varying the ACK budget:

```bash
for duty in 0.10 0.05 0.02 0.01 0.005; do
    python3 main.py --policy state-aware --duty-limit $duty 2>&1 | tee runB_$duty.log
    mkdir -p campaign2/B_$duty && cp data/training/*.csv campaign2/B_$duty/
done
```

This produces the reliability-vs-feedback-budget curve. Shorter runs are
acceptable here (~20 min each) since the effect should be large.

---

## 6. Experiment C — HMM validation (interferer ON)

Power the interferer on. All 12 nodes on `DFRAG_EXP3`.

```bash
python3 main.py --policy state-aware 2>&1 | tee runC_interference.log
mkdir -p campaign2/C_interference && cp data/training/*.csv campaign2/C_interference/
```

Run at least 60 minutes so several ON/OFF cycles are captured. **Save
the interferer's serial output** — without the transition log this run
is worthless.

---

## 7. Experiment D — D-FRAG baselines (interferer OFF)

Requires reflashing all 12 nodes per policy. Three additional runs:

```bash
# Reflash all 12 with DFRAG_FIXED, then:
python3 main.py --policy state-aware 2>&1 | tee runD_fixed.log
mkdir -p campaign2/D_fixed && cp data/training/*.csv campaign2/D_fixed/

# Reflash with DFRAG_RANDOM, then:
python3 main.py --policy state-aware 2>&1 | tee runD_random.log
mkdir -p campaign2/D_random && cp data/training/*.csv campaign2/D_random/

# Reflash with DFRAG_RR, then:
python3 main.py --policy state-aware 2>&1 | tee runD_rr.log
mkdir -p campaign2/D_rr && cp data/training/*.csv campaign2/D_rr/
```

EXP3's run from Experiment A is the fourth arm. Keep the scheduler
policy fixed across all four so only the node-side policy varies.

**This is the expensive one** — 36 flash operations. If time is short,
drop it and report D-FRAG as "implemented and exercised with genuine
feedback" rather than "demonstrated to learn." The IoT-J review offers
that explicitly as the honest fallback.

---

## 8. Back up before disconnecting

```bash
# from Windows
scp -r isac-rx@<pi-ip>:~/isac-lora-receiver/receiver/campaign2 ./campaign2_backup
scp isac-rx@<pi-ip>:~/isac-lora-receiver/receiver/run*.log ./campaign2_backup/
```

Plus the interferer serial log from your laptop.

Verify the backup before powering anything down. The last campaign's
`run3` copy silently failed and produced a duplicate of `run1`.

---

## 9. Known limitation of this campaign

The downlink belief is **primed from uplink state and never learns from
actual ACK delivery**, because nodes do not report ACK receipt upstream.
`send_downlink()` returning True means a TCP write succeeded, not that
the radio transmitted or the node heard it.

Consequence: `state-aware` and `ewma` will behave similarly, since
`ewma` has no observations either. The comparison against
`round-robin`, `random`, and `broadcast` is still valid, and tests the
AUC = 0.619 finding in practice.

Adding receipt reporting needs a new uplink message type parsed by both
gateway and server. Deferred deliberately; state it as a limitation.

---

## 10. Time estimate

| Experiment | Runs | Duration | Total |
|---|---|---|---|
| A — policy comparison | 5 | 45 min | ~4 h |
| B — budget sweep | 5 | 20 min | ~2 h |
| C — HMM validation | 1 | 60 min | ~1 h |
| D — D-FRAG baselines | 3 + reflashing | 45 min | ~3 h |

Roughly 10 hours of hardware time, plus reflashing. A and C are the
highest value: A is the headline result, C is what validates the HMM the
reviewers called unsupported.
