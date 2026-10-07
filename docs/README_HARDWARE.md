# ISAC-LoRa: Hardware Implementation

Interference-state-aware cooperative reception for split-telegram LoRa networks — real hardware testbed, validated under live radio conditions.

This README covers the **hardware side** of the project: transmitter/gateway firmware, the RPi4 network server, and deployment instructions. For the simulation environments (Python discrete-event simulator and FLoRa/OMNeT++), see `README_SIMULATION.md`.

---

## 1. Architecture Overview

```
10x Transmitter Nodes (ESP32 + SX1262 or SX1276)
        |  LoRa RF (868 MHz, SF7, BW125kHz, CR4/5, implicit header, 15B fixed length)
        v
2x Gateways (ESP32+SX1262)
        |  WiFi / TCP (length-prefixed observation records, 42B each)
        v
Raspberry Pi 4 -- Network Server
  - Fragment association + MDS (5,3) Reed-Solomon reconstruction
  - HMM interference-state estimation
  - Gradient Boosting reliability classifier
  - D-FRAG downlink ACK relay (EXP3 timing-allocation feedback)
  - TGAT + PPO gateway-selection pipeline (offline-trained)
```

Each telegram is split into `F=5` fragments using systematic Reed-Solomon erasure coding over GF(256); any `K=3` of the 5 fragments are sufficient to reconstruct the original telegram.

---

## 2. Hardware Requirements

| Component | Quantity | Notes |
|---|---|---|
| ESP32 + SX1262 module | 7 | Transmitter nodes 4–10 |
| ESP32 + SX1276 module | 3 | Transmitter nodes 1–3 (used interchangeably with SX1262 due to parts availability; see note below) |
| ESP32 + SX1262 module | 2 | Gateways 1 and 2 |
| Raspberry Pi 4 | 1 | Network server |
| WiFi access point | 1 | Gateways and RPi4 must share this network |

**Note on mixed transceiver hardware**: nodes were built using either SX1262 or SX1276 modules depending on parts availability. Both chips were configured identically (same frequency, spreading factor, bandwidth, coding rate) and are handled uniformly by RadioLib, with no protocol-level or behavioral difference between node types — the split-telegram/MDS/D-FRAG logic is chip-agnostic.

### Radio configuration (all nodes and gateways)

```
Frequency:        868.0 MHz
Bandwidth:        125 kHz
Spreading Factor: 7
Coding Rate:      4/5
Sync word:        0x12
Preamble length:  8 symbols
Header mode:      Implicit, fixed length = 15 bytes
CRC:              Enabled
```

**Important**: implicit-header mode requires every device to agree on the 15-byte packet length in advance — this includes the downlink ACK packets, which carry only 7 meaningful bytes but must be zero-padded to 15 bytes to be receivable at all. This was a real bug caught during development (see Section 6).

### SX1262 pin configuration
```
CS   = 5
DIO1 = 26
RST  = 14
BUSY = 27
```

### SX1276 pin configuration
```
MISO = 19    MOSI = 27    SCK = 5    NSS  = 18
RST  = 14    DIO0 = 26    DIO1 = 35
LED1 = 2     LED2 = 13
```

---

## 3. Repository Structure (hardware-relevant files)

```
common/
  mds_encoder.h          -- Reed-Solomon (5,3) erasure encoder, GF(256), generator=3
  dfrag_bandit.h          -- EXP3 timing-allocation bandit (D-FRAG)

transmitter_sx1262/
  main.cpp                -- flash to nodes 4-10, set NODE_ID per board

transmitter_sx1276/
  main.cpp                -- flash to nodes 1-3, set NODE_ID per board

gateway_esp32/
  gateway1.ino             -- GATEWAY_ID=1, port 5000
  gateway2.ino             -- GATEWAY_ID=2, port 5001

receiver/
  main.py                  -- network server entry point
  network/
    gateway_server.py      -- TCP listener + downlink ACK relay
    protocol.py
  association/
    telegram_table.py
    fragment_association.py
  reconstruction/
    telegram_reconstructor.py
    delivery.py
  fec/
    mds.py                 -- Reed-Solomon decoder, matches common/mds_encoder.h exactly
  hmm/
    hmm_state.py
  ml/
    gradient_boosting.py
    feature_extractor.py
    train_gb.py
    models/gradient_boosting.pkl
  ml_advanced/
    graph_logger.py
    graph_dataset.py
    tgat_model.py
    tgat_pretrain.py
    ppo_env.py
    ppo_train.py
    inference.py
    models/tgat_encoder.pt
    models/ppo_policy.zip
  data/training/
    graph_observations.csv
    reliability_examples.csv
    dfrag_ack_log.csv
```

---

## 4. Flashing the Transmitter Nodes

1. Open Arduino IDE, install the **RadioLib** library.
2. For each of the 10 nodes, create a sketch folder containing:
   - The appropriate `main.cpp` (`transmitter_sx1262` or `transmitter_sx1276`, renamed to match the sketch folder)
   - `mds_encoder.h`
   - `dfrag_bandit.h`
3. Set `NODE_ID` at the top of the file to a unique value (1–10 across all boards).
4. Select board **ESP32 Dev Module**, select the correct COM port, and upload.

Each node, once running, transmits a telegram every ~4–7 seconds (jittered by the D-FRAG bandit's selected timing configuration), splits it into 5 MDS-coded fragments, and listens for a downlink acknowledgment after transmitting.

---

## 5. Flashing the Gateways

1. Create two sketch folders, one per gateway, each containing only that gateway's `.ino` file (no `mds_encoder.h` or `dfrag_bandit.h` needed — gateways don't perform MDS or bandit logic).
2. Update `WIFI_SSID`, `WIFI_PASSWORD`, and `RPI4_HOST` (the RPi4's current IP address) at the top of each file.
3. Flash `gateway1.ino` to one board, `gateway2.ino` to the other.

**Known issue — RPi4 IP drift**: the RPi4's DHCP-assigned IP address can change between sessions, breaking the gateway's connection silently. Check the RPi4's current IP with `hostname -I` before flashing, and consider assigning it a static IP (see Section 8) to avoid re-flashing gateways after every network change.

---

## 6. Running the Network Server

```bash
cd receiver
python3 main.py
```

Optional flags:
```bash
python3 main.py --host 0.0.0.0 --gw1-port 5000 --gw2-port 5001
```

On startup, the server loads the trained Gradient Boosting model if present (falls back to a neutral reliability score otherwise — the core reconstruction pipeline never depends on a trained model being available), and begins listening for both gateway connections.

### Watch for in the logs
```
gateway connected from (...) on port 5000       <- both gateways connecting
RECONSTRUCTED (3/5 fragments)                    <- successful telegram reconstruction
DELIVERY ... "Hello isac-lora"                   <- decoded payload
ACK result: SUCCESS                              <- (on the transmitter side) D-FRAG feedback loop closed
```

---

## 7. Real Bugs Found During Development (for anyone extending this project)

These are documented here because they were non-obvious and cost real debugging time — future contributors should know about them:

1. **GF(256) generator error**: an early version of the Reed-Solomon encoder used generator element 2, which only cycles through 51 of 255 field elements for this polynomial (0x11B) rather than the full multiplicative group. Fixed to generator 3.
2. **Implicit-header length mismatch**: the downlink ACK packet is only 7 meaningful bytes, but the radio is configured for a fixed 15-byte implicit-header length. Sending or receiving with a buffer sized to 7 bytes silently fails — both sides must pad to the full 15 bytes.
3. **ACK pacing / half-duplex collision**: reconstruction succeeds as soon as 3 of 5 fragments arrive, but the transmitter keeps sending all 5 regardless. If the ACK is sent while the node is still transmitting its remaining fragments, it is unreceivable (LoRa is half-duplex) — not just delayed. A 3.0-second pacing delay before sending each ACK resolves this.
4. **Unbounded ACK queue under load**: spawning one thread per ACK allowed the queue to grow unboundedly under multi-node load, delivering ACKs minutes late. A single background worker with a staleness threshold bounds this.
5. **TGAT label leakage**: an earlier pretraining task (predict telegram-level reconstruction success) was solvable by a trivial one-line fragment-counting rule at near-identical accuracy to the trained model. Reframed to a next-observation prediction task, verified to have no such leakage (trivial baseline: 71.1%, trained model: 84.1%).

---

## 8. Recommended — Set a Static IP on the RPi4

Given repeated IP drift issues during development:

```bash
sudo nano /etc/dhcpcd.conf
```
Add:
```
interface wlan0
static ip_address=<your-desired-static-ip>/24
static routers=<your-router-ip>
```
Reboot the Pi. Update `RPI4_HOST` in both gateway `.ino` files to match, once, permanently.

---

## 9. Real Results Summary (13-node deployment)

| Metric | Value |
|---|---|
| Logged observations | 15,765 |
| Gradient Boosting test accuracy | 93% |
| TGAT accuracy vs. trivial baseline | 84.1% vs. 71.1% |
| PPO final mean reward (50K timesteps) | 6.88 (from 4.60) |
| D-FRAG logged ACK outcomes | 1,684+ |

See the project paper for full methodology, simulation results, and discussion.
