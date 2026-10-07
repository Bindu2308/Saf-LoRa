"""
receiver/main.py

ISAC-LoRa RPi4 receiver ("Network Server"), per the architecture doc:
  GW1/GW2 --TCP--> observation validation --> fragment association -->
  duplicate handling --> gateway observation storage --> soft/hard
  combining --> reconstruction --> application delivery.

ML layer (HMM + Gradient Boosting) is now wired in:
  - HmmStateTracker maintains a per-(node,gateway) Good/Bad interference
    belief, updated from every observation's SNR.
  - GradientBoostingReliability predicts P(this gateway fully receives
    this telegram) from HMM state + SNR/RSSI + historical success rate.
    Falls back to a neutral 0.5 score until a model is trained (see
    ml/train_gb.py) -- the receiver works identically with or without a
    trained model, per spec section 43 ("ML failure must not stop the
    PHY/network receiver").
  - hard_combiner.py uses HMM-P(GOOD) * GB-reliability as a weight
    multiplier when multiple gateways report the same fragment.
  - At telegram finalize time, one training example per gateway that
    reported anything is logged (label = did that gateway get ALL
    fragments for this telegram) -- this is how the GB model eventually
    gets real training data, with zero extra hardware instrumentation.

Run:
    python3 main.py
    python3 main.py --host 0.0.0.0 --gw1-port 5000 --gw2-port 5001
"""

import argparse
import csv
import os
import queue
import threading
import time
from collections import defaultdict

from config.receiver_config import (
    GATEWAY1_TCP_PORT, GATEWAY2_TCP_PORT, TELEGRAM_TIMEOUT_MS,
    TIMEOUT_SWEEP_INTERVAL_S, METRICS_PRINT_INTERVAL_S,
)
from network.gateway_server import GatewayServer
from network.connection_manager import ConnectionManager
from network.protocol import GatewayObservation
from observation.validator import validate, ValidationError
from association.telegram_table import TelegramTable
from association.fragment_association import FragmentAssociation
from reconstruction.telegram_reconstructor import try_reconstruct
from reconstruction.delivery import Delivery
from logging_utils.logger import get_logger
from logging_utils.metrics import Metrics
from hmm.hmm_state import HmmStateTracker
from ml.gradient_boosting import GradientBoostingReliability
from ml.feature_extractor import extract_features
from ml.training_logger import TrainingLogger
from ml_advanced.graph_logger import GraphLogger

log = get_logger("main")

DOWNLINK_MAGIC = 0xBB


def build_downlink_ack(node_id: int, telegram_id: int, success: bool) -> bytes:
    """[0xBB magic][node_id][telegram_id 4B big-endian][success flag] --
    7 bytes total, matching gateway_esp32/main.cpp's checkAndRelayDownlink()
    and the transmitter firmware's waitForAck() exactly."""
    packet = bytearray(7)
    packet[0] = DOWNLINK_MAGIC
    packet[1] = node_id & 0xFF
    packet[2] = (telegram_id >> 24) & 0xFF
    packet[3] = (telegram_id >> 16) & 0xFF
    packet[4] = (telegram_id >> 8) & 0xFF
    packet[5] = telegram_id & 0xFF
    packet[6] = 1 if success else 0
    return bytes(packet)


class Receiver:
    def __init__(self, host: str, gw1_port: int, gw2_port: int):
        self.table = TelegramTable()
        self.association = FragmentAssociation(self.table)
        self.conn_mgr = ConnectionManager()
        self.metrics = Metrics()
        self.delivery = Delivery()

        # --- ML layer ---
        self.hmm = HmmStateTracker()
        self.gb = GradientBoostingReliability()
        self.training_logger = TrainingLogger()
        self.graph_logger = GraphLogger()
        # per-(node,gateway) recent completion history, for the
        # "historical_success_rate" feature -- simple rolling counters,
        # not persisted across restarts (fine: it warms back up quickly).
        self._history_total = defaultdict(int)
        self._history_full = defaultdict(int)
        # accumulate per-telegram, per-gateway SNR/RSSI readings so we can
        # log one training example per gateway at finalize time
        self._telegram_gateway_readings = defaultdict(lambda: defaultdict(lambda: {"snr": [], "rssi": []}))
        self._seen_channels = set()  # {(node_id, gateway_id)} -- for the periodic HMM state summary

        # D-FRAG ACK relay: single bounded queue + one worker thread
        ACK_QUEUE_MAXSIZE = 50
        self._ack_queue: queue.Queue = queue.Queue(maxsize=ACK_QUEUE_MAXSIZE)

        # D-FRAG ACK outcome log -- captures the real reward signal each
        # node's EXP3 bandit is learning from (success/failure per
        # telegram), for later convergence analysis.
        self._dfrag_log_path = os.path.join(os.path.dirname(__file__), "data", "training", "dfrag_ack_log.csv")
        os.makedirs(os.path.dirname(self._dfrag_log_path), exist_ok=True)
        if not os.path.exists(self._dfrag_log_path):
            with open(self._dfrag_log_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "node_id", "telegram_id", "success"])

        self.gw1_server = GatewayServer(gw1_port, expected_gateway_id=1,
                                         on_observation=self._on_observation,
                                         conn_mgr=self.conn_mgr, host=host)
        self.gw2_server = GatewayServer(gw2_port, expected_gateway_id=2,
                                         on_observation=self._on_observation,
                                         conn_mgr=self.conn_mgr, host=host)

        self._stop = threading.Event()

    def start(self):
        self.gw1_server.start()
        self.gw2_server.start()
        threading.Thread(target=self._timeout_sweep_loop, daemon=True).start()
        threading.Thread(target=self._metrics_loop, daemon=True).start()
        threading.Thread(target=self._ack_worker_loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        self.gw1_server.stop()
        self.gw2_server.stop()

    def _broadcast_ack(self, node_id: int, telegram_id: int, success: bool):
        """D-FRAG downlink relay: queues the ACK/NACK for a single
        background worker to send (see _ack_worker_loop)."""
        try:
            self._ack_queue.put_nowait((node_id, telegram_id, success, time.time()))
        except queue.Full:
            log.warn(f"ACK queue full, dropping ACK for node={node_id} telegram={telegram_id} "
                     f"-- gateway relay is falling behind real telegram rate")

    def _ack_worker_loop(self):
        """Single worker draining the ACK queue. Only waits the REMAINING
        pacing time (not a fresh delay per item), and discards ACKs that
        have sat queued too long (the node has likely moved on)."""
        STALE_ACK_MAX_AGE_S = 8.0
        PACING_DELAY_S = 3.0
        while not self._stop.is_set():
            try:
                node_id, telegram_id, success, queued_at = self._ack_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            age = time.time() - queued_at
            if age > STALE_ACK_MAX_AGE_S:
                log.warn(f"discarding stale ACK for node={node_id} telegram={telegram_id} "
                         f"(queued {age:.1f}s ago, node has likely moved on)")
                continue

            remaining = PACING_DELAY_S - age
            if remaining > 0:
                time.sleep(remaining)

            packet = build_downlink_ack(node_id, telegram_id, success)
            sent_gw1 = self.gw1_server.send_downlink(packet)
            sent_gw2 = self.gw2_server.send_downlink(packet)
            if not (sent_gw1 or sent_gw2):
                log.warn(f"could not relay ACK for node={node_id} telegram={telegram_id} "
                         f"-- no gateway connection currently available")
            else:
                with open(self._dfrag_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([time.time(), node_id, telegram_id, int(success)])

    def _current_success_rate(self, node_id: int, gateway_id: int) -> float:
        key = (node_id, gateway_id)
        total = self._history_total[key]
        return (self._history_full[key] / total) if total > 0 else 0.5

    def _reliability_lookup(self, obs: GatewayObservation) -> float:
        """Blends HMM state belief and GB-predicted reliability into a
        single multiplier for hard_combiner, using THIS observation's
        actual SNR/RSSI (not a placeholder). Neutral (~1.0) when either
        signal is unavailable, so behavior degrades gracefully."""
        p_good = self.hmm.p_good(obs.node_id, obs.gateway_id)
        success_rate = self._current_success_rate(obs.node_id, obs.gateway_id)

        features = extract_features(
            snr_values=[obs.demod_confidence],
            rssi_values=[obs.received_power_db],
            hmm_p_good=p_good,
            historical_success_rate=success_rate,
        )
        gb_score = self.gb.predict_reliability(features)

        # Scale so 1.0 = neutral (matches plain weight_of() when both
        # signals are at their neutral midpoints).
        return (p_good * 2.0) * (gb_score * 2.0) / 2.0

    def _on_observation(self, obs: GatewayObservation):
        # --- validation ---
        try:
            validate(obs)
        except ValidationError as e:
            log.warn(f"rejected observation from GW{obs.gateway_id}: {e}")
            return

        self.metrics.record_observation(obs.gateway_id)

        # --- update HMM state for this (node, gateway) channel ---
        p_good = self.hmm.update(obs.node_id, obs.gateway_id, obs.demod_confidence)  # demod_confidence carries SNR, see gateway_esp32/main.cpp
        self._seen_channels.add((obs.node_id, obs.gateway_id))
        self.graph_logger.record_observation(
            obs.node_id, obs.telegram_id, obs.gateway_id, obs.fragment_id,
            obs.total_fragments, obs.received_power_db, obs.demod_confidence, p_good,
        )

        # Make the ML layer actually visible instead of running silently --
        # log the live state on every observation. gb_score reuses the
        # exact same computation _reliability_lookup will use for this
        # observation when combining runs later in this method, so what
        # you see logged here IS the real value influencing the decision,
        # not a separate/approximate readout.
        gb_score = self.gb.predict_reliability(extract_features(
            snr_values=[obs.demod_confidence], rssi_values=[obs.received_power_db],
            hmm_p_good=p_good,
            historical_success_rate=self._current_success_rate(obs.node_id, obs.gateway_id),
        ))
        log.info(f"[ml] GW{obs.gateway_id} node={obs.node_id}: "
                 f"P(GOOD)={p_good:.3f} GB_reliability={gb_score:.3f}")

        # --- accumulate readings for this telegram's eventual training example ---
        key = (obs.node_id, obs.telegram_id)
        readings = self._telegram_gateway_readings[key][obs.gateway_id]
        readings["snr"].append(obs.demod_confidence)
        readings["rssi"].append(obs.received_power_db)

        # --- association (also handles duplicate detection internally) ---
        ctx = self.association.associate(obs)

        num_gateways_for_fragment = len(ctx.fragments.get(obs.fragment_id, {}))
        self.metrics.record_fragment_diversity(num_gateways_for_fragment)
        if obs.crc_ok:
            self.metrics.record_fragment_recovered()

        # --- attempt reconstruction immediately (don't wait for timeout
        #     if we already have everything we need) ---
        with ctx.finalize_lock:
            if not ctx.finalized:
                message = try_reconstruct(ctx, reliability_lookup=self._reliability_lookup)
                if message is not None:
                    ctx.finalized = True
                    self.metrics.record_telegram_reconstructed()
                    self.delivery.deliver(ctx.node_id, ctx.telegram_id, message)
                    self._log_training_examples(ctx, success=True)
                    self._broadcast_ack(ctx.node_id, ctx.telegram_id, success=True)

    def _log_training_examples(self, ctx, success: bool):
        """At finalize time, log one training example per gateway that
        reported anything for this telegram: label=1 if that gateway had
        ALL fragments on its own, 0 if only some -- free labels from data
        we already collected, no extra instrumentation needed."""
        key = (ctx.node_id, ctx.telegram_id)
        per_gateway_readings = self._telegram_gateway_readings.pop(key, {})

        # write out this telegram's raw observation-level rows now that
        # the outcome is known -- separate from the per-gateway aggregate
        # loop below, which feeds ml/training_logger.py instead
        self.graph_logger.finalize_telegram(ctx.node_id, ctx.telegram_id, reconstructed=success)

        for gateway_id, readings in per_gateway_readings.items():
            frags_from_this_gateway = sum(
                1 for frag_id in range(ctx.total_fragments)
                if gateway_id in ctx.fragments.get(frag_id, {})
            )
            label = 1 if frags_from_this_gateway == ctx.total_fragments else 0

            hist_key = (ctx.node_id, gateway_id)
            self._history_total[hist_key] += 1
            if label == 1:
                self._history_full[hist_key] += 1
            success_rate = self._history_full[hist_key] / self._history_total[hist_key]

            p_good = self.hmm.p_good(ctx.node_id, gateway_id)
            features = extract_features(
                snr_values=readings["snr"], rssi_values=readings["rssi"],
                hmm_p_good=p_good, historical_success_rate=success_rate,
            )
            self.training_logger.log_example(features, label)

    def _timeout_sweep_loop(self):
        while not self._stop.is_set():
            time.sleep(TIMEOUT_SWEEP_INTERVAL_S)
            for ctx in self.table.all_contexts():
                with ctx.finalize_lock:
                    if ctx.finalized:
                        continue
                    if ctx.is_expired(TELEGRAM_TIMEOUT_MS):
                        ctx.finalized = True
                        log.warn(f"telegram node={ctx.node_id} id={ctx.telegram_id} TIMED OUT, incomplete:\n{ctx.summary()}")
                        self.metrics.record_telegram_timed_out()
                        self._log_training_examples(ctx, success=False)  # a timeout is itself a useful (label=0) example for any gateway that reported something
                        self._broadcast_ack(ctx.node_id, ctx.telegram_id, success=False)

    def _metrics_loop(self):
        while not self._stop.is_set():
            time.sleep(METRICS_PRINT_INTERVAL_S)
            log.info(f"METRICS: {self.metrics.summary()}")
            for gw_id, status in sorted(self.conn_mgr.snapshot().items()):
                log.info(f"  GW{gw_id}: connected={status.connected} "
                         f"received={status.observations_received} rejected={status.observations_rejected}")
            if self._seen_channels:
                log.info("  HMM channel states:")
                for node_id, gateway_id in sorted(self._seen_channels):
                    p_good = self.hmm.p_good(node_id, gateway_id)
                    state = "GOOD" if p_good >= 0.5 else "BAD"
                    log.info(f"    node={node_id} GW{gateway_id}: P(GOOD)={p_good:.3f} ({state})")


def main():
    parser = argparse.ArgumentParser(description="ISAC-LoRa RPi4 receiver")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gw1-port", type=int, default=GATEWAY1_TCP_PORT)
    parser.add_argument("--gw2-port", type=int, default=GATEWAY2_TCP_PORT)
    args = parser.parse_args()

    receiver = Receiver(args.host, args.gw1_port, args.gw2_port)
    receiver.start()
    log.info("ISAC-LoRa receiver running. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("shutting down...")
        receiver.stop()


if __name__ == "__main__":
    main()
