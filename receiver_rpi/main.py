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

D-FRAG downlink ACK relay -- after a telegram finalizes (success or
timeout), queues a 7-byte ACK/NACK packet for a single background
worker to send through both gateway connections. CONFIRMED WORKING on
real hardware: PACING_DELAY_S=3.0 gives the node enough time to finish
its 5-fragment transmit burst before the ACK arrives (verified: ACK
result: SUCCESS across many consecutive telegrams after this fix).
Every ACK actually sent is also logged to data/training/dfrag_ack_log.csv
(timestamp, node_id, telegram_id, success) -- the real reward signal
each node's EXP3 bandit is learning from, for later convergence analysis.

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

from scheduling.downlink_scheduler import DownlinkScheduler, POLICIES
from experiments.ack_ledger import AckLedger
from experiments.interferer_listener import InterfererListener

from config.receiver_config import (
    GATEWAY1_TCP_PORT, GATEWAY2_TCP_PORT, GATEWAY3_TCP_PORT, TELEGRAM_TIMEOUT_MS,
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

NODE_ID_MAX = 12   # highest valid node ID in this deployment

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
    def __init__(self, host: str, gw1_port: int, gw2_port: int, gw3_port: int = None,
                 downlink_policy: str = "state-aware",
                 duty_limit: float = 0.10,
                 scheduler_seed: int = None,
                 ack_quota: int = 0, quota_window: float = 60.0,
                 run_id: str = "default", block: int = 0):
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
        self._history_total = defaultdict(int)
        self._history_full = defaultdict(int)
        self._telegram_gateway_readings = defaultdict(lambda: defaultdict(lambda: {"snr": [], "rssi": []}))
        self._latest_snr = {}  # (node, gw) -> latest uplink SNR dB, all observations incl. CRC-failed
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
                csv.writer(f).writerow(["timestamp", "node_id", "telegram_id",
                                         "success", "gateway_id", "policy"])

        # Downlink ACK scheduling. Replaces the previous
        # broadcast-to-both-gateways path, which spent downlink airtime
        # twice per acknowledgment and left no gateway-selection decision
        # for state information to influence.
        self.scheduler = DownlinkScheduler(
            policy=downlink_policy,
            gateway_ids=(1, 2, 3) if gw3_port else (1, 2),
            duty_limit=duty_limit,
            seed=scheduler_seed,
            ack_quota=ack_quota,
            quota_window_s=quota_window,
        )
        log.info(f"[scheduler] downlink policy={downlink_policy} "
                 f"duty_limit={duty_limit} ack_quota={ack_quota}/{quota_window}s "
                 f"run={run_id} block={block}")
        exp_dir = os.path.join(os.path.dirname(__file__), "data", "experiments", run_id)
        self.interferer = InterfererListener(exp_dir)
        self.ledger = AckLedger(exp_dir, downlink_policy, block, self.scheduler,
                                self.interferer.state)

        self.gw1_server = GatewayServer(gw1_port, expected_gateway_id=1,
                                         on_observation=self._on_observation,
                                         conn_mgr=self.conn_mgr, host=host)
        self.gw2_server = GatewayServer(gw2_port, expected_gateway_id=2,
                                         on_observation=self._on_observation,
                                         conn_mgr=self.conn_mgr, host=host)
        self.gw3_server = None
        if gw3_port:
            self.gw3_server = GatewayServer(gw3_port, expected_gateway_id=3,
                                             on_observation=self._on_observation,
                                             conn_mgr=self.conn_mgr, host=host)

        self._stop = threading.Event()

    def start(self):
        self.gw1_server.start()
        self.gw2_server.start()
        if self.gw3_server: self.gw3_server.start()
        threading.Thread(target=self._timeout_sweep_loop, daemon=True).start()
        threading.Thread(target=self._metrics_loop, daemon=True).start()
        threading.Thread(target=self._ack_worker_loop, daemon=True).start()
        self.interferer.start()
        threading.Thread(target=self.ledger.expiry_loop, args=(self._stop,),
                         daemon=True).start()

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
                self.scheduler.last_scores = {}
                self.ledger.record_decision(node_id, telegram_id, success, [],
                    self.hmm.p_good(node_id, 1), self.hmm.p_good(node_id, 2), "stale",
                b3=self.hmm.p_good(node_id, 3) if self.gw3_server else None)
                continue

            remaining = PACING_DELAY_S - age
            if remaining > 0:
                time.sleep(remaining)

            packet = build_downlink_ack(node_id, telegram_id, success)

            servers = {1: self.gw1_server, 2: self.gw2_server}
            if self.gw3_server: servers[3] = self.gw3_server
            available = [g for g, s in servers.items() if s.has_connection()]

            chosen = self.scheduler.select(
                node_id, available,
                uplink_prior_fn=lambda n, g: self.hmm.p_good(n, g),
                uplink_snr_fn=lambda n, g: self._latest_snr.get((n, g)),
            )

            if not chosen:
                log.warn(f"no ACK sent for node={node_id} telegram={telegram_id} "
                         f"-- {self.scheduler.last_reject_reason}")
                self.ledger.record_decision(node_id, telegram_id, success, [],
                    self.hmm.p_good(node_id, 1), self.hmm.p_good(node_id, 2),
                    self.scheduler.last_reject_reason,
                    b3=self.hmm.p_good(node_id, 3) if self.gw3_server else None)
                continue

            # One row per gateway that actually transmitted, so downlink
            # airtime and per-(node, gateway) outcomes are both
            # attributable. The previous code logged one row regardless of
            # sending through both gateways, which undercounted real
            # transmissions by 2x.
            sent_gws = []
            for gw_id in chosen:
                if servers[gw_id].send_downlink(packet):
                    sent_gws.append(gw_id)
                    with open(self._dfrag_log_path, "a", newline="") as f:
                        csv.writer(f).writerow([
                            time.time(), node_id, telegram_id, int(success),
                            gw_id, self.scheduler.policy,
                        ])
                else:
                    log.warn(f"send_downlink failed on GW{gw_id} for "
                             f"node={node_id} telegram={telegram_id}")
            self.ledger.record_decision(node_id, telegram_id, success, sent_gws,
                self.hmm.p_good(node_id, 1), self.hmm.p_good(node_id, 2),
                None if sent_gws else "tcp_write_failed",
                b3=self.hmm.p_good(node_id, 3) if self.gw3_server else None)

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

        return (p_good * 2.0) * (gb_score * 2.0) / 2.0

    def _on_observation(self, obs: GatewayObservation):
        try:
            validate(obs)
        except ValidationError as e:
            log.warn(f"rejected observation from GW{obs.gateway_id}: {e}")
            return

        # NODE WHITELIST -- must run BEFORE the HMM update below, since
        # self.hmm.update() creates a table entry for whatever node_id it
        # is handed. Only nodes 1-12 exist in this deployment; any other
        # ID is either a corrupted CRC-failed header (the node_id byte is
        # covered by the same failed CRC) or an unrelated external LoRa
        # device sharing the 0x12 sync word.
        if not (1 <= obs.node_id <= NODE_ID_MAX):
            return

        self.metrics.record_observation(obs.gateway_id)

        p_good = self.hmm.update(obs.node_id, obs.gateway_id, obs.demod_confidence)
        self._seen_channels.add((obs.node_id, obs.gateway_id))
        self.graph_logger.record_observation(
            obs.node_id, obs.telegram_id, obs.gateway_id, obs.fragment_id,
            obs.total_fragments, obs.received_power_db, obs.demod_confidence, p_good,
        )

        gb_score = self.gb.predict_reliability(extract_features(
            snr_values=[obs.demod_confidence], rssi_values=[obs.received_power_db],
            hmm_p_good=p_good,
            historical_success_rate=self._current_success_rate(obs.node_id, obs.gateway_id),
        ))
        log.info(f"[ml] GW{obs.gateway_id} node={obs.node_id}: "
                 f"P(GOOD)={p_good:.3f} GB_reliability={gb_score:.3f}")

        key = (obs.node_id, obs.telegram_id)
        readings = self._telegram_gateway_readings[key][obs.gateway_id]

        # CAUSALITY FIX 2: snapshot the HMM belief as it stood BEFORE this
        # telegram contributed anything, on first sight of this
        # (telegram, gateway) pair. Reading self.hmm.p_good() at finalize
        # time instead would include telegram t's own SNR observations,
        # leaking them into the hmm_p_good feature (0.104 importance).
        if "hmm_p_good_prior" not in readings:
            readings["hmm_p_good_prior"] = self.hmm.p_good(obs.node_id, obs.gateway_id)

        readings["snr"].append(obs.demod_confidence)
        self._latest_snr[(obs.node_id, obs.gateway_id)] = obs.demod_confidence
        readings["rssi"].append(obs.received_power_db)

        # CRC-failed observations have already updated the HMM above, which
        # is their entire value. Their header fields (node_id, telegram_id,
        # fragment_id, total_fragments) are covered by the SAME failed CRC
        # and cannot be trusted -- associating on them creates phantom
        # telegram contexts that never complete and pollute training data
        # with spurious label=0 examples.
        if not obs.crc_ok:
            return

        if obs.feedback:
            self.ledger.on_feedback(obs.node_id, obs.telegram_id, obs.feedback)

        ctx = self.association.associate(obs)

        num_gateways_for_fragment = len(ctx.fragments.get(obs.fragment_id, {}))
        self.metrics.record_fragment_diversity(num_gateways_for_fragment)
        if obs.crc_ok:
            self.metrics.record_fragment_recovered()

        with ctx.finalize_lock:
            if not ctx.finalized:
                message = try_reconstruct(ctx, reliability_lookup=self._reliability_lookup)
                if message is not None:
                    ctx.finalized = True
                    self.metrics.record_telegram_reconstructed()
                    self.delivery.deliver(ctx.node_id, ctx.telegram_id, message)
                    self.ledger.on_telemetry(ctx.node_id, ctx.telegram_id,
                        message if isinstance(message, (bytes, bytearray))
                        else str(message).encode('latin1'))
                    self._log_training_examples(ctx, success=True)
                    self._broadcast_ack(ctx.node_id, ctx.telegram_id, success=True)

    def _log_training_examples(self, ctx, success: bool):
        """At finalize time, log one training example per gateway that
        reported anything for this telegram: label=1 if that gateway had
        ALL fragments on its own, 0 if only some -- free labels from data
        we already collected, no extra instrumentation needed."""
        key = (ctx.node_id, ctx.telegram_id)
        per_gateway_readings = self._telegram_gateway_readings.pop(key, {})

        self.graph_logger.finalize_telegram(ctx.node_id, ctx.telegram_id, reconstructed=success)

        for gateway_id, readings in per_gateway_readings.items():
            frags_from_this_gateway = sum(
                1 for frag_id in range(ctx.total_fragments)
                if gateway_id in ctx.fragments.get(frag_id, {})
            )
            label = 1 if frags_from_this_gateway == ctx.total_fragments else 0

            hist_key = (ctx.node_id, gateway_id)

            # CAUSALITY FIX: read history from telegrams STRICTLY BEFORE
            # this one. Previously the counters were incremented with this
            # telegram's own label before success_rate was computed, leaking
            # the label into its own highest-importance feature (0.521).
            prior_total = self._history_total[hist_key]
            prior_full = self._history_full[hist_key]
            success_rate = (prior_full / prior_total) if prior_total > 0 else 0.5

            # Use the belief snapshotted before this telegram's own
            # observations were absorbed (see CAUSALITY FIX 2 above).
            # Falls back to a neutral 0.5 only if this gateway somehow has
            # no snapshot, which shouldn't happen for a gateway that reported.
            p_good = readings.get("hmm_p_good_prior", 0.5)
            features = extract_features(
                snr_values=readings["snr"], rssi_values=readings["rssi"],
                hmm_p_good=p_good, historical_success_rate=success_rate,
            )
            self.training_logger.log_example(features, label)

            # Update history only AFTER logging, so telegram t never
            # contributes to its own feature.
            self._history_total[hist_key] += 1
            if label == 1:
                self._history_full[hist_key] += 1

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
                        self._log_training_examples(ctx, success=False)
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
    parser.add_argument("--gw3-port", type=int, default=None,
                        help="omit to run with only 2 gateways")
    parser.add_argument("--policy", default="state-aware",
                        choices=list(POLICIES),
                        help="downlink ACK scheduling policy. 'broadcast' "
                             "reproduces the pre-scheduler behaviour and is "
                             "the experimental baseline.")
    parser.add_argument("--duty-limit", type=float, default=0.10,
                        help="per-gateway downlink airtime fraction (0.10 = 10%%)")
    parser.add_argument("--scheduler-seed", type=int, default=None,
                        help="RNG seed for the random policy, for reproducible runs")
    parser.add_argument("--ack-quota", type=int, default=0,
                        help="H3: max ACKs per gateway per quota window (0 = off)")
    parser.add_argument("--quota-window", type=float, default=60.0)
    parser.add_argument("--run-id", default="default",
                        help="logs go to data/experiments/<run-id>/")
    parser.add_argument("--block", type=int, default=0)
    args = parser.parse_args()

    receiver = Receiver(args.host, args.gw1_port, args.gw2_port, args.gw3_port,
                        downlink_policy=args.policy,
                        duty_limit=args.duty_limit,
                        scheduler_seed=args.scheduler_seed,
                        ack_quota=args.ack_quota, quota_window=args.quota_window,
                        run_id=args.run_id, block=args.block)
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
