"""
receiver/ml_advanced/graph_dataset.py

Turns the flat CSV from graph_logger.py into per-telegram bipartite graphs
(gateway nodes <-> fragment nodes, edges = observations), the input format
TGAT needs. Pure Python/numpy -- no torch dependency here, so this part is
fully testable without a GPU or even without PyTorch installed.
"""

import csv
from dataclasses import dataclass, field
import numpy as np

# Edge feature order -- MUST match tgat_model.py's expected input width.
EDGE_FEATURE_NAMES = [
    "rssi_db", "snr_db", "time_since_telegram_start_s",
    "is_duplicate", "fragments_seen_before_this_norm", "telegram_progress",
]


@dataclass
class TelegramGraph:
    """One telegram's observation graph. Gateway/fragment IDs are
    remapped to local 0-indexed node indices for this graph."""
    telegram_id: int
    node_id: int
    gateway_ids: list          # local index -> real gateway_id
    fragment_ids: list         # local index -> real fragment_id
    edge_gateway_idx: np.ndarray   # [num_edges] int
    edge_fragment_idx: np.ndarray  # [num_edges] int
    edge_features: np.ndarray      # [num_edges, len(EDGE_FEATURE_NAMES)] float32
    total_fragments: int
    label: int  # 1 = telegram reconstructed, 0 = not


def load_observations(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_telegram_graphs(csv_path: str) -> list[TelegramGraph]:
    """Groups the flat observation log by (node_id, telegram_id) and
    builds one TelegramGraph per group."""
    rows = load_observations(csv_path)

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (int(row["node_id"]), int(row["telegram_id"]))
        grouped.setdefault(key, []).append(row)

    graphs = []
    for (node_id, telegram_id), group_rows in grouped.items():
        gateway_ids = sorted({int(r["gateway_id"]) for r in group_rows})
        fragment_ids = sorted({int(r["fragment_id"]) for r in group_rows})
        gw_index = {gw: i for i, gw in enumerate(gateway_ids)}
        frag_index = {fr: i for i, fr in enumerate(fragment_ids)}

        total_fragments = int(group_rows[0]["total_fragments"])

        edge_gw, edge_frag, edge_feat = [], [], []
        for r in group_rows:
            edge_gw.append(gw_index[int(r["gateway_id"])])
            edge_frag.append(frag_index[int(r["fragment_id"])])

            fragments_seen_norm = float(r["fragments_seen_before_this"]) / max(total_fragments, 1)
            telegram_progress = min(fragments_seen_norm, 1.0)

            edge_feat.append([
                float(r["rssi_db"]),
                float(r["snr_db"]),
                float(r["time_since_telegram_start_ms"]) / 1000.0,
                float(r["is_duplicate"]),
                fragments_seen_norm,
                telegram_progress,
            ])

        label = int(group_rows[-1]["telegram_reconstructed"])  # same for all rows in the group

        graphs.append(TelegramGraph(
            telegram_id=telegram_id,
            node_id=node_id,
            gateway_ids=gateway_ids,
            fragment_ids=fragment_ids,
            edge_gateway_idx=np.array(edge_gw, dtype=np.int64),
            edge_fragment_idx=np.array(edge_frag, dtype=np.int64),
            edge_features=np.array(edge_feat, dtype=np.float32),
            total_fragments=total_fragments,
            label=label,
        ))

    return graphs


def normalize_edge_features(graphs: list[TelegramGraph]) -> tuple[list[TelegramGraph], dict]:
    """Z-score normalize RSSI/SNR/time columns across the whole dataset
    (duplicate/progress columns are already bounded [0,1], left alone).
    Returns the normalized graphs plus the {mean, std} used, so the SAME
    values can be applied at inference time on the RPi4."""
    if not graphs:
        return graphs, {}

    all_feats = np.concatenate([g.edge_features for g in graphs], axis=0)
    cols_to_norm = [0, 1, 2]  # rssi_db, snr_db, time_since_telegram_start_s

    stats = {}
    for col in cols_to_norm:
        mean = float(all_feats[:, col].mean())
        std = float(all_feats[:, col].std()) or 1.0
        stats[EDGE_FEATURE_NAMES[col]] = {"mean": mean, "std": std}
        for g in graphs:
            g.edge_features[:, col] = (g.edge_features[:, col] - mean) / std

    return graphs, stats
