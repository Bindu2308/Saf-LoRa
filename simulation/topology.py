import numpy as np
from dataclasses import dataclass

@dataclass
class Topology:
    area_size_m: float
    node_positions: np.ndarray
    gateway_positions: np.ndarray
    def distance(self, node_idx: int, gateway_idx: int) -> float:
        return float(np.linalg.norm(self.node_positions[node_idx] - self.gateway_positions[gateway_idx]))

def make_topology(num_nodes=10, num_gateways=2, area_size_m=1300.0, seed=42):
    rng = np.random.RandomState(seed)
    node_positions = rng.uniform(0, area_size_m, size=(num_nodes, 2))
    if num_gateways == 2:
        gateway_positions = np.array([[area_size_m*0.2, area_size_m*0.2],[area_size_m*0.8, area_size_m*0.8]])
    else:
        angles = np.linspace(0, 2*np.pi, num_gateways, endpoint=False)
        cx, cy = area_size_m/2, area_size_m/2
        r = area_size_m*0.3
        gateway_positions = np.stack([cx+r*np.cos(angles), cy+r*np.sin(angles)], axis=1)
    return Topology(area_size_m=area_size_m, node_positions=node_positions, gateway_positions=gateway_positions)
