"""
isac_lora_sim/gilbert_elliott.py

Two-state (Good/Bad) Gilbert-Elliott channel model.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GEParams:
    p_good_to_bad: float = 0.04
    p_bad_to_good: float = 0.12
    loss_prob_good: float = 0.02
    loss_prob_bad: float = 0.7


class GEChannel:
    def __init__(self, params: GEParams, rng: np.random.RandomState):
        self.params = params
        self.rng = rng
        self.state = "GOOD"

    def step(self) -> tuple:
        p = self.params
        if self.state == "GOOD":
            if self.rng.random() < p.p_good_to_bad:
                self.state = "BAD"
        else:
            if self.rng.random() < p.p_bad_to_good:
                self.state = "GOOD"

        loss_prob = p.loss_prob_good if self.state == "GOOD" else p.loss_prob_bad
        if self.rng.random() < loss_prob:
            return "LOST", None, self.state

        if self.state == "GOOD":
            quality_db = self.rng.normal(12.5, 1.5)
        else:
            quality_db = self.rng.normal(5.0, 4.0)

        corrupt_prob = 1.0 / (1.0 + np.exp((quality_db - 7.0) / 2.0))
        outcome = "CORRUPTED" if self.rng.random() < corrupt_prob else "CLEAN"

        return outcome, quality_db, self.state


def distance_scaled_params(distance_m: float, base: GEParams = None) -> GEParams:
    if base is None:
        base = GEParams(p_good_to_bad=0.05, p_bad_to_good=0.3,
                         loss_prob_good=0.03, loss_prob_bad=0.5)

    coverage_radius_m = 350.0
    sharpness = 80.0
    t = 1.0 / (1.0 + np.exp(-(distance_m - coverage_radius_m) / sharpness))

    out_of_coverage_good_floor = 0.88
    out_of_coverage_bad_floor = 0.98

    loss_good = base.loss_prob_good + (out_of_coverage_good_floor - base.loss_prob_good) * t
    loss_bad = base.loss_prob_bad + (out_of_coverage_bad_floor - base.loss_prob_bad) * t

    return GEParams(
        p_good_to_bad=base.p_good_to_bad,
        p_bad_to_good=base.p_bad_to_good,
        loss_prob_good=min(loss_good, 0.90),
        loss_prob_bad=min(loss_bad, 0.97),
    )
