"""Corrected jamming: degrades SNR quality (what a real interferer does
to arriving packets), not just raw loss probability."""

import numpy as np
from gilbert_elliott import GEParams


class JammedGEChannel:
    def __init__(self, params, rng, jam_severity_db=0.0):
        self.params = params
        self.rng = rng
        self.state = "GOOD"
        self.jam_severity_db = jam_severity_db

    def step(self):
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
        quality_db -= self.jam_severity_db
        corrupt_prob = 1.0 / (1.0 + np.exp((quality_db - 7.0) / 2.0))
        outcome = "CORRUPTED" if self.rng.random() < corrupt_prob else "CLEAN"
        return outcome, quality_db, self.state
