import math
from dataclasses import dataclass

@dataclass
class HMMParams:
    p_good_to_good: float = 0.85
    p_bad_to_bad: float = 0.60
    good_snr_mean: float = 12.5
    good_snr_std: float = 1.5
    bad_snr_mean: float = 5.0
    bad_snr_std: float = 4.0
    p_lost_given_good: float = 0.70
    p_lost_given_bad: float = 0.90

def _gaussian_pdf(x, mean, std):
    if std <= 0:
        std = 0.01
    coef = 1.0 / (std * math.sqrt(2 * math.pi))
    return coef * math.exp(-((x - mean) ** 2) / (2 * std * std))

class ChannelHMM:
    def __init__(self, params: HMMParams = None):
        self.params = params or HMMParams()
        self.p_good = 0.75

    def update(self, quality_db) -> float:
        p = self.params
        p_bad = 1.0 - self.p_good
        pred_good = self.p_good * p.p_good_to_good + p_bad * (1 - p.p_bad_to_bad)
        pred_bad = self.p_good * (1 - p.p_good_to_good) + p_bad * p.p_bad_to_bad
        if quality_db is None:
            like_good = p.p_lost_given_good
            like_bad = p.p_lost_given_bad
        else:
            like_good = (1 - p.p_lost_given_good) * _gaussian_pdf(quality_db, p.good_snr_mean, p.good_snr_std)
            like_bad = (1 - p.p_lost_given_bad) * _gaussian_pdf(quality_db, p.bad_snr_mean, p.bad_snr_std)
        unnorm_good = pred_good * like_good
        unnorm_bad = pred_bad * like_bad
        total = unnorm_good + unnorm_bad
        if total > 0:
            self.p_good = unnorm_good / total
        return self.p_good
