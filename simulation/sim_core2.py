"""
sim_core2.py -- sim_core.generate() extended with a jamming severity on gateway 0
(sev_db), so the severity benchmark (Table X), the asymmetry sweep (Table XI) and the
blend ablation (Table XV) can all be produced by ONE engine (generate + evaluate).
With sev_db=0 it is bit-identical to sim_core.generate (checked in run_unified.py).
Gateway 0 here == "gateway 1" (the jammed gateway) in run_benchmark_layerb.py.
"""
import random
import numpy as np
from gilbert_elliott_jam_patch import JammedGEChannel
from hmm_estimator import ChannelHMM, HMMParams
from sim_core import (evaluate, gw_params, PI_GOOD, P_DELIV_GOOD, P_DELIV_BAD,
                      DEFAULT_L_BAD)

DELIV_LOSS_PER_DB = 0.04   # same constant as run_benchmark_layerb.py


def generate(n_nodes, T, seed, dp=0.0, l_bad=DEFAULT_L_BAD, obs_noise_db=0.0,
             matched_hmm=False, sev_db=0.0, jam_gw=0):
    rng = np.random.RandomState(seed)
    py = random.Random(seed)
    nrng = np.random.RandomState(seed + 424242)
    pi2 = max(0.05, min(0.95, PI_GOOD - dp))
    pars = (gw_params(PI_GOOD, l_bad), gw_params(pi2, l_bad))
    shape = (n_nodes, T, 2)
    snr = np.empty(shape); bel = np.empty(shape)
    belm = np.empty(shape) if matched_hmm else None
    tp = np.empty(shape); dv = np.zeros(shape, dtype=bool)
    for n in range(n_nodes):
        ch = [JammedGEChannel(pars[g], rng,
                              jam_severity_db=(sev_db if g == jam_gw else 0.0)) for g in (0, 1)]
        h = [ChannelHMM() for _ in (0, 1)]
        hm = ([ChannelHMM(HMMParams(p_good_to_good=1.0 - pars[g].p_good_to_bad,
                                    p_bad_to_bad=1.0 - pars[g].p_bad_to_good))
               for g in (0, 1)] if matched_hmm else None)
        for t in range(T):
            for g in (0, 1):
                _, q, state = ch[g].step()
                if q is not None:
                    q = q + obs_noise_db * nrng.standard_normal()
                snr[n, t, g] = q if q is not None else -99.0
                bel[n, t, g] = h[g].update(q)
                if matched_hmm:
                    belm[n, t, g] = hm[g].update(q)
                p = P_DELIV_GOOD if state == "GOOD" else P_DELIV_BAD
                if g == jam_gw:
                    p = max(0.0, min(1.0, p - DELIV_LOSS_PER_DB * sev_db))
                tp[n, t, g] = p
                dv[n, t, g] = py.random() < p
    return {"snr": snr, "bel": bel, "belm": belm, "tp": tp, "dv": dv}
