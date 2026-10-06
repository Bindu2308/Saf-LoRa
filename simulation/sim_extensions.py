"""
sim_extensions.py -- new SAF-LoRa simulation experiments E7-E11.

Self-contained, vectorised over nodes. Same model as the paper:
  * per (node, gateway) Gilbert-Elliott uplink state U, HMM parameters of Table II
    (P(G->G)=0.85, P(B->B)=0.60, SNR N(12.5,1.5) GOOD / N(5.0,4.0) BAD)
  * downlink state D coupled to U with probability rho (rho=1: D=U, as in the paper)
  * ACK delivery probability 0.75 (D GOOD) / 0.25 (D BAD)
  * 10 nodes x 200 rounds x 50 seeds; every policy sees the SAME channel
    realisation and the SAME outcome random numbers per seed (paired design).

Gateway asymmetry dp: gateway 1's stationary GOOD probability is lowered by dp.

Experiments
  E7  stale information   : uplink info (and outcome feedback) delayed by tau rounds
  E8  sparse observations : node transmits an uplink in a round with prob tx_prob
  E9  observation noise   : Gaussian noise sigma on SNR; adds HMM with matched emissions
  E10 uplink/downlink mismatch : coupling rho from 1 (identical) to 0 (independent)
  E11 opposite downlink asymmetry : uplink degrades GW1, downlink degrades GW0 (rho=0)

Policies
  random, round-robin, oracle         reference policies
  snr-greedy   last received SNR reading (the paper's SNR-greedy)
  snr-greedy-loss  last reading, with a lost/CRC-failed packet counted as SNR = -20 dB
               (fair baseline: gives SNR-greedy the same loss information the HMM uses)
  snr-mean     running mean of received SNR per link (long-run baseline)
  belief       HMM filtered belief, shared fixed parameters (the paper's SAF-LoRa)
  belief-pred  NEW: last belief propagated forward over its age toward the link's
               own long-run mean, b_hat = m + LAM^age (b - m), LAM = 0.45
  belief-matched  (E9 only) HMM whose emission std includes the measurement noise
  ewma         EWMA of confirmed outcomes (alpha = 0.2)
  blend-paper  (1-w) belief + w EWMA, w = n/(n+5)   (the paper's fixed-weight rule)
  saf-adaptive NEW: Bayesian model averaging of two delivery predictors,
               belief-pred (mapped to 0.25 + 0.5 b) and EWMA; each predictor's weight
               follows its log-likelihood on confirmed ACK outcomes (pooled over nodes,
               forgetting factor 0.98). Replaces the fixed rule w = n/(n+5).

Output: extensions_results.csv
Run:    python sim_extensions.py            (all experiments)
        python sim_extensions.py --exp E10  (one experiment)
        python sim_extensions.py --seeds 10 (quick test)
"""
import argparse
import csv
import numpy as np

# ---------------------------------------------------------------- model constants
P_GG, P_BB = 0.85, 0.60
LAM = P_GG + P_BB - 1.0                                 # mixing rate, 0.45
PI_G = (1 - P_BB) / ((1 - P_BB) + (1 - P_GG))           # 0.727
MU_G, SD_G, MU_B, SD_B = 12.5, 1.5, 5.0, 4.0
LOSS_G, LOSS_B = 0.03, 0.50                             # uplink packet loss per state
Q_G, Q_B = 0.75, 0.25                                   # ACK delivery per downlink state
EWMA_ALPHA = 0.2
LOST_SNR = -20.0                                        # reading assigned to a lost/CRC-failed packet
BMA_FORGET = 0.98                                       # saf-adaptive forgetting factor
N_NODES, N_ROUNDS = 10, 200

POLICIES = ["random", "round-robin", "oracle", "snr-greedy", "snr-greedy-loss", "snr-mean", "belief",
            "belief-pred", "ewma", "blend-paper", "saf-adaptive"]


# ---------------------------------------------------------------- channel generation
def markov_chain(rng, pi_good, shape, T):
    """Two-state chain with P(B->B)=0.60 fixed and stationary GOOD prob pi_good.
    pi_good broadcasts over the last axis (gateways). Returns bool array (T,*shape)."""
    p_bg = 1 - P_BB
    p_gb = p_bg * (1 - pi_good) / pi_good
    p_gb = np.broadcast_to(p_gb, shape)
    s = np.empty((T,) + shape, dtype=bool)
    s[0] = rng.random(shape) < np.broadcast_to(pi_good, shape)
    for t in range(1, T):
        u = rng.random(shape)
        s[t] = np.where(s[t - 1], u >= p_gb, u < p_bg)
    return s


def make_world(seed, dp=0.2, rho=1.0, tx_prob=1.0, obs_noise=0.0, dp_dn=None, flip=False):
    rng = np.random.RandomState(seed)
    shape = (N_NODES, 2)
    pi_up = np.array([PI_G, PI_G - dp])
    if dp_dn is None:
        dp_dn = dp
    pi_dn = np.array([PI_G - dp_dn, PI_G]) if flip else np.array([PI_G, PI_G - dp_dn])
    U = markov_chain(rng, pi_up, shape, N_ROUNDS)
    V = markov_chain(rng, pi_dn, shape, N_ROUNDS)
    D = np.where(rng.random(U.shape) < rho, U, V)
    tx = rng.random((N_ROUNDS, N_NODES, 1)) < tx_prob
    lost = rng.random(U.shape) < np.where(U, LOSS_G, LOSS_B)
    status = np.where(~tx, 0, np.where(lost, 1, 2))         # 0 none, 1 lost/CRC-fail, 2 SNR
    snr = np.where(U, rng.normal(MU_G, SD_G, U.shape), rng.normal(MU_B, SD_B, U.shape))
    snr = snr + rng.normal(0, obs_noise, U.shape) if obs_noise > 0 else snr
    q = np.where(D, Q_G, Q_B)
    y_u = rng.random(U.shape)                                # common outcome randomness
    return dict(status=status, snr=snr, q=q, y_u=y_u)


# ---------------------------------------------------------------- estimators
def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / s


def hmm_filter(status, snr, sd_g=SD_G, sd_b=SD_B):
    """Filtered GOOD belief after each round's observation (shared fixed parameters)."""
    T = status.shape[0]
    b = np.full(status.shape[1:], PI_G)
    out = np.empty(status.shape)
    for t in range(T):
        pred = b * P_GG + (1 - b) * (1 - P_BB)
        st, x = status[t], snr[t]
        lg = np.where(st == 0, 1.0, np.where(st == 1, LOSS_G, (1 - LOSS_G) * npdf(x, MU_G, sd_g)))
        lb = np.where(st == 0, 1.0, np.where(st == 1, LOSS_B, (1 - LOSS_B) * npdf(x, MU_B, sd_b)))
        num = pred * lg
        b = num / np.maximum(num + (1 - pred) * lb, 1e-300)
        out[t] = b
    return out


def info_tables(status, snr, belief):
    """Per round: last SNR, running SNR mean, time of last observation, belief then,
    and running mean of belief at observation times (link's long-run level)."""
    T = status.shape[0]
    shp = status.shape[1:]
    last_snr = np.full(shp, np.nan); last_rd = np.full(shp, np.nan); snr_sum = np.zeros(shp); snr_n = np.zeros(shp)
    last_t = np.full(shp, -1); b_obs = np.full(shp, PI_G)
    m_sum = np.full(shp, PI_G); m_n = np.ones(shp)          # one prior pseudo-observation
    tabs = {k: np.empty((T,) + shp) for k in ("last_snr", "last_rd", "snr_mean", "last_t", "b_obs", "m")}
    for t in range(T):
        got = status[t] == 2
        last_snr = np.where(got, snr[t], last_snr)
        last_rd = np.where(got, snr[t], np.where(status[t] == 1, LOST_SNR, last_rd))
        snr_sum += np.where(got, snr[t], 0); snr_n += got
        obs = status[t] > 0
        last_t = np.where(obs, t, last_t); b_obs = np.where(obs, belief[t], b_obs)
        m_sum += np.where(obs, belief[t], 0); m_n += obs
        tabs["last_snr"][t] = last_snr; tabs["last_rd"][t] = last_rd
        tabs["snr_mean"][t] = np.where(snr_n > 0, snr_sum / np.maximum(snr_n, 1), np.nan)
        tabs["last_t"][t] = last_t; tabs["b_obs"][t] = b_obs; tabs["m"][t] = m_sum / m_n
    return tabs


# ---------------------------------------------------------------- policy simulation
def pick(score, rng):
    """argmax over the 2 gateways, NaN = unknown, ties broken uniformly at random."""
    s = np.where(np.isnan(score), -np.inf, score)
    tie = np.isclose(s[:, 0], s[:, 1]) | (np.isinf(s[:, 0]) & np.isinf(s[:, 1]))
    c = (s[:, 1] > s[:, 0]).astype(int)
    return np.where(tie, rng.randint(0, 2, len(c)), c)


def run_policies(world, tau=0, seed=0, extra_beliefs=None, policies=POLICIES):
    status, snr, q, y_u = world["status"], world["snr"], world["q"], world["y_u"]
    belief = hmm_filter(status, snr)
    tabs = info_tables(status, snr, belief)
    beliefs = {"belief": belief}
    if extra_beliefs:
        beliefs.update(extra_beliefs)
    idx = np.arange(N_NODES)
    res = {}
    for pol in policies:
        rng = np.random.RandomState(seed + 7919)
        ew = np.full((N_NODES, 2), 0.5); cnt = np.zeros((N_NODES, 2))
        logw = np.array([np.log(0.9), np.log(0.1)])        # saf-adaptive prior: 0.9 belief, 0.1 EWMA
        hist = []                                          # (t, choice, features) for delayed feedback
        got_q, agree_n, agree_d = [], 0, 0
        for t in range(N_ROUNDS):
            # ---- apply outcome feedback from round j = t-1-tau (information delay)
            j = t - 1 - tau
            if j >= 0:
                cj, xj = hist[j]
                yj = (y_u[j, idx, cj] < q[j, idx, cj]).astype(float)
                if pol == "saf-adaptive":
                    # Bayesian model averaging: log-loss of each predictor on the confirmed outcome
                    pj = np.clip(xj, 1e-3, 1 - 1e-3)                       # (N, 2): [belief, ewma] preds
                    ll = yj[:, None] * np.log(pj) + (1 - yj[:, None]) * np.log(1 - pj)
                    logw = BMA_FORGET * logw + ll.sum(0)
                ew[idx, cj] = (1 - EWMA_ALPHA) * ew[idx, cj] + EWMA_ALPHA * yj
                cnt[idx, cj] += 1
            # ---- information available at decision time: uplink up to k = t - tau
            k = t - tau
            if k >= 0:
                b_k = beliefs.get(pol, belief)[k]
                age = t - tabs["last_t"][k]
                m = tabs["m"][k]
                b_pred = np.where(tabs["last_t"][k] >= 0, m + LAM ** age * (tabs["b_obs"][k] - m), PI_G)
                last_snr, snr_mean, last_rd = tabs["last_snr"][k], tabs["snr_mean"][k], tabs["last_rd"][k]
            else:
                b_k = np.full((N_NODES, 2), PI_G); b_pred = b_k.copy()
                last_snr = snr_mean = last_rd = np.full((N_NODES, 2), np.nan)
            p_bel = Q_B + (Q_G - Q_B) * b_pred                     # belief -> delivery probability
            feats = np.stack([p_bel, ew], -1)                        # (N,2,2) predictor outputs
            # ---- choose
            if pol == "random":
                c = rng.randint(0, 2, N_NODES)
            elif pol == "round-robin":
                c = (idx + t) % 2
            elif pol == "oracle":
                c = pick(q[t], rng)
            elif pol == "snr-greedy":
                c = pick(last_snr, rng)
            elif pol == "snr-greedy-loss":
                c = pick(last_rd, rng)
            elif pol == "snr-mean":
                c = pick(snr_mean, rng)
            elif pol in ("belief", "belief-matched"):
                c = pick(b_k, rng)
            elif pol == "belief-pred":
                c = pick(b_pred, rng)
            elif pol == "ewma":
                c = pick(ew, rng)
            elif pol == "blend-paper":
                wt = cnt / (cnt + 5)
                c = pick((1 - wt) * b_k + wt * ew, rng)
            elif pol == "saf-adaptive":
                wts = np.exp(logw - logw.max()); wts /= wts.sum()
                c = pick(feats @ wts, rng)
            else:
                raise ValueError(pol)
            hist.append((c, feats[idx, c]))
            got_q.append(q[t, idx, c])
            diff = q[t, :, 0] != q[t, :, 1]
            best = q[t].argmax(1)
            agree_n += np.sum((c == best) & diff); agree_d += np.sum(diff)
        res[pol] = (float(np.mean(got_q)), agree_n / max(agree_d, 1))
    return res


# ---------------------------------------------------------------- experiment grid
def grid():
    for dp in (0.0, 0.2):
        for tau in (0, 1, 2, 4, 8, 16):
            yield "E7", dict(dp=dp), dict(tau=tau)
    for txp in (1.0, 0.5, 0.25, 0.1):
        yield "E8", dict(dp=0.2, tx_prob=txp), dict(tau=0)
    for sig in (0.0, 1.0, 2.0, 3.0, 5.0):
        yield "E9", dict(dp=0.2, obs_noise=sig), dict(tau=0)
    for rho in (1.0, 0.75, 0.5, 0.25, 0.0):
        yield "E10", dict(dp=0.2, rho=rho), dict(tau=0)
    for dpd in (0.0, 0.1, 0.2, 0.3):
        yield "E11", dict(dp=0.2, rho=0.0, dp_dn=dpd, flip=True), dict(tau=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="run only this experiment, e.g. E10")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", default="extensions_results.csv")
    a = ap.parse_args()
    fields = ["exp", "dp", "rho", "tau", "tx_prob", "obs_noise", "dp_dn", "flip", "seed", "policy",
              "delivery", "agree"]
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        for exp, wkw, rkw in grid():
            if a.exp and exp != a.exp:
                continue
            print(f"{exp} {wkw} {rkw}", flush=True)
            for seed in range(a.seeds):
                world = make_world(seed, **wkw)
                extra, pols = None, POLICIES
                if exp == "E9":
                    s = wkw["obs_noise"]
                    extra = {"belief-matched": hmm_filter(world["status"], world["snr"],
                                                          np.hypot(SD_G, s), np.hypot(SD_B, s))}
                    pols = POLICIES + ["belief-matched"]
                res = run_policies(world, seed=seed, extra_beliefs=extra, policies=pols, **rkw)
                base = dict(exp=exp, dp=wkw.get("dp", 0.2), rho=wkw.get("rho", 1.0), tau=rkw["tau"],
                            tx_prob=wkw.get("tx_prob", 1.0), obs_noise=wkw.get("obs_noise", 0.0),
                            dp_dn=wkw.get("dp_dn", wkw.get("dp", 0.2)), flip=int(wkw.get("flip", False)),
                            seed=seed)
                for pol, (dlv, agr) in res.items():
                    wr.writerow(dict(base, policy=pol, delivery=f"{dlv:.6f}", agree=f"{agr:.6f}"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
