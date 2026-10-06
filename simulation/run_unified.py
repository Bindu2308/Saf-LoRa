"""
run_unified.py -- ONE engine (sim_core2.generate + sim_core.evaluate) for:
  U1  severity benchmark        -> Table X  (RR, SNR, SAF, Oracle, G_norm, SAF-SNR)
  U2  asymmetry sweep           -> Table XI (incl. hardware-regime dp = 0.025..0.09)
  U3  blend ablation            -> Table XV
Because U1 (sev=0) and U3 (dp=0) evaluate the SAME world with the SAME decisions,
Random/Oracle at dp=0 are identical in both tables by construction.

Usage:  NSEEDS=50 python run_unified.py          (quick test: NSEEDS=5)
Output: unified_results.csv  + printed tables
"""
import csv, os, statistics as st
from sim_core2 import generate, evaluate

NSEEDS = int(os.environ.get("NSEEDS", 50))
N, T = 10, 200
T_CRIT = 2.01 if NSEEDS >= 50 else 2.0

_world, _eval = {}, {}
def world(dp, sev, seed):
    k = (dp, sev, seed)
    if k not in _world:
        _world.clear() if len(_world) > 400 else None
        _world[k] = generate(N, T, seed, dp=dp, sev_db=sev)
    return _world[k]

def ev(dp, sev, seed, pol, lam=0.5):
    k = (dp, sev, seed, pol, lam)
    if k not in _eval:
        _eval[k] = 100 * evaluate(world(dp, sev, seed), pol, lam=lam, seed=seed + 100000)["delivery_sent"]
    return _eval[k]

def mean(x): return st.mean(x)
def ci(x): return T_CRIT * st.stdev(x) / len(x) ** 0.5
def col(dp, sev, pol, **kw): return [ev(dp, sev, s, pol, **kw) for s in range(NSEEDS)]
def diff(a, b): return [x - y for x, y in zip(a, b)]

rows = []
def log(exp, dp, sev, pol, vals, extra=""):
    for s, v in enumerate(vals):
        rows.append(dict(exp=exp, dp=dp, sev=sev, policy=pol + extra, seed=s, delivery=f"{v:.6f}"))

# ------------------------------------------------------------------ U1
print("\n=== U1 severity benchmark (Table X) ; G_norm = (D_pi - D_RR)/(D_oracle - D_RR) ===")
print("sev  |   RR    Random   SNR     SAF    EWMA  BLEND  Oracle | Gn(SAF) Gn(SNR) Gn(EWMA) Gn(BLND) | SAF-SNR   BLND-SAF  BLND-EWMA (pp)")
for sev in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
    V = {p: col(0.0, sev, p) for p in ("round-robin", "random", "snr-greedy", "belief", "ewma", "blend-paper", "oracle")}
    for p, v in V.items(): log("U1", 0.0, sev, p, v)
    m = {p: mean(v) for p, v in V.items()}
    g = lambda p: (m[p] - m["round-robin"]) / (m["oracle"] - m["round-robin"])
    d  = diff(V["belief"], V["snr-greedy"])
    db = diff(V["blend-paper"], V["belief"])
    de = diff(V["blend-paper"], V["ewma"])
    print(f"{sev:3.1f}  | {m['round-robin']:6.2f} {m['random']:6.2f} {m['snr-greedy']:6.2f} {m['belief']:6.2f} "
          f"{m['ewma']:6.2f} {m['blend-paper']:6.2f} {m['oracle']:6.2f} | {g('belief'):6.3f} {g('snr-greedy'):6.3f} "
          f"{g('ewma'):6.3f} {g('blend-paper'):6.3f} | {mean(d):+5.2f}+/-{ci(d):.2f}  {mean(db):+5.2f}+/-{ci(db):.2f}  {mean(de):+5.2f}+/-{ci(de):.2f}")

# ------------------------------------------------------------------ U2
print("\n=== U2 asymmetry sweep (Table XI). Prop.4 prediction for the increment: 25*dp pp ===")
print("dp     | gain SAF-RR     incr(SAF)  incr(Oracle)  Prop4 | SAF-Oracle")
DPS = (0.0, 0.025, 0.05, 0.075, 0.09, 0.10, 0.20, 0.30, 0.40)
G = {}
for dp in DPS:
    rr, sa, orc = col(dp, 0.0, "round-robin"), col(dp, 0.0, "belief"), col(dp, 0.0, "oracle")
    for p, v in (("round-robin", rr), ("belief", sa), ("oracle", orc)): log("U2", dp, 0.0, p, v)
    G[dp] = (diff(sa, rr), diff(orc, rr), diff(sa, orc))
xs, ys = [], []
for dp in DPS:
    inc = diff(G[dp][0], G[0.0][0]); inco = diff(G[dp][1], G[0.0][1])
    print(f"{dp:5.3f}  | {mean(G[dp][0]):5.2f} +/- {ci(G[dp][0]):.2f}   {mean(inc):5.2f}+/-{ci(inc) if dp else 0:.2f}  "
          f"{mean(inco):5.2f}        {25*dp:5.2f} | {mean(G[dp][2]):+5.2f} +/- {ci(G[dp][2]):.2f}")
    xs.append(dp); ys.append(mean(inc))
n = len(xs); sx, sy = sum(xs), sum(ys)
slope = (n * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / (n * sum(x * x for x in xs) - sx ** 2)
print(f"least-squares slope of SAF increment = {slope:.1f} pp per unit dp  ({100*slope/25:.0f}% of the predicted 25)")
print("dp definition (from sim_core.generate): stationary GOOD share of gateway 2 = PI_GOOD - dp, "
      "PI_GOOD=0.857, delivery 0.75/0.25 by state, so |b1-b2| = dp and (eps_B-eps_G)/2 = 0.25.")

# ------------------------------------------------------------------ U3
print("\n=== U3 blend ablation (Table XV); same worlds/decisions as U1 sev=0 at dp=0 ===")
print("score                  " + "  ".join(f"dp={dp:<4}" for dp in (0.0, 0.2, 0.4)))
def line(name, f):
    print(f"{name:22s} " + "  ".join(f"{f(dp):7.2f}" for dp in (0.0, 0.2, 0.4)))
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    for dp in (0.0, 0.2, 0.4): log("U3", dp, 0.0, "blend", col(dp, 0.0, "blend", lam=lam), extra=f"@{lam}")
    line(f"lambda={lam}", lambda dp, lam=lam: mean(col(dp, 0.0, "blend", lam=lam)))
for name, pol in (("paper rule n/(n+5)", "blend-paper"), ("SNR-greedy", "snr-greedy"),
                  ("Random", "random"), ("Oracle", "oracle")):
    line(name, lambda dp, pol=pol: mean(col(dp, 0.0, pol)))
    for dp in (0.0, 0.2, 0.4): log("U3", dp, 0.0, pol, col(dp, 0.0, pol))
print("paper rule - belief-only (paired, pp):")
for dp in (0.0, 0.2, 0.4):
    d = diff(col(dp, 0.0, "blend-paper"), col(dp, 0.0, "blend", lam=0.0))
    print(f"   dp={dp}: {mean(d):+.2f} +/- {ci(d):.2f}")

with open("unified_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["exp", "dp", "sev", "policy", "seed", "delivery"])
    w.writeheader(); w.writerows(rows)
print("\nSaved unified_results.csv")
