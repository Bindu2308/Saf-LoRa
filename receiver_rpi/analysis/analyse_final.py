import csv, sys, itertools
from collections import defaultdict
import numpy as np
from scipy import stats

rows = list(csv.DictReader(open(sys.argv[1])))
SUCC = {'success'}; FAIL = {'failed','failure'}

S  = defaultdict(lambda: [0,0,0])     # (block,policy) -> succ, fail, unavail
GW = defaultdict(lambda: defaultdict(int))
ST = defaultdict(lambda: [0,0,0,0])   # policy -> OFFs,OFFn,ONs,ONn

for r in rows:
    b = r['block'].strip()
    if not b.isdigit(): continue
    b = int(b); p = r['policy'].strip(); o = r['outcome'].strip().lower()
    S[(b,p)][0 if o in SUCC else 1 if o in FAIL else 2] += 1
    if o in SUCC or o in FAIL:
        GW[p][r['gws'].strip()] += 1
        st = r.get('interferer_at_send','').strip().upper()
        i = 0 if st == 'OFF' else 2
        ST[p][i]   += 1 if o in SUCC else 0
        ST[p][i+1] += 1

pol = sorted({p for _,p in S}); blk = sorted({b for b,_ in S})
D = {k: v[0]/(v[0]+v[1]) for k,v in S.items() if v[0]+v[1]}

print("REPORT AVAILABILITY BY POLICY  (must be similar across policies)")
for p in pol:
    s,f,u = (sum(S[(b,p)][i] for b in blk if (b,p) in S) for i in range(3))
    print(f"  {p:<13} attempts={s+f+u:5d} confirmed={s+f:5d} unavail={u:4d}  avail={(s+f)/(s+f+u):.3f}")

print("\nPOLICY MEANS OVER BLOCKS")
for p in pol:
    v = [D[(b,p)] for b in blk if (b,p) in D]
    print(f"  {p:<13} n={len(v):2d}  mean={np.mean(v):.4f}  sd={np.std(v,ddof=1):.4f}")

print("\nGATEWAY SELECTION SHARE  (GW1 = interfered, GW2 = healthy)")
for p in pol:
    tot = sum(GW[p].values())
    sh  = {g: 100*c/tot for g,c in sorted(GW[p].items())}
    print(f"  {p:<13} " + "  ".join(f"GW{g}={v:5.1f}%" for g,v in sh.items()))

print("\nDELIVERY BY INTERFERER STATE")
for p in pol:
    offs,offn,ons,onn = ST[p]
    print(f"  {p:<13} OFF={offs/offn:.3f} (n={offn:4d})   ON={ons/onn:.3f} (n={onn:4d})"
          if offn and onn else f"  {p:<13} incomplete")

print("\nPAIRED BLOCK DIFFERENCES (pp), two-sided paired t")
for a,b_ in itertools.combinations(pol,2):
    d = np.array([100*(D[(k,a)]-D[(k,b_)]) for k in blk if (k,a) in D and (k,b_) in D])
    if len(d) < 2: continue
    hw = stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
    pv = stats.ttest_1samp(d,0).pvalue
    star = " *" if pv < 0.05 else ""
    print(f"  {a:<13}- {b_:<13} n={len(d):2d}  {d.mean():+6.2f} pp  95%CI +/-{hw:5.2f}  p={pv:.4f}{star}")

print("\nPER-BLOCK DELIVERY")
print("  block " + "".join(f"{p:>14}" for p in pol))
for b in blk:
    print(f"  {b:5d} " + "".join(f"{D[(b,p)]:>14.4f}" if (b,p) in D else f"{'-':>14}" for p in pol))
