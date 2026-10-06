import csv, sys, itertools
from collections import defaultdict
import numpy as np
from scipy import stats
rows=list(csv.DictReader(open(sys.argv[1])))
for STATE in ('ON','OFF'):
    S=defaultdict(lambda:[0,0])
    for r in rows:
        b=r['block'].strip(); o=r['outcome'].strip().lower()
        if not b.isdigit() or r.get('interferer_at_send','').strip().upper()!=STATE: continue
        if o=='success': S[(int(b),r['policy'].strip())][0]+=1
        elif o in('failed','failure'): S[(int(b),r['policy'].strip())][1]+=1
    D={k:v[0]/(v[0]+v[1]) for k,v in S.items() if v[0]+v[1]>=5}
    pol=sorted({p for _,p in D}); blk=sorted({b for b,_ in D})
    print(f"\n=== INTERFERER {STATE} (blocks with >=5 confirmed ACKs) ===")
    for p in pol:
        v=[D[(b,p)] for b in blk if (b,p) in D]
        print(f"  {p:<13} n={len(v):2d} mean={np.mean(v):.4f}")
    for a,b_ in itertools.combinations(pol,2):
        d=np.array([100*(D[(k,a)]-D[(k,b_)]) for k in blk if (k,a) in D and (k,b_) in D])
        if len(d)<3: continue
        hw=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        pv=stats.ttest_1samp(d,0).pvalue
        print(f"  {a:<13}- {b_:<13} n={len(d):2d} {d.mean():+6.2f} pp CI+/-{hw:5.2f} p={pv:.4f}{' *' if pv<.05 else ''}")
