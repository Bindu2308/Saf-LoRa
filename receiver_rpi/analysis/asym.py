import csv,sys
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1])))
g=defaultdict(lambda:[0,0])
for r in rows:
    o=r['outcome'].strip().lower()
    if o not in ('success','failed','failure'): continue
    g[r['gws'].strip()][0 if o=='success' else 1]+=1
q={}
for k,(s,f) in sorted(g.items()):
    q[k]=s/(s+f); print(f"  GW {k}: delivered {s:4d} / {s+f:4d} = {q[k]:.3f}")
if len(q)>=2:
    v=sorted(q.values()); bound=100*(v[-1]-v[0])/2
    print(f"\n  |q1-q2|/2 = {bound:.1f} pp  <- ceiling for ANY gateway-selection policy")
    print("  >6 pp: go.   3-6 pp: marginal.   <3 pp: move the interferer closer to one GW.")
