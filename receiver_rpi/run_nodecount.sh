#!/bin/bash
# Node-count sweep, repeated: 5 node counts x 3 reps, under interference.
# Requires the interferer board (interferer.ino) running throughout.
RUN=nodecount_$(date +%Y%m%d_%H%M); DUR=${DUR:-1200}
mkdir -p data/experiments/$RUN
for N in 2 4 6 8 10; do
  for REP in 1 2 3; do
    echo ""
    echo "=== Node count = $N, repetition $REP ==="
    echo "Power ON exactly $N nodes, power OFF the rest. Press ENTER when ready."
    read _
    timeout $DUR python3 main.py --policy state-aware \
      --run-id ${RUN} --block ${N}_${REP} \
      > data/experiments/$RUN/n${N}_r${REP}.log 2>&1
    echo "Run n=$N rep=$REP finished."
  done
done
echo "Sweep complete: data/experiments/$RUN"
