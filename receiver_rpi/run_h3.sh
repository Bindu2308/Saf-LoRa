#!/bin/bash
# H3: increasingly restrictive per-gateway ACK quota (ACKs per 60 s).
RUN=h3_$(date +%Y%m%d_%H%M); DUR=${DUR:-900}; POLICY=${POLICY:-state-aware}
mkdir -p data/experiments/$RUN
for REP in 1 2; do
  for Q in 0 30 15 8 4 2; do
    echo "$REP,$Q,$(date +%s)" >> data/experiments/$RUN/quota_runs.csv
    timeout -s INT $DUR python3 main.py --policy $POLICY --run-id $RUN --block $REP \
      --ack-quota $Q --quota-window 60 > data/experiments/$RUN/q${Q}_r${REP}.log 2>&1
    sleep 60
  done
done
