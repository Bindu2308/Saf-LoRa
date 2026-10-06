#!/bin/bash
# H2: 8 blocks x 4 policies, order randomized per block (seed logged).
RUN=h2_$(date +%Y%m%d_%H%M); DUR=${DUR:-1200}; GAP=${GAP:-60}; QUOTA=${QUOTA:-0}
mkdir -p data/experiments/$RUN
for B in 1 2 3 4 5 6 7 8; do
  ORDER=$(python3 -c "import random;p=['state-aware','ewma','round-robin','random'];random.Random($B*1009).shuffle(p);print(' '.join(p))")
  echo "block=$B seed=$((B*1009)) order=$ORDER" | tee -a data/experiments/$RUN/block_order.txt
  for P in $ORDER; do
    timeout -s INT $DUR python3 main.py --policy $P --run-id $RUN --block $B \
      --ack-quota $QUOTA --scheduler-seed $((B*100)) > data/experiments/$RUN/b${B}_$P.log 2>&1
    sleep $GAP
  done
done
