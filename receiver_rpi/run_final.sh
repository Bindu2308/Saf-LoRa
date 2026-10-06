#!/bin/bash
set -u
RUN="final12_$(date +%Y%m%d_%H%M%S)"
DUR="${DUR:-240}"; GAP="${GAP:-30}"; BLOCKS="${BLOCKS:-12}"
POLICIES="state-aware ewma round-robin snr-greedy"
mkdir -p "data/experiments/$RUN"
cp ~/PREREG.txt "data/experiments/$RUN/" 2>/dev/null
echo "RUN=$RUN DUR=$DUR GAP=$GAP BLOCKS=$BLOCKS" | tee "data/experiments/$RUN/META.txt"
echo "POLICIES=$POLICIES" | tee -a "data/experiments/$RUN/META.txt"
for B in $(seq 1 "$BLOCKS"); do
  SEED=$((B*1009))
  ORDER=$(python3 -c "
import random
p='$POLICIES'.split(); random.Random($SEED).shuffle(p); print(' '.join(p))")
  echo "block=$B seed=$SEED order=$ORDER" | tee -a "data/experiments/$RUN/block_order.txt"
  for P in $ORDER; do
    echo "START b=$B p=$P $(date)" | tee -a "data/experiments/$RUN/run_status.txt"
    timeout -s INT "$DUR" python3 main.py --policy "$P" --run-id "$RUN" \
      --block "$B" --ack-quota 0 --scheduler-seed "$((B*100))" \
      > "data/experiments/$RUN/b${B}_${P}.log" 2>&1
    echo "END   b=$B p=$P rc=$? $(date)" | tee -a "data/experiments/$RUN/run_status.txt"
    sleep "$GAP"
  done
done
echo "COMPLETE $RUN"
