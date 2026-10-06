#!/bin/bash
cd ~/isac-lora-receiver/receiver
POL="state-aware ewma round-robin snr-greedy belief-only"

echo "=== 1. policy names accepted by CLI ==="
python3 main.py --help 2>&1 | grep -iE 'state-aware|ewma|round-robin|snr-greedy|belief-only|policy'

echo; echo "=== 2. 30 s smoke per policy ==="
for P in $POL; do
  rm -rf "data/experiments/pf_$P"
  timeout -s INT 30 python3 main.py --policy "$P" --run-id "pf_$P" \
    --block 0 --ack-quota 0 --scheduler-seed 999 > "/tmp/pf_$P.log" 2>&1
  echo "  $P rc=$?"
done

echo; echo "=== 3. rows written ==="
for P in $POL; do
  O="data/experiments/pf_$P/outcomes.csv"; D="data/experiments/pf_$P/decisions.csv"
  if [ -f "$O" ]; then
    printf "  %-12s decisions=%-4s outcomes=%-4s  " "$P" \
      "$(($(wc -l < "$D")-1))" "$(($(wc -l < "$O")-1))"
    awk -F, 'NR>1{print $NF}' "$O" | sort | uniq -c | tr '\n' ' '; echo
  else
    echo "  $P  NO OUTPUT  <-- FAIL"
  fi
done

echo; echo "=== 4. real errors (outcome 'failed' excluded) ==="
grep -nE 'Traceback|ERROR|Exception|queue full|stale ACK|send_downlink failed|disconnect|refused' \
  /tmp/pf_*.log | grep -v ',failed' | head -40 || echo "  none"

echo; echo "=== 5. policy actually recorded (must match the run) ==="
for P in $POL; do
  echo "  $P -> $(awk -F, 'NR>1{print $6}' "data/experiments/pf_$P/outcomes.csv" 2>/dev/null | sort -u | tr '\n' ' ')"
done
