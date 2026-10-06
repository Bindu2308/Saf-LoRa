#!/bin/bash
cd ~/isac-lora-receiver/receiver
run() {
    local label=$1; shift; local mins=$1; shift
    echo ""; echo "=== $label (${mins}m) $(date +%H:%M:%S) ==="
    timeout ${mins}m python3 main.py "$@" 2>&1 | tee sweep_$label.log
    mkdir -p sweep/$label && cp data/training/*.csv sweep/$label/ 2>/dev/null
    grep "METRICS:" sweep_$label.log | tail -1
    sleep 5
}
echo "SESSION A STARTED $(date +%H:%M:%S)  -- interferer must be OFF"
for r in 1 2 3; do
  run P${r}_broadcast   8 --policy broadcast   --scheduler-seed 42
  run P${r}_stateaware  8 --policy state-aware --scheduler-seed 42
  run P${r}_ewma        8 --policy ewma        --scheduler-seed 42
  run P${r}_rr          8 --policy round-robin --scheduler-seed 42
  run P${r}_random      8 --policy random      --scheduler-seed $((42+r))
done
for duty in 0.10 0.05 0.02 0.01 0.005 0.002; do
  run B_$(echo $duty | tr -d '.') 5 --policy state-aware --duty-limit $duty
done
echo ""; echo "SESSION A COMPLETE $(date +%H:%M:%S)"
