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
echo "SESSION B STARTED $(date +%H:%M:%S)  -- interferer must be ON"
for r in 1 2 3; do
  run I${r}_stateaware 20 --policy state-aware
done
echo ""; echo "SESSION B COMPLETE -- power off interferer, save the log"
