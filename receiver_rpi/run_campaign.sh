#!/bin/bash
cd ~/isac-lora-receiver/receiver

run() {
    local label=$1; shift
    local mins=$1; shift
    echo ""
    echo "=========================================="
    echo "  $label  (${mins} min)  $(date +%H:%M:%S)"
    echo "=========================================="
    timeout ${mins}m python3 main.py "$@" 2>&1 | tee runs_$label.log
    mkdir -p campaign2/$label
    cp data/training/*.csv campaign2/$label/ 2>/dev/null
    grep "METRICS:" runs_$label.log | tail -1
    sleep 5
}

# --- A: policy comparison, 8 min each = 40 min ---
run A_broadcast   8 --policy broadcast   --scheduler-seed 42
run A_stateaware  8 --policy state-aware --scheduler-seed 42
run A_ewma        8 --policy ewma        --scheduler-seed 42
run A_rr          8 --policy round-robin --scheduler-seed 42
run A_random      8 --policy random      --scheduler-seed 42

# --- B: budget sweep, 5 min each = 15 min ---
run B_duty050  5 --policy state-aware --duty-limit 0.05
run B_duty010  5 --policy state-aware --duty-limit 0.01
run B_duty002  5 --policy state-aware --duty-limit 0.002

echo ""
echo "=========================================="
echo "  A and B done. POWER ON THE INTERFERER."
echo "  Note the wall-clock time of INTERFERER,0,STARTED"
echo "  Press ENTER when ready for Experiment C."
echo "=========================================="
read

# --- C: HMM validation, 25 min ---
run C_interference 25 --policy state-aware

echo ""
echo "CAMPAIGN COMPLETE  $(date +%H:%M:%S)"
ls campaign2/
