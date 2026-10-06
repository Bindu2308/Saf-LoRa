#!/bin/bash
cd ~/isac-lora-receiver/receiver

run() {
    local label=$1; shift
    local mins=$1; shift
    echo ""
    echo "=========================================="
    echo "  $label   (${mins} min)   $(date +%H:%M:%S)"
    echo "=========================================="
    timeout ${mins}m python3 main.py "$@" 2>&1 | tee sweep_$label.log
    mkdir -p sweep/$label
    cp data/training/*.csv sweep/$label/ 2>/dev/null
    grep "METRICS:" sweep_$label.log | tail -1
    sleep 5
}

echo "SWEEP STARTED $(date +%H:%M:%S)"

# ---------- PHASE 1: policy comparison, 3 rounds (~2h) ----------
echo ""
echo "##### PHASE 1: POLICY COMPARISON (interferer OFF) #####"
for r in 1 2 3; do
  run P${r}_broadcast   8 --policy broadcast   --scheduler-seed 42
  run P${r}_stateaware  8 --policy state-aware --scheduler-seed 42
  run P${r}_ewma        8 --policy ewma        --scheduler-seed 42
  run P${r}_rr          8 --policy round-robin --scheduler-seed 42
  run P${r}_random      8 --policy random      --scheduler-seed $((42+r))
done

# ---------- PHASE 2: budget sweep (~30m) ----------
echo ""
echo "##### PHASE 2: FEEDBACK BUDGET SWEEP (interferer OFF) #####"
for duty in 0.10 0.05 0.02 0.01 0.005 0.002; do
  tag=$(echo $duty | tr -d '.')
  run B_$tag 5 --policy state-aware --duty-limit $duty
done

# ---------- PHASE 3: interference, 3 rounds (~1h) ----------
echo ""
echo "############################################"
echo "  PHASE 3 STARTING IN 60 SECONDS"
echo "  >>> POWER ON THE INTERFERER NOW <<<"
echo "  >>> Confirm interferer_log.txt is capturing <<<"
echo "############################################"
sleep 60
for r in 1 2 3; do
  run I${r}_stateaware 20 --policy state-aware
done

echo ""
echo "############################################"
echo "  PHASE 3 DONE -- POWER OFF THE INTERFERER"
echo "  >>> SAVE interferer_log.txt NOW <<<"
echo "  Phase 4 needs node reflashing. Waiting 3 min."
echo "############################################"
sleep 180

# ---------- PHASE 4: D-FRAG baselines (~1.5h) ----------
echo ""
echo "##### PHASE 4: D-FRAG BASELINES #####"
echo ">>> Flash all 10 nodes with DFRAG_POLICY DFRAG_FIXED <<<"
echo ">>> Starting in 10 minutes -- reflash now <<<"
sleep 600
run D_fixed 30 --policy state-aware

echo ""
echo ">>> Flash all 10 nodes with DFRAG_POLICY DFRAG_RANDOM <<<"
echo ">>> Starting in 10 minutes <<<"
sleep 600
run D_random 30 --policy state-aware

echo ""
echo ">>> Flash all 10 nodes with DFRAG_POLICY DFRAG_RR <<<"
echo ">>> Starting in 10 minutes <<<"
sleep 600
run D_rr 30 --policy state-aware

echo ""
echo "=========================================="
echo "  SWEEP COMPLETE  $(date +%H:%M:%S)"
echo "  Restore DFRAG_EXP3 and reflash nodes."
echo "=========================================="
ls sweep/
