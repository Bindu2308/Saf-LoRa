#!/bin/bash
cd ~/isac-lora-receiver/receiver
for R in 1 2; do
  for P in state-aware round-robin; do
    echo "START $P rep$R $(date)"
    timeout -s INT 600 python3 main.py --policy "$P" --run-id clean_base_v2 \
      --block $((R*10 + $([ "$P" = state-aware ] && echo 1 || echo 2))) \
      --ack-quota 0 --scheduler-seed $((R*77))
    echo "END   $P rep$R rc=$?"
    sleep 30
  done
done
echo COMPLETE
