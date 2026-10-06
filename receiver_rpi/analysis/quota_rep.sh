#!/bin/bash
cd ~/isac-lora-receiver/receiver
for Q in 8 4 2; do for R in 1 2; do
  echo "QUOTA $Q rep$R $(date)"
  timeout -s INT 240 python3 main.py --policy state-aware --run-id quota_rep \
    --block $((Q*10+R)) --ack-quota $Q --scheduler-seed $((Q*100+R))
  sleep 30
done; done
echo COMPLETE
