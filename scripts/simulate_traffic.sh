#!/bin/bash
# Sends a steady stream of prediction requests to the API, so the
# Grafana dashboard has real, moving data to show during a demo
# instead of flat lines. Run this in a separate terminal a minute or
# two before showing Grafana, and leave it running in the background.
# Stop it with Ctrl+C when done.
#
# Both payloads below are real rows pulled from the training data
# (one label=0, one label=1), alternated so the "Predictions by
# Outcome" panel shows both categories.

NORMAL_ROW='{"dur": 0.12147799879312515, "proto": "tcp", "service": "-", "state": "FIN", "spkts": 6, "dpkts": 4, "sbytes": 258, "dbytes": 172, "rate": 74.08748626708984, "sload": 14158.9423828125, "dload": 8495.365234375, "sloss": 0, "dloss": 0, "sinpkt": 24.29560089111328, "dinpkt": 8.375, "sjit": 30.177547454833984, "djit": 11.83060359954834, "swin": 255, "dwin": 255, "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 43, "dmean": 43, "trans_depth": 0, "response_body_len": 0, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1}'

ATTACK_ROW='{"dur": 9.000000318337698e-06, "proto": "ddp", "service": "-", "state": "INT", "spkts": 2, "dpkts": 0, "sbytes": 200, "dbytes": 0, "rate": 111111.109375, "sload": 88888888.0, "dload": 0.0, "sloss": 0, "dloss": 0, "sinpkt": 0.008999999612569809, "dinpkt": 0.0, "sjit": 0.0, "djit": 0.0, "swin": 0, "dwin": 0, "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 100, "dmean": 0, "trans_depth": 0, "response_body_len": 0, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1}'

echo "Sending predictions every 2 seconds. Press Ctrl+C to stop."
echo ""

count=0
while true; do
  count=$((count + 1))
  if [ $((count % 2)) -eq 0 ]; then
    payload="$ATTACK_ROW"
  else
    payload="$NORMAL_ROW"
  fi

  response=$(curl -s -X POST http://localhost:8000/v1/predict \
    -H "Content-Type: application/json" \
    -d "$payload")

  echo "[$count] $response"
  sleep 2
done