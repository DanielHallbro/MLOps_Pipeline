#!/bin/bash
# Paced demo runner for the Aug 11 presentation.
# Each stage prints a short narration cue, then waits for Enter before
# doing anything live - so you control pacing and can take a question
# mid-demo without the script racing ahead.
#
# Run scripts/demo_check.sh FIRST, a few minutes before presenting.
# It brings up Docker Compose AND Kubernetes/minikube/HPA and verifies
# everything including a live K8s prediction, so any problem surfaces
# before the clock starts. This script assumes that already passed.

DEMO_START=$(date +%s)

NORMAL_ROW='{"dur": 0.12147799879312515, "proto": "tcp", "service": "-", "state": "FIN", "spkts": 6, "dpkts": 4, "sbytes": 258, "dbytes": 172, "rate": 74.08748626708984, "sload": 14158.9423828125, "dload": 8495.365234375, "sloss": 0, "dloss": 0, "sinpkt": 24.29560089111328, "dinpkt": 8.375, "sjit": 30.177547454833984, "djit": 11.83060359954834, "swin": 255, "dwin": 255, "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 43, "dmean": 43, "trans_depth": 0, "response_body_len": 0, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1}'

ATTACK_ROW='{"dur": 9.000000318337698e-06, "proto": "ddp", "service": "-", "state": "INT", "spkts": 2, "dpkts": 0, "sbytes": 200, "dbytes": 0, "rate": 111111.109375, "sload": 88888888.0, "dload": 0.0, "sloss": 0, "dloss": 0, "sinpkt": 0.008999999612569809, "dinpkt": 0.0, "sjit": 0.0, "djit": 0.0, "swin": 0, "dwin": 0, "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 100, "dmean": 0, "trans_depth": 0, "response_body_len": 0, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1}'

elapsed() {
  local now=$(date +%s)
  local diff=$((now - DEMO_START))
  printf "[elapsed %dm%02ds]\n" "$((diff / 60))" "$((diff % 60))"
}

pause() {
  echo ""
  elapsed
  read -rp ">>> $1 (press Enter to continue) "
}

echo "=========================================="
echo "DEMO - Network Intrusion Detection MLOps Pipeline"
echo "=========================================="

# ------------------------------------------------------------------
# STAGE 1: Live prediction - normal vs. attack, side by side. A
# single request just returns a JSON blob with nothing to compare it
# against; seeing the model correctly tell two different traffic
# patterns apart is the actual proof it works.
# ------------------------------------------------------------------
pause "STAGE 1/6: Live prediction. Say: 'FastAPI loads whichever model MLflow has marked champion, no hardcoded path.' Watch it classify normal vs. attack traffic."
echo ""
echo "--- Normal traffic ---"
curl -s -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" -d "$NORMAL_ROW" | python3 -m json.tool
echo ""
echo "--- Attack traffic ---"
curl -s -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" -d "$ATTACK_ROW" | python3 -m json.tool

# ------------------------------------------------------------------
# STAGE 2: Live traffic + Grafana
# ------------------------------------------------------------------
pause "STAGE 2/6: Switch to the Grafana tab now. Say: 'request rate, prediction outcomes, and p95 latency, all real metrics.'"
echo ""
echo "--- Sending live traffic for 20s so the dashboard visibly moves ---"
timeout 20 bash -c '
  count=0
  while true; do
    count=$((count + 1))
    if [ $((count % 2)) -eq 0 ]; then
      payload="$2"
    else
      payload="$1"
    fi
    curl -s -X POST http://localhost:8000/v1/predict -H "Content-Type: application/json" -d "$payload" > /dev/null
    sleep 1
  done
' _ "$NORMAL_ROW" "$ATTACK_ROW"
echo "Done - Grafana panels should show the spike, with both outcome categories represented."

# ------------------------------------------------------------------
# STAGE 3: MLflow registry
# ------------------------------------------------------------------
pause "STAGE 3/6: Switch to MLflow (http://192.168.49.1:5001), open the experiment's run list table (not 'Compare Runs') and click the F1/accuracy column to sort descending. Say: 'Random Forest baseline beat every tuned attempt across both model families, real measured result.'"

# ------------------------------------------------------------------
# STAGE 4: Airflow
# ------------------------------------------------------------------
pause "STAGE 4/6: Switch to Airflow (http://localhost:8080), open the DAG graph + run history. Say: 'scheduled retraining, sequential not parallel (VM constraint), Slack alert on failure.' Not triggering a live run, a full retrain takes too long for this window."

# ------------------------------------------------------------------
# STAGE 5: Kubernetes/HPA - live scale-up only. Real timing from an
# actual run: CPU crossed the 60% target and jumped straight to 4
# replicas around the 90s mark, then held there for HPA's default
# 5-minute scale-down window - completely out of budget to watch
# live, so we stop and pivot to the README chart instead.
# ------------------------------------------------------------------
pause "STAGE 5/6: Kubernetes/HPA. Say: 'HPA watches CPU against a 60% target, minReplicas 1, maxReplicas 4.' About to launch load."

if ! kubectl get deployment fastapi > /dev/null 2>&1; then
  echo ""
  echo "!!! fastapi Deployment not found - minikube/manifests weren't set up before starting."
  echo "!!! Run: minikube start && kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml"
  echo "!!! Skipping Stage 5 live portion - fall back to the README chart only."
else
  echo ""
  echo "--- Launching in-cluster load generator ---"
  echo "$NORMAL_ROW" > /tmp/demo_payload.json
  kubectl run load-generator --image=curlimages/curl:latest --restart=Never -- sleep 3600 > /dev/null
  kubectl wait --for=condition=Ready pod/load-generator --timeout=30s > /dev/null
  kubectl cp /tmp/demo_payload.json load-generator:/tmp/payload.json

  kubectl exec load-generator -- sh -c "
    for i in \$(seq 1 20); do
      ( while true; do
          curl -sf -X POST http://fastapi:8000/v1/predict \
            -H 'Content-Type: application/json' -d @/tmp/payload.json > /dev/null 2>&1
        done ) &
    done
    sleep 150
  " > /dev/null 2>&1 &

  echo "Watching replicas + CPU live. Press any key once it's climbed enough to move on (max 150s)."
  echo ""
  WATCH_START=$(date +%s)
  while [ "$(( $(date +%s) - WATCH_START ))" -lt 150 ]; do
    REPLICAS=$(kubectl get deployment fastapi -o jsonpath='{.status.replicas}' 2>/dev/null || echo "?")
    CPU=$(kubectl get hpa fastapi-hpa -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null || echo "")
    echo "  replicas=$REPLICAS  cpu=${CPU}%"
    if read -t 5 -n 1 -s -r; then
      echo "(stopped early)"
      break
    fi
  done

  echo ""
  echo "--- Stopping load generator ---"
  kubectl delete pod load-generator --ignore-not-found --now > /dev/null 2>&1
  echo "Say: 'it'll hold here and scale back to 1 automatically after the 5-minute stabilization window - here's a full run.' Switch to the README chart now."
fi

# ------------------------------------------------------------------
# STAGE 6: CI/CD + wrap-up
# ------------------------------------------------------------------
pause "STAGE 6/6: Point at the CI badge in the README or the GitHub Actions tab. Say: 'every push builds an isolated stack, trains a dummy model, and smoke-tests the real API code path.' Then open for questions."

echo ""
elapsed
echo "Demo complete."