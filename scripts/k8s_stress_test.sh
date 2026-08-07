#!/bin/bash
# Cold-start test for the Kubernetes + HPA piece.
# Assumes NOTHING is running yet - no docker compose stack, no
# minikube. Brings everything up from scratch, then proves the HPA
# actually scales FastAPI up under load and back down again once
# load stops. This is the thing to run before merging
# feature/kubernetes-hpa, to have real proof it works, not just "I
# watched it once."
#
# Estimated total runtime: ~15-18 minutes (dominated by the load
# window + HPA's default 5-minute scale-down stabilization).
#
# Tunable via env vars, e.g.:
#   LOAD_WORKERS=30 LOAD_DURATION=240 ./scripts/k8s_stress_test.sh

set -e  # fail fast during setup - a broken foundation invalidates the test

LOAD_WORKERS="${LOAD_WORKERS:-20}"     # concurrent request loops
LOAD_DURATION="${LOAD_DURATION:-180}"  # seconds of sustained load
SCALE_DOWN_TIMEOUT=420                 # seconds to wait for scale-back-to-1
API_IMAGE="mlops_pipeline-api:latest"

PORT_FORWARD_PID=""
LOG_FILE="/tmp/k8s_stress_log_$(date +%s).csv"
echo "elapsed_seconds,phase,replicas,cpu_percent" > "$LOG_FILE"

cleanup() {
  echo ""
  echo "--- Cleaning up background processes ---"
  if [ -n "$PORT_FORWARD_PID" ]; then
    kill "$PORT_FORWARD_PID" 2>/dev/null || true
  fi
  kubectl delete pod load-generator --ignore-not-found --now 2>/dev/null || true
}
trap cleanup EXIT

START_TIME=$(date +%s)

echo "=========================================="
echo "K8S/HPA COLD-START STRESS TEST"
echo "=========================================="

# ------------------------------------------------------------------
# STEP 1/10: Start minikube and make sure metrics-server is running,
# BEFORE Docker Compose. MLflow's port binds to minikube's own
# docker-bridge gateway IP (not 0.0.0.0/127.0.0.1 - see the comment
# in docker-compose.yml for why), which only exists once minikube's
# docker network has been created - so this has to come first or
# `docker compose up` fails with "cannot assign requested address"
# on a machine that's never run minikube before.
# ------------------------------------------------------------------
echo ""
echo "--- STEP 1/10: Starting minikube + metrics-server ---"
minikube start
minikube addons enable metrics-server

echo "Waiting for metrics-server to report real numbers..."
for i in $(seq 1 30); do
  if kubectl top nodes > /dev/null 2>&1; then
    echo "metrics-server is reporting."
    break
  fi
  echo "  Waiting... ($i/30)"
  sleep 5
done

# ------------------------------------------------------------------
# STEP 2/10: Bring up the full Docker Compose stack fresh.
# The FastAPI pod depends on MLflow being reachable over HTTP, so
# this has to be healthy before anything Kubernetes-side matters.
# ------------------------------------------------------------------
echo ""
echo "--- STEP 2/10: Bringing up Docker Compose stack ---"
docker compose up -d --build

echo "Waiting for MLflow to be healthy..."
for i in $(seq 1 30); do
  if curl -sf http://192.168.49.1:5001/health > /dev/null 2>&1; then
    echo "MLflow healthy."
    break
  fi
  echo "  Waiting... ($i/30)"
  sleep 3
done

# ------------------------------------------------------------------
# STEP 3/10: Confirm a champion model actually exists. If it doesn't,
# a new K8s pod will boot, fail to load a model, and crash-loop -
# and that failure would have nothing to do with Kubernetes itself.
# ------------------------------------------------------------------
echo ""
echo "--- STEP 3/10: Confirming a champion model exists ---"
CHAMPION_INFO=$(docker compose run --rm training python3 -c "
import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri('http://mlflow:5000')
client = MlflowClient()
try:
    champion = client.get_model_version_by_alias('network-intrusion-detector', 'champion')
    print(f'OK: version {champion.version}')
except Exception as e:
    print(f'MISSING: {e}')
" 2>/dev/null | tail -1)
echo "$CHAMPION_INFO"
if [[ "$CHAMPION_INFO" == MISSING* ]]; then
  echo "No champion model set - run scripts/rebuild_pipeline.sh first."
  exit 1
fi

# ------------------------------------------------------------------
# STEP 4/10: Build the API image directly into minikube's own Docker
# daemon (imagePullPolicy: Never means it must already be there -
# minikube can't pull from anywhere else).
# ------------------------------------------------------------------
echo ""
echo "--- STEP 4/10: Building API image into minikube ---"
eval "$(minikube docker-env)"
docker build -t "$API_IMAGE" -f Dockerfile.api .
eval "$(minikube docker-env -u)"

# ------------------------------------------------------------------
# STEP 5/10: Apply the manifests and wait for the pod to be Ready
# (not just Running - Ready means the readiness probe is passing,
# i.e. the model has actually finished loading).
# ------------------------------------------------------------------
echo ""
echo "--- STEP 5/10: Applying k8s manifests ---"
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml

echo "Waiting for pod to be Ready..."
kubectl wait --for=condition=Ready pod -l app=fastapi --timeout=150s

# ------------------------------------------------------------------
# STEP 6/10: Port-forward so we can hit the API from the host, and
# confirm a real prediction works before we start hammering it.
# ------------------------------------------------------------------
echo ""
echo "--- STEP 6/10: Port-forwarding and sanity-checking prediction ---"
kubectl port-forward svc/fastapi 8000:8000 > /tmp/k8s_stress_portforward.log 2>&1 &
PORT_FORWARD_PID=$!
sleep 3

NORMAL_ROW='{"dur": 0.12147799879312515, "proto": "tcp", "service": "-", "state": "FIN", "spkts": 6, "dpkts": 4, "sbytes": 258, "dbytes": 172, "rate": 74.08748626708984, "sload": 14158.9423828125, "dload": 8495.365234375, "sloss": 0, "dloss": 0, "sinpkt": 24.29560089111328, "dinpkt": 8.375, "sjit": 30.177547454833984, "djit": 11.83060359954834, "swin": 255, "dwin": 255, "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 43, "dmean": 43, "trans_depth": 0, "response_body_len": 0, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1}'

PREDICT_RESPONSE=$(curl -sf -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" -d "$NORMAL_ROW" 2>/dev/null)
if [[ "$PREDICT_RESPONSE" == *"prediction"* ]]; then
  echo "Baseline prediction OK: $PREDICT_RESPONSE"
else
  echo "Baseline prediction FAILED - aborting before load test."
  exit 1
fi

BASELINE_REPLICAS=$(kubectl get deployment fastapi -o jsonpath='{.status.replicas}')
echo "Baseline replica count: $BASELINE_REPLICAS"

# ------------------------------------------------------------------
# STEP 7/10: Generate real concurrent load FROM INSIDE THE CLUSTER,
# not through kubectl port-forward. port-forward proxies everything
# through a single stream via the K8s API server and is genuinely
# low-throughput - a curl loop hitting localhost through that tunnel
# hits the tunnel's ceiling, not the pod's real capacity. Running the
# load generator as its own pod hitting the Service by its in-cluster
# DNS name (fastapi:8000) is both faster and more honest: it's the
# same path real in-cluster traffic would take.
# ------------------------------------------------------------------
echo ""
echo "--- STEP 7/10: Generating load from inside the cluster ($LOAD_WORKERS workers, ${LOAD_DURATION}s) ---"

echo "$NORMAL_ROW" > /tmp/k8s_stress_payload.json

kubectl run load-generator --image=curlimages/curl:latest --restart=Never -- sleep 3600
kubectl wait --for=condition=Ready pod/load-generator --timeout=60s
kubectl cp /tmp/k8s_stress_payload.json load-generator:/tmp/payload.json

kubectl exec load-generator -- sh -c "
  for i in \$(seq 1 $LOAD_WORKERS); do
    ( while true; do
        curl -sf -X POST http://fastapi:8000/v1/predict \
          -H 'Content-Type: application/json' -d @/tmp/payload.json > /dev/null 2>&1
      done ) &
  done
  sleep $LOAD_DURATION
" &
LOAD_EXEC_PID=$!

MAX_REPLICAS_SEEN=1
LOAD_START=$(date +%s)
while [ "$(( $(date +%s) - LOAD_START ))" -lt "$LOAD_DURATION" ]; do
  ELAPSED_LOAD=$(( $(date +%s) - LOAD_START ))
  CURRENT_REPLICAS=$(kubectl get deployment fastapi -o jsonpath='{.status.replicas}' 2>/dev/null || echo "?")
  CURRENT_CPU=$(kubectl get hpa fastapi-hpa -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null || echo "")
  echo "  [${ELAPSED_LOAD}s/${LOAD_DURATION}s] replicas=$CURRENT_REPLICAS  cpu=${CURRENT_CPU}%"
  echo "${ELAPSED_LOAD},load,${CURRENT_REPLICAS},${CURRENT_CPU}" >> "$LOG_FILE"
  if [[ "$CURRENT_REPLICAS" =~ ^[0-9]+$ ]] && [ "$CURRENT_REPLICAS" -gt "$MAX_REPLICAS_SEEN" ]; then
    MAX_REPLICAS_SEEN=$CURRENT_REPLICAS
  fi
  sleep 10
done

echo "Load window done. Waiting for load-generator to finish..."
wait "$LOAD_EXEC_PID" 2>/dev/null || true
kubectl delete pod load-generator --ignore-not-found --now

echo "Highest replica count observed under load: $MAX_REPLICAS_SEEN"

# ------------------------------------------------------------------
# STEP 8/10: Confirm it scales back down. HPA's default scale-down
# stabilization window is 5 minutes, so this genuinely takes a
# while - that's expected, not a bug.
# ------------------------------------------------------------------
echo ""
echo "--- STEP 8/10: Waiting for scale-down back to 1 replica ---"
SCALE_DOWN_ELAPSED=0
FINAL_REPLICAS=$MAX_REPLICAS_SEEN
while [ "$SCALE_DOWN_ELAPSED" -lt "$SCALE_DOWN_TIMEOUT" ]; do
  FINAL_REPLICAS=$(kubectl get deployment fastapi -o jsonpath='{.status.replicas}' 2>/dev/null || echo "?")
  CURRENT_CPU=$(kubectl get hpa fastapi-hpa -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null || echo "")
  echo "  [${SCALE_DOWN_ELAPSED}s/${SCALE_DOWN_TIMEOUT}s] replicas=$FINAL_REPLICAS"
  echo "$(( LOAD_DURATION + SCALE_DOWN_ELAPSED )),scaledown,${FINAL_REPLICAS},${CURRENT_CPU}" >> "$LOG_FILE"
  if [ "$FINAL_REPLICAS" = "1" ]; then
    echo "Back to 1 replica."
    break
  fi
  sleep 15
  SCALE_DOWN_ELAPSED=$((SCALE_DOWN_ELAPSED + 15))
done

# ------------------------------------------------------------------
# STEP 9/10: Check for any pod restarts/failures during the test -
# scaling up "successfully" while pods were crash-looping in the
# background wouldn't actually be a pass.
# ------------------------------------------------------------------
echo ""
echo "--- STEP 9/10: Checking pod health during the test ---"
kubectl get pods -l app=fastapi
RESTART_COUNT=$(kubectl get pods -l app=fastapi -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' | tr ' ' '\n' | awk '{s+=$1} END {print s+0}')
echo "Total restarts across fastapi pods: $RESTART_COUNT"

# ------------------------------------------------------------------
# STEP 10/10: Turn the logged data into a chart for the README -
# real numbers from this run, not a generic illustration.
# ------------------------------------------------------------------
echo ""
echo "--- STEP 10/10: Generating scaling chart ---"
mkdir -p docs/images
if python3 -c "import matplotlib" 2>/dev/null; then
  python3 scripts/plot_hpa_scaling.py "$LOG_FILE" docs/images/hpa-scaling.png
else
  echo "matplotlib not installed - skipping chart (raw data is still in $LOG_FILE)."
  echo "Install it with: pip install matplotlib --break-system-packages"
  echo "Then run: python3 scripts/plot_hpa_scaling.py $LOG_FILE docs/images/hpa-scaling.png"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "RESULT"
echo "=========================================="
echo "Scaled up beyond baseline:  $([ "$MAX_REPLICAS_SEEN" -gt 1 ] && echo "YES (max $MAX_REPLICAS_SEEN)" || echo "NO")"
echo "Scaled back down to 1:      $([ "$FINAL_REPLICAS" = "1" ] && echo "YES" || echo "NO (still at $FINAL_REPLICAS)")"
echo "Pod restarts during test:   $RESTART_COUNT"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Raw log:    $LOG_FILE"
echo ""

if [ "$MAX_REPLICAS_SEEN" -gt 1 ] && [ "$FINAL_REPLICAS" = "1" ] && [ "$RESTART_COUNT" -eq 0 ]; then
  echo "PASS - HPA scaling confirmed working end to end."
  exit 0
else
  echo "FAIL - see numbers above."
  exit 1
fi