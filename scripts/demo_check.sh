#!/bin/bash
# Brings everything up (Docker Compose + Kubernetes) and checks it's
# all actually working before a demo. Doesn't retrain anything (that's
# rebuild_pipeline.sh) - assumes a champion model already exists, just
# makes sure every service, including the K8s/HPA piece, is live and
# can actually return a prediction.

FAILED=0
API_IMAGE="mlops_pipeline-api:latest"

check_health() {
  local name=$1
  local url=$2
  local retries=${3:-20}
  for i in $(seq 1 "$retries"); do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "✓ $name: OK"
      return
    fi
    sleep 3
  done
  echo "✗ $name: NOT RESPONDING after $((retries * 3))s ($url)"
  FAILED=1
}

echo "=========================================="
echo "DEMO READINESS CHECK"
echo "=========================================="

echo ""
echo "--- Starting minikube (before Compose - MLflow's port binds to"
echo "    the minikube docker-bridge gateway IP, which only exists"
echo "    once minikube's own docker network has been created) ---"
minikube start
minikube addons enable metrics-server > /dev/null

echo ""
echo "--- Bringing up Docker Compose stack ---"
docker compose up -d --build

echo "--- Container status ---"
docker compose ps
echo ""

echo "--- Service health ---"
check_health "API" "http://localhost:8000/health"
# MLflow's health check uses 192.168.49.1 (minikube's docker-bridge
# gateway), not localhost - matches how the port is bound in
# docker-compose.yml. If that IP ever changes (see the comment there),
# this needs updating too.
check_health "MLflow" "http://192.168.49.1:5001/health"
check_health "Prometheus" "http://localhost:9090/-/healthy"
check_health "Grafana" "http://localhost:3000/api/health"
check_health "Airflow webserver" "http://localhost:8080/health"
echo ""

echo "--- Champion model check ---"
CHAMPION_INFO=$(docker compose run --rm training python3 -c "
import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri('http://mlflow:5000')
client = MlflowClient()
try:
    champion = client.get_model_version_by_alias('network-intrusion-detector', 'champion')
    run = client.get_run(champion.run_id)
    f1 = run.data.metrics.get('f1_score')
    print(f'OK: version {champion.version}, F1={f1}')
except Exception as e:
    print(f'MISSING: {e}')
" 2>/dev/null | tail -1)
echo "$CHAMPION_INFO"
if [[ "$CHAMPION_INFO" == MISSING* ]]; then
  echo "✗ No champion set - API predictions will fail!"
  FAILED=1
else
  echo "✓ Champion model: $CHAMPION_INFO"
fi
echo ""

echo "--- Live prediction test ---"
PREDICT_RESPONSE=$(curl -sf -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dur": 0.5, "proto": "tcp", "service": "http", "state": "FIN",
    "spkts": 10, "dpkts": 8, "sbytes": 500, "dbytes": 4000,
    "rate": 20.0, "sload": 8000.0, "dload": 64000.0,
    "sloss": 0, "dloss": 0, "sinpkt": 50.0, "dinpkt": 60.0,
    "sjit": 1.0, "djit": 1.5, "swin": 255, "dwin": 255,
    "tcprtt": 0.1, "synack": 0.05, "ackdat": 0.05,
    "smean": 50, "dmean": 500, "trans_depth": 1,
    "response_body_len": 3000, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1
  }' 2>/dev/null)
if [[ "$PREDICT_RESPONSE" == *"prediction"* ]]; then
  echo "✓ Prediction endpoint working: $PREDICT_RESPONSE"
else
  echo "✗ Prediction endpoint FAILED"
  FAILED=1
fi
echo ""

echo "--- Airflow DAG status ---"
docker compose exec -T airflow-webserver airflow dags list 2>/dev/null | grep network_intrusion_retraining
echo ""

echo "--- Bringing up Kubernetes manifests (minikube already started above) ---"
eval "$(minikube docker-env)"
docker build -t "$API_IMAGE" -f Dockerfile.api . > /dev/null
eval "$(minikube docker-env -u)"

kubectl delete -f k8s/ --ignore-not-found
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
echo "Waiting for fastapi pod to be Ready..."
if kubectl wait --for=condition=Ready pod -l app=fastapi --timeout=150s > /dev/null 2>&1; then
  echo "✓ fastapi pod: Ready"
else
  echo "✗ fastapi pod: NOT READY - check 'kubectl describe pod -l app=fastapi'"
  FAILED=1
fi
echo ""

echo "--- Kubernetes health ---"
kubectl get pods -l app=fastapi
echo "Waiting for metrics-server to report real numbers..."
METRICS_OK=0
for i in $(seq 1 30); do
  if kubectl top nodes > /dev/null 2>&1; then
    METRICS_OK=1
    break
  fi
  sleep 5
done
if [ "$METRICS_OK" -eq 1 ]; then
  echo "✓ metrics-server: reporting (HPA can see CPU)"
else
  echo "✗ metrics-server: NOT reporting after 150s - HPA won't scale until this works"
  FAILED=1
fi

HPA_STATUS=$(kubectl get hpa fastapi-hpa -o jsonpath='{.status.currentReplicas}' 2>/dev/null)
if [ -n "$HPA_STATUS" ]; then
  echo "✓ HPA: active ($HPA_STATUS replica(s) currently)"
else
  echo "✗ HPA: not found - check 'kubectl apply -f k8s/hpa.yaml'"
  FAILED=1
fi

# HPA can transiently scale up during a pod's cold start if
# metrics-server briefly can't read CPU from an unready pod (a known
# HPA behavior, not a real load event - verified via 'kubectl describe
# hpa fastapi-hpa', which will show FailedGetResourceMetric events
# around the same time). Give it up to the 5-minute stabilization
# window to settle back to 1 before actually failing this check.
if [ "$HPA_STATUS" != "1" ]; then
  echo "  Replica count is $HPA_STATUS, not 1 yet - waiting up to 6 min for it to settle"
  echo "  (this can happen if metrics-server briefly missed the pod during its cold start)"
  SETTLE_OK=0
  for i in $(seq 1 24); do
    sleep 15
    HPA_STATUS=$(kubectl get hpa fastapi-hpa -o jsonpath='{.status.currentReplicas}' 2>/dev/null)
    echo "  [$((i * 15))s] replicas=$HPA_STATUS"
    if [ "$HPA_STATUS" = "1" ]; then
      SETTLE_OK=1
      break
    fi
  done
  if [ "$SETTLE_OK" -eq 1 ]; then
    echo "✓ Settled at 1 replica."
  else
    echo "✗ Still at $HPA_STATUS replicas after 6 min - this is NOT the metrics-gap"
    echo "  behavior, something real is keeping CPU up. Investigate before presenting:"
    echo "  kubectl describe hpa fastapi-hpa"
    echo "  kubectl top pods -l app=fastapi"
    FAILED=1
  fi
fi
echo ""

echo "--- Live prediction test (via Kubernetes, not Compose) ---"
kubectl port-forward svc/fastapi 8001:8000 > /tmp/demo_check_portforward.log 2>&1 &
PF_PID=$!
sleep 3
K8S_PREDICT_RESPONSE=$(curl -sf -X POST http://localhost:8001/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dur": 0.5, "proto": "tcp", "service": "http", "state": "FIN",
    "spkts": 10, "dpkts": 8, "sbytes": 500, "dbytes": 4000,
    "rate": 20.0, "sload": 8000.0, "dload": 64000.0,
    "sloss": 0, "dloss": 0, "sinpkt": 50.0, "dinpkt": 60.0,
    "sjit": 1.0, "djit": 1.5, "swin": 255, "dwin": 255,
    "tcprtt": 0.1, "synack": 0.05, "ackdat": 0.05,
    "smean": 50, "dmean": 500, "trans_depth": 1,
    "response_body_len": 3000, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1
  }' 2>/dev/null)
kill "$PF_PID" 2>/dev/null
if [[ "$K8S_PREDICT_RESPONSE" == *"prediction"* ]]; then
  echo "✓ K8s prediction endpoint working: $K8S_PREDICT_RESPONSE"
else
  echo "✗ K8s prediction endpoint FAILED"
  FAILED=1
fi
echo ""

echo "=========================================="
if [ "$FAILED" -eq 0 ]; then
  echo "ALL CHECKS PASSED - READY TO PRESENT"
else
  echo "SOME CHECKS FAILED - FIX BEFORE PRESENTING"
fi
echo "=========================================="

exit $FAILED