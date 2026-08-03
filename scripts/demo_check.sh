#!/bin/bash
# Checks everything's actually up and working before a demo.
# Doesn't rebuild or retrain anything (that's rebuild_pipeline.sh),
# just pings each service and confirms the champion model is loaded
# and can actually return a prediction.

FAILED=0

check_health() {
  local name=$1
  local url=$2
  if curl -sf "$url" > /dev/null 2>&1; then
    echo "✓ $name: OK"
  else
    echo "✗ $name: NOT RESPONDING ($url)"
    FAILED=1
  fi
}

echo "=========================================="
echo "DEMO READINESS CHECK"
echo "=========================================="
echo ""

echo "--- Container status ---"
docker compose ps
echo ""

echo "--- Service health ---"
check_health "API" "http://localhost:8000/health"
check_health "MLflow" "http://localhost:5001/health"
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

echo "=========================================="
if [ "$FAILED" -eq 0 ]; then
  echo "ALL CHECKS PASSED - READY TO PRESENT"
else
  echo "SOME CHECKS FAILED - FIX BEFORE PRESENTING"
fi
echo "=========================================="

exit $FAILED