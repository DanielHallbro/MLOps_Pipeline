#!/bin/bash
set -e  # stop immediately if any step fails

# Estimated total runtime: ~10-15 minutes (dominated by the Airflow
# DAG run, which trains 4 candidate models sequentially).
START_TIME=$(date +%s)

echo "=========================================="
echo "REBUILDING FULL PIPELINE FROM SCRATCH"
echo "Estimated time: ~10-15 minutes"
echo "=========================================="

echo ""
echo "--- STEP 1/8: Bringing up all Docker services ---"
docker compose up -d --build

echo ""
echo "--- STEP 2/8: Waiting for core services to be healthy ---"
for i in $(seq 1 30); do
  healthy_count=$(docker compose ps postgres mlflow airflow-postgres airflow-webserver | grep -c "healthy")
  if [ "$healthy_count" -eq 4 ]; then
    echo "Core services healthy."
    break
  fi
  echo "Waiting... ($i/30)"
  sleep 3
done

echo ""
echo "--- STEP 3/8: Pulling raw dataset from DVC/S3 ---"
docker compose run --rm training dvc pull \
  data/UNSW_NB15_training-set.parquet.dvc \
  data/UNSW_NB15_testing-set.parquet.dvc \
  --force

echo ""
echo "--- STEP 4/8: Preprocessing raw data ---"
docker compose run --rm training python3 src/training/preprocess.py

echo ""
echo "--- STEP 5/8: Triggering Airflow retraining DAG ---"
docker compose exec -T airflow-webserver airflow dags unpause network_intrusion_retraining

# 2>/dev/null discards Airflow's warning/log noise so only the real
# JSON output gets captured - mixing stderr in here (2>&1) previously
# caused the JSON to arrive interleaved/malformed when captured
# through command substitution.
TRIGGER_OUTPUT=$(docker compose exec -T airflow-webserver airflow dags trigger network_intrusion_retraining -o json 2>/dev/null)
RUN_ID=$(echo "$TRIGGER_OUTPUT" | python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
        if isinstance(data, list):
            print(data[0]['dag_run_id'])
            break
    except (json.JSONDecodeError, IndexError, KeyError):
        continue
")
echo "Triggered run: $RUN_ID"

echo ""
echo "--- STEP 6/8: Waiting for DAG run to complete ---"
for i in $(seq 1 120); do
  LIST_OUTPUT=$(docker compose exec -T airflow-webserver airflow dags list-runs -d network_intrusion_retraining -o json 2>/dev/null)
STATE=$(echo "$LIST_OUTPUT" | python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        runs = json.loads(line)
        if isinstance(runs, list):
            for r in runs:
                if r.get('run_id') == '$RUN_ID':
                    print(r['state'])
                    break
            break
    except (json.JSONDecodeError, IndexError, KeyError):
        continue
")
  echo "  [$i/120] DAG state: $STATE"
  if [ "$STATE" = "success" ]; then
    echo "DAG completed successfully."
    break
  fi
  if [ "$STATE" = "failed" ]; then
    echo "DAG run FAILED. Check Airflow UI for task logs."
    exit 1
  fi
  sleep 15
done

echo ""
echo "--- STEP 7/8: Restarting API to pick up new champion ---"
docker compose restart api

echo ""
echo "--- STEP 8/8: Verifying the full stack responds ---"

check_health() {
  local name=$1
  local url=$2
  echo "$name health:"
  for i in $(seq 1 15); do
    if curl -sf "$url" > /dev/null 2>&1; then
      curl -sf "$url"
      echo ""
      return 0
    fi
    echo "  Waiting for $name... ($i/15)"
    sleep 2
  done
  echo "$name health check FAILED after waiting"
  return 1
}

check_health "API" "http://localhost:8000/health" || exit 1
check_health "MLflow" "http://localhost:5001/health" || exit 1
check_health "Prometheus" "http://localhost:9090/-/healthy" || exit 1
check_health "Grafana" "http://localhost:3000/api/health" || exit 1
check_health "Airflow webserver" "http://localhost:8080/health" || exit 1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "FULL PIPELINE REBUILD COMPLETE"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "=========================================="