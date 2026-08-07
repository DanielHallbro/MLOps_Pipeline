#!/bin/bash
# Checks that the tools this repo actually needs on the HOST machine
# are installed, before you get partway through Running it locally
# and hit a missing command. Doesn't check dvc - that's only ever
# invoked inside containers (see Dockerfile.training/.airflow), never
# on the host directly. Doesn't check the aws CLI either - nothing in
# this repo calls it; Terraform and boto3/dvc[s3] talk to AWS
# directly via credentials, not through the CLI tool.

FAILED=0

check_tool() {
  local name=$1
  local check_cmd=$2
  local install_hint=$3
  if eval "$check_cmd" > /dev/null 2>&1; then
    echo "✓ $name: found"
  else
    echo "✗ $name: NOT FOUND - $install_hint"
    FAILED=1
  fi
}

echo "=========================================="
echo "PREREQUISITE CHECK"
echo "=========================================="
echo ""

echo "--- Always required ---"
check_tool "Docker" "command -v docker" \
  "install from https://docs.docker.com/engine/install/"
check_tool "Docker Compose" "docker compose version" \
  "included with modern Docker installs - if missing, see https://docs.docker.com/compose/install/"

echo ""
echo "--- Required for Step 1 (only if you don't already have AWS credentials for this project's S3 bucket) ---"
check_tool "Terraform" "command -v terraform" \
  "install from https://developer.hashicorp.com/terraform/install"

echo ""
echo "--- Required for Steps 3 and 5 (Kubernetes/HPA) ---"
check_tool "minikube" "command -v minikube" \
  "install from https://minikube.sigs.k8s.io/docs/start/"
check_tool "kubectl" "command -v kubectl" \
  "install from https://kubernetes.io/docs/tasks/tools/#kubectl"

echo ""
echo "=========================================="
if [ "$FAILED" -eq 0 ]; then
  echo "ALL PREREQUISITES FOUND"
else
  echo "SOME PREREQUISITES MISSING - see above"
fi
echo "=========================================="
exit $FAILED