<a id="readme-top"></a>

# Network Intrusion Detection - MLOps Pipeline

![Type](https://img.shields.io/badge/Type-MLOps_Pipeline-990000)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)
![School](https://img.shields.io/badge/School-Frans_Schartau-blue)
![CI](https://github.com/DanielHallbro/MLOps_Pipeline/actions/workflows/ci.yml/badge.svg)

**Author:** Daniel Hållbro (Student)<br>
**Course:** AI, Automation and Machine Learning<br>
**Assignment:** Advanced MLOps project<br>
**Program:** IT and Cybersecurity, Frans Schartaus Handelsinstitut<br>
**Tools:** DVC · AWS S3 · MLflow · PostgreSQL · FastAPI · Apache Airflow · Prometheus · Grafana · Docker Compose · Kubernetes · GitHub Actions<br>
**Dataset:** UNSW-NB15 - 175,341 training / 82,332 test network connections

---

## Features

- End-to-end MLOps pipeline, from raw data to a served, monitored prediction API
- Scheduled retraining with Apache Airflow
- Experiment tracking and model registry with MLflow
- Data versioning with DVC + AWS S3
- Infrastructure as Code with Terraform
- FastAPI prediction service, loading the current champion model dynamically
- Kubernetes deployment with a Horizontal Pod Autoscaler, scaling the API under real load
- Prometheus metrics + Grafana dashboard
- CI pipeline with an isolated Docker stack and real smoke tests

---

## Table of contents

1. [Features](#features)
2. [Overview](#overview)
3. [Why this dataset](#why-this-dataset)
4. [Results](#results)
5. [Runtime architecture](#runtime-architecture)
6. [Monitoring](#monitoring)
7. [Kubernetes & autoscaling](#kubernetes--autoscaling)
8. [Tech stack](#tech-stack)
9. [Data cleaning decisions](#data-cleaning-decisions)
10. [Project structure](#project-structure)
11. [Running it locally](#running-it-locally)
12. [CI/CD](#cicd)
13. [Security](#security)
14. [Key learnings](#key-learnings)
15. [What's not (yet) included](#whats-not-yet-included)

---

## Overview

![Architecture overview](docs/images/architecture-overview.png)

Data flows from a public network-traffic dataset through cleaning and feature engineering, into two candidate model families (Random Forest and XGBoost), tracked and compared in MLflow. The best-performing model is served through a FastAPI endpoint, with Apache Airflow handling scheduled retraining and Prometheus/Grafana monitoring the live service. GitHub Actions builds and smoke-tests the whole stack on every push.

[⬆ Back to top](#readme-top)

---

## Why this dataset

UNSW-NB15 was chosen over several other candidate intrusion-detection datasets after evaluating class balance, size, and format. It's a modern network intrusion dataset created by the Australian Centre for Cyber Security (ACCS), generated using the IXIA PerfectStorm tool to produce a mix of real normal traffic and synthetic contemporary attack behavior, a more current alternative to older benchmark datasets from the late 1990s/early 2000s.

It also ships with an official, pre-split train/test file pair. The split is intentionally more difficult than a random shuffle, since the test set contains different traffic patterns than training. Model results reported below come from this official split, not an easier random resample, which is why the numbers look lower than some published results elsewhere (see [Key learnings](#key-learnings)).

- 175,341 training rows / 82,332 test rows, parquet format
- 68% attack / 32% normal traffic
- 9 attack categories (Reconnaissance, Exploits, DoS, and others)
- 36 original features per connection record

**Source:** [UNSW-NB15 on Kaggle](https://www.kaggle.com/datasets/dhoogla/unswnb15)

[⬆ Back to top](#readme-top)

---

## Results

Six candidate configurations were trained and compared in total: two untuned baselines, and four tuning attempts across both model families. The table below shows the four still-active candidates. Two earlier Random Forest tuning attempts (an overfitting demonstration and a since-replaced slower version) were retired and are documented in `src/training/legacy/` rather than repeated here.

The untuned Random Forest baseline won every comparison:

| Model | F1 (test set) |
|---|---|
| **Random Forest (baseline)** | **0.9214** ← champion |
| XGBoost (tuned) | 0.9189 |
| Random Forest (tuned v3) | 0.9167 |
| XGBoost (baseline) | 0.9175 |

Every GridSearchCV tuning attempt, across all six configurations tested, not just the four above, lost to the simple, untuned baseline. A real, measured result rather than an assumption that more tuning always helps. Random Forest's default settings already generalized well to the harder, official test split, XGBoost is often expected to edge it out, but not here, and not after real tuning attempts on both.

[⬆ Back to top](#readme-top)

---

## Runtime architecture

<img src="docs/images/architecture-technical.png" width="650">

The overview diagram (above) highlights the model-selection workflow, comparing Random Forest and XGBoost candidates. This diagram focuses on what's actually deployed and running, which is why it shows one training container rather than the individual model families, the model comparison already happened by the time this stage of the pipeline runs.

Terraform doesn't appear in this diagram on purpose: it provisions the S3 bucket and IAM user once, ahead of time, and isn't part of the running system shown here.

The diagram itself traces the actual flow, arrows and labels show exactly what triggers what. It reads top to bottom, starting with Airflow, which is what actually kicks off a training cycle. GitHub Actions sits on its own arrow into FastAPI since it runs independently on every push, not as a step inside the scheduled training cycle above it.

Color groups reflect subsystems: purple for the Airflow stack, blue for MLflow's, teal for the serving-and-monitoring trio, green for external services, tan for one-time setup containers. Airflow's Postgres is deliberately separate from MLflow's, since the two have different access patterns (Airflow's scheduler polls constantly), matching Apache's own reference architecture rather than sharing one database across both.

Two containers (`artifacts-init`, `airflow-init`) run once at startup, fixing shared-volume permissions and creating Airflow's database schema and admin user, then exit. They're not part of the ongoing request flow, which is why they're shown separately rather than inline with the persistent containers.

[⬆ Back to top](#readme-top)

---

## Monitoring

![Grafana dashboard](docs/images/grafana-dashboard.png)

Three panels, each with a hover description explaining what it shows and why: request rate (is the API actually being used, and is that changing), predictions by outcome (a sudden shift in the attack/normal ratio is worth investigating), and API response time, worst-case (how slow the slowest normal requests get, not just the average, since an average can hide a real slowdown).

[⬆ Back to top](#readme-top)

---

## Kubernetes & autoscaling

The rest of the stack (Postgres, MLflow, Airflow, Prometheus, Grafana) still runs via Docker Compose exactly as described above. Only FastAPI, the one component that actually benefits from scaling under request load, runs in Kubernetes on a local `minikube` cluster, with a Horizontal Pod Autoscaler (HPA) watching it. A deliberate scope decision: the stateful services don't gain anything from horizontal scaling, so a full-stack migration is left as a future direction rather than something this addition needed to solve.

| Piece | File | Role |
|---|---|---|
| Deployment | `k8s/deployment.yaml` | Runs the FastAPI container as pods; each pod requests 100m CPU / 256Mi memory, capped at 500m CPU / 512Mi |
| Service | `k8s/service.yaml` | Stable internal address (`fastapi:8000`), load-balances across whichever pods currently exist |
| HorizontalPodAutoscaler | `k8s/hpa.yaml` | Watches average CPU utilization across pods, scales between 1 and 4 replicas, targeting 60% utilization |

![HPA scaling FastAPI under load](docs/images/hpa-scaling.png)

Replica count and CPU utilization from a real run of `scripts/k8s_stress_test.sh`: a cold-start script that brings up Docker Compose, minikube, and the manifests from nothing, drives real concurrent load at the API, and confirms scaling. This run jumped from 1 to 4 replicas within about 90 seconds of load starting (CPU peaked at ~488% of the 100m request, i.e. genuinely saturating each pod's 500m limit), held at 4 for HPA's default 5-minute scale-down stabilization window after load stopped, then dropped back to 1. See [Key learnings](#key-learnings) for the one real wrinkle running FastAPI outside Compose caused.

[⬆ Back to top](#readme-top)

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Data versioning | DVC + AWS S3 | Dataset never touches git, versioned like code instead |
| Infrastructure as Code | Terraform | Provisions the S3 bucket and IAM user as code, rather than clicking them together manually in the AWS console |
| Model training | scikit-learn (Random Forest), XGBoost | Two model families compared honestly, not assumed |
| Experiment tracking | MLflow + Postgres | Postgres over SQLite for real concurrent access, since the API, Airflow, and CI all touch it |
| Serving | FastAPI | Loads the current champion model dynamically via MLflow's registry, no hardcoded model path |
| Orchestration | Apache Airflow (LocalExecutor) | Schedules retraining, with a Slack alert on task failure |
| Monitoring | Prometheus + Grafana | Custom metrics: request rate, prediction outcomes, p95 latency |
| Containerization | Docker Compose | Every long-running service above is defined and orchestrated from one compose file |
| Autoscaling | Kubernetes (minikube) + HPA | Scales the FastAPI service between 1 and 4 pods based on CPU load, scoped to the one stateless service that benefits from it |
| CI/CD | GitHub Actions | Builds an isolated copy of the Docker stack on GitHub's own infrastructure, trains a dummy model, and smoke-tests the real API code path |

[⬆ Back to top](#readme-top)

---

## Data cleaning decisions

- `attack_cat` dropped, since it directly encodes the label (target leakage) and would never be available for a real, unseen connection
- `service='-'` (54% of rows) kept as its own `unknown` category, not dropped or treated as missing, since it's a legitimate value (e.g. ICMP/ARP have no application-layer service)
- `proto` (133 unique values, long-tail distribution): anything under 0.5% of training rows grouped into `other`, with the threshold computed dynamically from the data rather than hardcoded
- Two engineered ratio features (`sbytes/dbytes`, `spkts/dpkts`), measured to genuinely improve F1 by +0.003, and one ended up as the single most important feature in the final model

[⬆ Back to top](#readme-top)

---

## Project structure

The repository is organized by responsibility rather than by framework, training code, serving code, and orchestration each live in their own top-level folder.

```
├── src/training/                # preprocessing, training scripts, MLflow promotion
│   ├── legacy/                  # retired experiments (kept for reference)
│   └── experiments/             # standalone analyses (feature importance, ratio-feature impact)
├── api/                          # FastAPI service
├── airflow/dags/                 # retraining DAG
├── k8s/                           # Deployment, Service, HPA manifests for FastAPI
├── infra/                        # Terraform, provisions the S3 bucket + IAM user
├── data/                         # DVC pointer files (actual dataset pulled via dvc pull)
├── notebooks/                    # exploratory data analysis scripts
├── prometheus/                   # scrape config
├── grafana/dashboards/           # exported dashboard JSON
├── scripts/                      # rebuild_pipeline.sh, demo_check.sh, simulate_traffic.sh, k8s_stress_test.sh
├── tests/                        # CI dummy-model smoke test
├── docs/images/                  # architecture diagrams, dashboard screenshot
├── .github/workflows/ci.yml      # GitHub Actions workflow: lint, build, smoke test
├── docker-compose.yml             # the real stack: all long-running services
├── docker-compose.ci.yml         # isolated stack used by CI
├── Dockerfile.api                 # builds the FastAPI service image
├── Dockerfile.mlflow              # builds the MLflow tracking server image
├── Dockerfile.airflow             # builds the Airflow webserver/scheduler image
├── Dockerfile.training            # builds the training container image (also used by Airflow tasks)
└── .env.example                   # template for real credentials, no real secrets
```

[⬆ Back to top](#readme-top)

---

## Running it locally

Requires Docker and Docker Compose.

**1. Create your own `.env`.** Copy `.env.example` to `.env` and fill in real values yourself (AWS access keys, Postgres passwords, Airflow's Fernet key and admin credentials, Slack webhook, Grafana admin). There's no automated setup script for this yet (see [What's not (yet) included](#whats-not-yet-included)), so every credential needs to be generated and entered manually before anything below will actually run.

**2. Bring the data and a trained champion model up from nothing** (also brings up the Docker Compose stack itself; requires valid AWS credentials in `.env` to pull the dataset from S3):

```bash
./scripts/rebuild_pipeline.sh
```

**3. Bring up Kubernetes and check everything's healthy before a demo** (also brings up Docker Compose if it isn't already running):

```bash
./scripts/demo_check.sh
```

**4. Generate some live traffic to see the Grafana dashboard move:**

```bash
./scripts/simulate_traffic.sh
```

**5. Try the Kubernetes autoscaling** (requires `minikube` and `kubectl`, brings up the FastAPI Deployment/Service/HPA on top of the running stack and load-tests it end to end):

```bash
./scripts/k8s_stress_test.sh
```

**6. Access the services:**
- **FastAPI:** `http://localhost:8000/docs`
- **MLflow:** `http://localhost:5001`
- **Airflow:** `http://localhost:8080`
- **Prometheus:** `http://localhost:9090`
- **Grafana:** `http://localhost:3000`

[⬆ Back to top](#readme-top)

---

## CI/CD

Every push to `main` triggers a GitHub Actions workflow that spins up a second, fully isolated copy of the core stack (`docker-compose.ci.yml`), separate from the real development stack and using throwaway credentials, so nothing in CI ever touches real data, real secrets, or the real MLflow history.

Inside that isolated stack, the workflow:
1. Builds every Docker image fresh, to catch broken Dockerfiles or dependency issues early
2. Trains a small, fast dummy model (on synthetic data, not the real dataset) and registers it in the isolated MLflow instance as the champion
3. Starts the real API container and confirms it loads that dummy champion correctly via MLflow's registry, the exact same code path used in production
4. Sends a real prediction request and checks the response is well-formed
5. Sends a request with an unseen category value and confirms it's handled gracefully rather than crashing
6. Sends a malformed request and confirms it's rejected with a proper 422, not a silent failure

The whole isolated stack is torn down at the end of the run regardless of outcome, so nothing lingers between workflow runs.

[⬆ Back to top](#readme-top)

---

## Security

The following are never committed to this repo:
- AWS access keys and secret keys (DVC/S3 access), supplied via `.env`, never hardcoded
- Postgres, Airflow, and Grafana credentials, supplied via `.env`
- Slack webhook URL, supplied via `.env`
- The raw dataset itself, versioned via DVC/S3, not stored in git

`.env.example` documents every variable a real `.env` needs, with placeholder values only. `.gitignore` excludes `.env`, local DVC credentials, and MLflow's local SQLite database.

[⬆ Back to top](#readme-top)

---

## Key learnings

### Data and modeling

**The official train/test split matters more than the F1 number alone.** Published results on this dataset vary wildly (77%-99%) depending on whether the evaluation used the official, harder split or a random resample. A hold-out test on the real split gave a comparable result (~91%) to this project's own numbers, evidence that a lower-looking score can reflect a more honest evaluation, not a worse model.

**Tuning doesn't always help, and measuring that beats assuming it.** Every GridSearchCV attempt (across both model families, multiple grid sizes) lost to the untuned Random Forest baseline. One tuning run even reproduced a textbook overfitting case: highest cross-validation score of all six candidates, worst test-set score of all six, a direct demonstration of why held-out evaluation matters more than CV score alone.

### Infrastructure and operations

**Running everything in parallel isn't always the right call.** An early attempt to run all four training candidates simultaneously through Airflow oversubscribed a 4-core development machine, since each script also internally parallelizes via GridSearchCV. Capping Airflow to sequential execution traded some wall-clock time for a stable, predictable pipeline, the right tradeoff for the available hardware.

**A model registry needs a real backend once more than one thing touches it.** MLflow ran on SQLite during early development, then moved to Postgres once the API, Airflow, and CI all needed to read from it. SQLite's single-writer limitation becomes a real constraint the moment more than one process is involved, not just a theoretical concern.

**A `docker compose` command operates on every service in scope, not just the one you're thinking about.** Tearing down an isolated CI test stack with `docker compose -f docker-compose.yml -f docker-compose.ci.yml down -v` also wiped the real development stack's volumes, MLflow's champion history and Airflow's metadata included, since merging both compose files puts every service from both files in scope for that command, not just the CI-specific ones. Recovered via `rebuild_pipeline.sh`, but the real lesson was to be explicit about exactly which services a teardown targets, not just which files get passed to the command.

**Kubernetes doesn't share anything with Docker Compose by default.** Pulling FastAPI out into a separate minikube cluster broke its normal path to MLflow: a pod can't resolve a Compose service name like `mlflow`, and can't mount a Compose volume either, both only exist inside the Compose network. Fixed by having MLflow also serve its artifacts over plain HTTP instead of relying solely on a shared volume, and pointing the pod at `host.minikube.internal`, minikube's built-in route back to the host machine. A concrete example of how adding a new orchestration layer can quietly invalidate assumptions the rest of the stack didn't know it was relying on.

[⬆ Back to top](#readme-top)

---

## What's not (yet) included

- **Full Kubernetes migration**: Postgres, MLflow, Airflow, Prometheus, and Grafana still run via Docker Compose; only FastAPI runs in Kubernetes so far (see [Kubernetes & autoscaling](#kubernetes--autoscaling))
- **A guided first-time setup script** (prompting for credentials to generate `.env`): planned, not yet built

[⬆ Back to top](#readme-top)

---

*Repo: [MLOps_Pipeline](https://github.com/DanielHallbro/MLOps_Pipeline)*