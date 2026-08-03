"""
app.py - FastAPI service that serves live predictions from the
current MLflow "champion" model.

The whole point of this file is that it contains NO assumptions about
which model is currently best. It always asks MLflow for whichever
version has the "champion" alias, and pulls that run's own scaler,
encoder, and protocol-grouping list alongside it. If promote_champion.py
is run again later (e.g. after Airflow triggers a retrain and a new
model wins), this file never needs to change or be redeployed - it
will pick up the new champion automatically the next time it starts.

Endpoint is versioned (/v1/predict) per the project plan, so a future
breaking change to the input schema could be introduced as /v2/predict
without breaking whatever is already calling /v1.
"""

import json
import os
import sys
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

import mlflow
from mlflow import MlflowClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "training"))
from common import get_tracking_uri

MODEL_NAME = "network-intrusion-detector"

app = FastAPI(title="Network Intrusion Detection API")

# ---------------------------------------------------------------
# STEP 1: Load the champion model and its matching preprocessing
# artifacts once, when the API starts (not on every request - that
# would be slow and pointless, since the champion doesn't change
# mid-run).
# ---------------------------------------------------------------
print("STEP 1: Connecting to MLflow and loading champion model...")
mlflow.set_tracking_uri(get_tracking_uri())
client = MlflowClient()

# Resolve the alias "champion" to an actual run, so we know which
# run's artifacts (scaler/encoder/protocol list) belong with this
# specific model version - these must always come from the SAME run,
# never mixed and matched between versions.
champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
run_id = champion_version.run_id
print(f"  Champion is version {champion_version.version} (run_id: {run_id})")

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
print("  Model loaded.")

# Download the three preprocessing artifacts logged alongside this
# specific run, then load them into memory.
artifact_dir = mlflow.artifacts.download_artifacts(
    run_id=run_id, artifact_path="preprocessors"
)
scaler = joblib.load(f"{artifact_dir}/scaler.pkl")
encoder = joblib.load(f"{artifact_dir}/encoder.pkl")
with open(f"{artifact_dir}/common_protocols.json") as f:
    common_protocols = set(json.load(f))
print(f"  Preprocessing artifacts loaded (scaler, encoder, {len(common_protocols)} common protocols).\n")

# Columns the encoder was fit on and the full list of numeric columns
# the scaler was fit on - needed so we build the row in exactly the
# same shape the model expects, every time.
CATEGORICAL_COLS = ["proto", "service", "state"]
NUMERIC_COLS = scaler.feature_names_in_.tolist()

# ---------------------------------------------------------------
# Prometheus metrics - three basic, meaningful signals for an ML API:
# how many requests, how fast, and what the model is actually
# predicting (useful for spotting drift later, e.g. if the attack
# ratio suddenly shifts).
# ---------------------------------------------------------------
REQUEST_COUNT = Counter(
    "predict_requests_total", "Total number of prediction requests"
)
REQUEST_LATENCY = Histogram(
    "predict_request_duration_seconds", "Time spent processing a prediction request"
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions by outcome", ["label"]
)


# ---------------------------------------------------------------
# STEP 2: Define what a raw request looks like.
# These are the fields a real captured network flow would have,
# BEFORE any of preprocess.py's cleaning steps run. Only the columns
# that actually survive to become model features are listed - the 6
# columns preprocess.py drops (stcpb, dtcpb, is_ftp_login, etc.)
# aren't asked for at all, since the model never uses them anyway.
# ---------------------------------------------------------------
class ConnectionFeatures(BaseModel):
    dur: float
    proto: str
    service: str
    state: str
    spkts: int
    dpkts: int
    sbytes: int
    dbytes: int
    rate: float
    sload: float
    dload: float
    sloss: int
    dloss: int
    sinpkt: float
    dinpkt: float
    sjit: float
    djit: float
    swin: int
    dwin: int
    tcprtt: float
    synack: float
    ackdat: float
    smean: int
    dmean: int
    trans_depth: int
    response_body_len: int
    ct_src_dport_ltm: int
    ct_dst_sport_ltm: int


# ---------------------------------------------------------------
# STEP 3: Preprocessing function - mirrors preprocess.py's steps
# 3 through 7, but applied to a single incoming row using the
# already-fitted scaler/encoder/protocol list, instead of fitting new
# ones. This is the exact reason those objects were saved in the
# first place.
# ---------------------------------------------------------------
def preprocess_request(features: ConnectionFeatures) -> pd.DataFrame:
    df = pd.DataFrame([features.model_dump()])

    # Same as preprocess.py STEP 3: '-' is a real category, not
    # missing data.
    df["service"] = df["service"].astype(str).replace("-", "unknown")

    # Same as preprocess.py STEP 4, but using the protocol list learned
    # at training time instead of recalculating a threshold here.
    df["proto"] = df["proto"].astype(str).apply(
        lambda p: p if p in common_protocols else "other"
    )

    # Same as preprocess.py STEP 5.
    df["sbytes_dbytes_ratio"] = df["sbytes"] / (df["dbytes"] + 1)
    df["spkts_dpkts_ratio"] = df["spkts"] / (df["dpkts"] + 1)

    # Same as preprocess.py STEP 6, but using .transform() instead of
    # .fit_transform() - this is exactly what makes OneHotEncoder
    # reusable where pd.get_dummies() wasn't.
    encoded = encoder.transform(df[CATEGORICAL_COLS])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(CATEGORICAL_COLS), index=df.index
    )
    df = pd.concat([df.drop(columns=CATEGORICAL_COLS), encoded_df], axis=1)

    # Same as preprocess.py STEP 7, again using .transform() instead
    # of .fit_transform().
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Make sure column order matches exactly what the model was
    # trained on - scikit-learn/XGBoost models care about column
    # order, not just column names.
    return df[NUMERIC_COLS]


# ---------------------------------------------------------------
# STEP 4: Endpoints
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "champion_version": champion_version.version}


@app.post("/v1/predict")
def predict(features: ConnectionFeatures):
    start_time = time.time()
    REQUEST_COUNT.inc()

    processed = preprocess_request(features)
    prediction = model.predict(processed)[0]
    label = "attack" if prediction == 1 else "normal"

    PREDICTION_COUNT.labels(label=label).inc()
    REQUEST_LATENCY.observe(time.time() - start_time)

    return {
        "prediction": int(prediction),
        "label": label,
        "model_version": champion_version.version,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)