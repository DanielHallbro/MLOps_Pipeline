"""
create_dummy_model.py - Creates and registers a tiny, fake champion
model for CI smoke testing.

This is NOT a real model - it's trained on a handful of random rows
just so the FastAPI service has something to load via MLflow's
registry (models:/network-intrusion-detector@champion), the exact
same loading path api/main.py uses in production. That lets CI
exercise the real API code (schema validation, preprocessing,
MLflow loading) without needing the real ~175K-row dataset or a real
multi-minute training run.

Run from the repo root, same as the real training scripts:
    python3 tests/create_dummy_model.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "training"))

import json

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
from common import get_tracking_uri, log_preprocessing_artifacts
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow

MODEL_NAME = "network-intrusion-detector"

# ---------------------------------------------------------------
# STEP 1: Build tiny synthetic data with the same shape/categories
# as the real pipeline expects - not real network traffic, just
# enough structure for the preprocessing and model code to run.
# ---------------------------------------------------------------
print("STEP 1: Generating synthetic data...")
np.random.seed(42)
N_ROWS = 50

NUMERIC_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "sbytes_dbytes_ratio", "spkts_dpkts_ratio",
]
# Same categories the real pipeline settles on: 6 common protocols +
# "other", the full set of observed services, and connection states.
PROTO_VALUES = ["tcp", "udp", "unas", "arp", "ospf", "sctp", "other"]
SERVICE_VALUES = ["http", "dns", "smtp", "ftp", "ftp-data", "ssh", "pop3",
                   "dhcp", "snmp", "ssl", "irc", "radius", "unknown"]
STATE_VALUES = ["FIN", "INT", "CON", "ECO", "REQ", "RST", "PAR", "URN", "no"]

data = pd.DataFrame(np.random.rand(N_ROWS, len(NUMERIC_COLS)), columns=NUMERIC_COLS)
data["proto"] = np.random.choice(PROTO_VALUES, N_ROWS)
data["service"] = np.random.choice(SERVICE_VALUES, N_ROWS)
data["state"] = np.random.choice(STATE_VALUES, N_ROWS)
data["label"] = np.random.randint(0, 2, N_ROWS)
print(f"  Generated {N_ROWS} synthetic rows.\n")

# ---------------------------------------------------------------
# STEP 2: Fit tiny preprocessing objects (same kind the real
# preprocess.py produces) so the API's loading code has real
# scaler/encoder artifacts to work with, not placeholders.
# ---------------------------------------------------------------
print("STEP 2: Fitting scaler and encoder...")
CATEGORICAL_COLS = ["proto", "service", "state"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(data[CATEGORICAL_COLS])
encoded_df = pd.DataFrame(
    encoded, columns=encoder.get_feature_names_out(CATEGORICAL_COLS), index=data.index
)
full_data = pd.concat([data.drop(columns=CATEGORICAL_COLS), encoded_df], axis=1)

scaler = StandardScaler()
scale_cols = [c for c in full_data.columns if c != "label"]
full_data[scale_cols] = scaler.fit_transform(full_data[scale_cols])
print("  Done.\n")

# ---------------------------------------------------------------
# STEP 3: Save preprocessing artifacts to the same local path the
# real pipeline uses, so log_preprocessing_artifacts() (shared with
# the real training scripts) can pick them up unchanged.
# ---------------------------------------------------------------
print("STEP 3: Saving preprocessing artifacts...")
os.makedirs("models/preprocessing", exist_ok=True)
joblib.dump(scaler, "models/preprocessing/scaler.pkl")
joblib.dump(encoder, "models/preprocessing/encoder.pkl")
with open("models/preprocessing/common_protocols.json", "w") as f:
    json.dump(PROTO_VALUES[:-1], f)  # exclude "other" itself from the "kept" list
print("  Saved.\n")

# ---------------------------------------------------------------
# STEP 4: Train a tiny, fast model - not meant to be accurate, only
# meant to exist and respond to predict() calls correctly.
# ---------------------------------------------------------------
print("STEP 4: Training dummy model...")
X = full_data[scale_cols]
y = data["label"]
model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
model.fit(X, y)
print("  Done.\n")

# ---------------------------------------------------------------
# STEP 5: Log to MLflow and set as champion, exactly like
# promote_champion.py does for a real run, so the API's
# models:/network-intrusion-detector@champion lookup succeeds.
# ---------------------------------------------------------------
print("STEP 5: Registering dummy model as champion...")
mlflow.set_tracking_uri(get_tracking_uri())
mlflow.set_experiment("network-intrusion-detection-ci")

with mlflow.start_run(run_name="ci_dummy_model"):
    mlflow.log_param("model_type", "CI_DUMMY")
    mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME)
    log_preprocessing_artifacts()

from mlflow import MlflowClient

client = MlflowClient()
latest_version = max(
    client.search_model_versions(f"name='{MODEL_NAME}'"),
    key=lambda v: int(v.version),
)
client.set_registered_model_alias(MODEL_NAME, "champion", latest_version.version)
print(f"  Champion set to version {latest_version.version}.\n")

print("Dummy model ready. api/main.py can now load it via the champion alias.")