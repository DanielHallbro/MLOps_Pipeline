"""
common.py - Shared helpers used by every training script, so the
pattern for logging preprocessing artifacts stays identical no matter
which model wins and gets promoted to champion later.
"""

import os

import mlflow


def get_tracking_uri():
    """
    Returns where MLflow should log to. Reads from the
    MLFLOW_TRACKING_URI environment variable if it's set (this is how
    Docker Compose will point every script at Postgres, without
    changing a single line of code), and falls back to the local
    SQLite file for plain, non-Docker runs on your VM, exactly what
    every script has been using so far.
    """
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db")

def log_preprocessing_artifacts():
    """
    Logs the fitted scaler and encoder (saved by preprocess.py) as
    MLflow artifacts inside the currently active run. Must be called
    while a run is open (inside a `with mlflow.start_run():` block),
    right before or after logging the model itself.

    This is what makes a run "complete": without this, a run has a
    model but no way to correctly preprocess a new, raw request
    before feeding it to that model.
    """
    mlflow.log_artifact("models/preprocessing/scaler.pkl", artifact_path="preprocessors")
    mlflow.log_artifact("models/preprocessing/encoder.pkl", artifact_path="preprocessors")
    mlflow.log_artifact("models/preprocessing/common_protocols.json", artifact_path="preprocessors")