"""
promote_champion.py - Compares all registered versions of
network-intrusion-detector and marks the best one as "champion".

Why this exists: FastAPI should never hardcode which model to load
(e.g. "the Random Forest one"). Instead it always loads whichever
version currently has the "champion" alias. That way, if a future
retrain (triggered manually or by Airflow) produces a better model,
promoting it here is the only change needed - the API code itself
never has to be touched or redeployed.

Rule used to pick the winner: highest F1 score on the test set, full
stop. No extra tie-breaking logic (latency, model size, etc.) - that
would be over-engineering for a solo project at this stage.
"""

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "network-intrusion-detector"

# ---------------------------------------------------------------
# STEP 1: Connect to MLflow
# ---------------------------------------------------------------
print("STEP 1: Connecting to MLflow...")
from common import get_tracking_uri
mlflow.set_tracking_uri(get_tracking_uri())
client = MlflowClient()
print("  Connected.\n")

# ---------------------------------------------------------------
# STEP 2: Get every registered version of our model
# ---------------------------------------------------------------
print(f"STEP 2: Fetching all registered versions of '{MODEL_NAME}'...")
# Each time a training script calls log_model(..., registered_model_name=...),
# it creates one more numbered version here (v1, v2, v3...). This pulls
# the full list so we can compare them.
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
print(f"  Found {len(versions)} version(s).\n")

# ---------------------------------------------------------------
# STEP 3: Look up each version's F1 score and find the best one
# ---------------------------------------------------------------
print("STEP 3: Comparing F1 scores...")
# A registered "version" just points at a training run under the hood.
# The actual F1 score isn't stored on the version itself - it's a
# metric logged on that run - so we fetch the run behind each version
# to read it out.
best_version = None
best_f1 = -1

for v in versions:
    run = client.get_run(v.run_id)
    f1 = run.data.metrics.get("f1_score")
    run_name = run.data.tags.get("mlflow.runName", v.run_id)
    print(f"  Version {v.version} (run: {run_name}): F1 = {f1}")

    if f1 is not None and f1 > best_f1:
        best_f1 = f1
        best_version = v.version

print(f"\n  Best: version {best_version} with F1 = {best_f1:.4f}\n")

# ---------------------------------------------------------------
# STEP 4: Point the "champion" alias at the winning version
# ---------------------------------------------------------------
print("STEP 4: Setting 'champion' alias...")
# An alias is just a movable label MLflow lets you attach to a
# version, like a nickname. Instead of the API needing to know
# "version 4 is currently best", it just asks for whichever version
# is labeled "champion" right now. Re-running this script later (after
# a new model is registered) simply moves the label to a new version.
client.set_registered_model_alias(MODEL_NAME, "champion", best_version)
print(f"  Done. '{MODEL_NAME}@champion' now points to version {best_version}.\n")

# ---------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------
print("=" * 60)
print("CHAMPION PROMOTION SUMMARY")
print("=" * 60)
print(f"Compared {len(versions)} registered versions")
print(f"Champion: version {best_version}, F1 = {best_f1:.4f}")
print(f"FastAPI will load this model via: models:/{MODEL_NAME}@champion")