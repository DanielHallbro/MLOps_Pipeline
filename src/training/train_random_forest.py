"""
train_random_forest.py - Trains a Random Forest baseline on the
processed UNSW-NB15 data and logs everything to MLflow.

This is the first of two models being compared (XGBoost comes next).
Random Forest is trained first because it's simpler to reason about
and gives a baseline to compare XGBoost against later.
"""

import mlflow.sklearn
import pandas as pd
from common import log_preprocessing_artifacts
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow

# ---------------------------------------------------------------
# STEP 1: Load the already-processed data
# ---------------------------------------------------------------
print("STEP 1: Loading processed data...")
train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

X_train = train.drop(columns=["label"])
y_train = train["label"]
X_test = test.drop(columns=["label"])
y_test = test["label"]
print(f"  Train: {X_train.shape}, Test: {X_test.shape}\n")

# ---------------------------------------------------------------
# STEP 2: Connect to MLflow and start a run
# ---------------------------------------------------------------
print("STEP 2: Connecting to MLflow...")
from common import get_tracking_uri

mlflow.set_tracking_uri(get_tracking_uri())
mlflow.set_experiment("network-intrusion-detection-v2")
print("  Experiment set to 'network-intrusion-detection-v2'\n")

with mlflow.start_run(run_name="random_forest_baseline"):

    # -------------------------------------------------------
    # STEP 3: Train the model
    # -------------------------------------------------------
    print("STEP 3: Training Random Forest...")
    n_estimators = 100
    # Number of trees. More = a more stable average vote, but
    # returns shrink fast past ~100.
    max_depth = 15
    # Max depth per tree. Deeper trees can memorize noise
    # (overfitting) - 15 beat shallower options in testing.

    # class_weight="balanced": weights the rarer class higher during
    # training so the model can't win by just favoring the common
    # class. Doesn't touch the actual data, only the training cost.

    # random_state=42: fixes all randomness so re-running this script
    # produces the identical model every time.

    # n_jobs=-1: uses all CPU cores for THIS training run. Not the
    # same as Airflow's PARALLELISM=1, which limits how many
    # DIFFERENT scripts run at once - they don't conflict.
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  Training complete.\n")

    # -------------------------------------------------------
    # STEP 4: Log the parameters used
    # -------------------------------------------------------
    print("STEP 4: Logging parameters to MLflow...")
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("class_weight", "balanced")
    print("  Logged.\n")

    # -------------------------------------------------------
    # STEP 5: Evaluate on the test set
    # -------------------------------------------------------
    print("STEP 5: Evaluating on test data...")
    # predict() averages each tree's predicted probability (soft
    # voting), not a simple majority vote of hard yes/no answers.
    y_pred = model.predict(X_test)

    # accuracy: % correct overall - can look good even on a weak
    # model here since classes aren't balanced (68/32).
    accuracy = accuracy_score(y_test, y_pred)
    # precision: of what we called "attack", how much really was.
    precision = precision_score(y_test, y_pred)
    # recall: of all real attacks, how many we caught - missing this
    # is the worse failure mode for an intrusion detector.
    recall = recall_score(y_test, y_pred)
    # f1: balances precision and recall - the metric that actually
    # picks the champion model (see promote_champion.py).
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}\n")

    # -------------------------------------------------------
    # STEP 6: Log metrics and the trained model itself
    # -------------------------------------------------------
    print("STEP 6: Logging metrics and model to MLflow...")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(
        model, "model",
        registered_model_name="network-intrusion-detector"
    )
    log_preprocessing_artifacts()
    print("  Logged. Run complete.\n")

print("Random Forest baseline finished. Check the MLflow UI to see this run.")