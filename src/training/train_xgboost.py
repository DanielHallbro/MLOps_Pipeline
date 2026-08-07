"""
train_xgboost.py - Trains an XGBoost baseline on the processed
UNSW-NB15 data and logs it to the same MLflow experiment as the
Random Forest run, so the two can be compared side by side.
"""

import mlflow.xgboost
import pandas as pd
from common import log_preprocessing_artifacts
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

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
# STEP 2: Connect to MLflow (same experiment as Random Forest)
# ---------------------------------------------------------------
print("STEP 2: Connecting to MLflow...")
from common import get_tracking_uri

mlflow.set_tracking_uri(get_tracking_uri())
mlflow.set_experiment("network-intrusion-detection-v2")
print("  Using experiment 'network-intrusion-detection-v2'\n")

with mlflow.start_run(run_name="xgboost_baseline"):

    # -------------------------------------------------------
    # STEP 3: Train the model
    # -------------------------------------------------------
    print("STEP 3: Training XGBoost...")
    # scale_pos_weight is XGBoost's equivalent of class_weight='balanced'.
    # It's the ratio of the majority class to the minority class, here
    # normal (32%) vs attack (68%), so we weight the smaller class up.
    n_normal = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()
    scale_pos_weight = n_normal / n_attack

    n_estimators = 100
    # Boosting rounds - NOT the same idea as RF's n_estimators. RF's
    # trees are independent and vote at the end. XGBoost's trees are
    # sequential: each new one corrects the current ensemble's errors.

    max_depth = 6
    # Much shallower than RF's 15, on purpose: since boosting is
    # sequential, a deep tree's mistakes get inherited by every tree
    # after it. Shallow keeps each correction small and controlled.

    learning_rate = 0.1
    # Weight given to each new tree's correction - lower is
    # safer/slower. 0.1 is the standard default.

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        # Loss function XGBoost optimizes DURING training - different
        # from F1, which only gets computed AFTER, to compare models.
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  Training complete.\n")

    # -------------------------------------------------------
    # STEP 4: Log the parameters used
    # -------------------------------------------------------
    print("STEP 4: Logging parameters to MLflow...")
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))
    print("  Logged.\n")

    # -------------------------------------------------------
    # STEP 5: Evaluate on the test set
    # -------------------------------------------------------
    print("STEP 5: Evaluating on test data...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
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
    mlflow.xgboost.log_model(
        model, "model",  # or `model` for the untuned baseline script
        registered_model_name="network-intrusion-detector"
    )
    log_preprocessing_artifacts()
    print("  Logged. Run complete.\n")

print("XGBoost baseline finished. Compare it with random_forest_baseline in the MLflow UI.")