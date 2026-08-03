"""
test_ratio_features_impact.py - Measures whether the two ratio
features (sbytes_dbytes_ratio, spkts_dpkts_ratio) actually help the
winning model (Random Forest baseline), rather than assuming they do.

Trains the exact same Random Forest configuration twice: once on the
full processed data (with ratio features), once with those two
columns dropped. Same train/test split, same hyperparameters - the
only difference is the two columns, so any difference in the result
can be attributed to them specifically.
"""

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow

RATIO_FEATURES = ["sbytes_dbytes_ratio", "spkts_dpkts_ratio"]

# ---------------------------------------------------------------
# STEP 1: Load the already-processed data
# ---------------------------------------------------------------
print("STEP 1: Loading processed data...")
train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")
print(f"  Train: {train.shape}, Test: {test.shape}\n")

# ---------------------------------------------------------------
# STEP 2: Connect to MLflow
# ---------------------------------------------------------------
print("STEP 2: Connecting to MLflow...")
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("network-intrusion-detection")
print("  Using experiment 'network-intrusion-detection'\n")


def train_and_log(train_df, test_df, run_name, feature_note):
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  [{run_name}] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1: {f1:.4f}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 15)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("features", feature_note)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

    return f1


# ---------------------------------------------------------------
# STEP 3: Train WITH ratio features (this is the current baseline)
# ---------------------------------------------------------------
print("STEP 3: Training WITH ratio features...")
f1_with = train_and_log(train, test, "rf_with_ratio_features", "with ratio features")
print()

# ---------------------------------------------------------------
# STEP 4: Train WITHOUT ratio features
# ---------------------------------------------------------------
print("STEP 4: Training WITHOUT ratio features...")
train_no_ratio = train.drop(columns=RATIO_FEATURES)
test_no_ratio = test.drop(columns=RATIO_FEATURES)
f1_without = train_and_log(train_no_ratio, test_no_ratio, "rf_without_ratio_features", "without ratio features")
print()

# ---------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------
print("=" * 60)
print("RATIO FEATURE IMPACT SUMMARY")
print("=" * 60)
print(f"F1 with ratio features:    {f1_with:.4f}")
print(f"F1 without ratio features: {f1_without:.4f}")
diff = f1_with - f1_without
if abs(diff) < 0.001:
    print("Difference is negligible - ratio features have no measurable impact here.")
elif diff > 0:
    print(f"Ratio features improved F1 by {diff:.4f}.")
else:
    print(f"Ratio features hurt F1 by {abs(diff):.4f} - consider dropping them.")