"""
test_feature_importance.py - Checks whether removing weak/noisy
features improves the model on the real, official test set.

Random Forest baseline (n_estimators=100, max_depth=15) is re-trained
here to extract feature importances, then a second model is trained
using only the strongest features, to see if trimming weak columns
helps rather than assuming it does.
"""

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow

# ---------------------------------------------------------------
# STEP 1: Load the processed data
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
# STEP 2: Connect to MLflow
# ---------------------------------------------------------------
print("STEP 2: Connecting to MLflow...")
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("network-intrusion-detection")
print("  Using experiment 'network-intrusion-detection'\n")


def train_and_log(X_train, y_train, X_test, y_test, run_name, feature_note):
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
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("features", feature_note)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

    return model, f1


# ---------------------------------------------------------------
# STEP 3: Train on all features (this is the current baseline) and
# extract feature importances
# ---------------------------------------------------------------
print("STEP 3: Training on all features to get importances...")
full_model, f1_full = train_and_log(
    X_train, y_train, X_test, y_test,
    "rf_all_features", f"all {X_train.shape[1]} features"
)

importances = pd.Series(
    full_model.feature_importances_, index=X_train.columns
).sort_values(ascending=False)

print("\n  Top 15 most important features:")
print(importances.head(15))
print("\n  Bottom 15 least important features:")
print(importances.tail(15))
print()

# ---------------------------------------------------------------
# STEP 4: Keep only the strongest features and retrain
# ---------------------------------------------------------------
print("STEP 4: Training on top features only...")
# Keep features that together account for the top 90% of total
# importance, a common, defensible cutoff rather than an arbitrary
# fixed number of columns.
cumulative = importances.cumsum() / importances.sum()
top_features = importances[cumulative <= 0.90].index.tolist()
if len(top_features) == 0:  # safety net in case the first feature alone exceeds 90%
    top_features = [importances.index[0]]

print(f"  Keeping {len(top_features)} of {X_train.shape[1]} features (90% of total importance)")
print(f"  Dropped: {[c for c in X_train.columns if c not in top_features]}\n")

X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

reduced_model, f1_reduced = train_and_log(
    X_train_top, y_train, X_test_top, y_test,
    "rf_top_features_only", f"top {len(top_features)} features (90% importance)"
)

# ---------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("FEATURE SELECTION IMPACT SUMMARY")
print("=" * 60)
print(f"F1 with all {X_train.shape[1]} features:       {f1_full:.4f}")
print(f"F1 with top {len(top_features)} features:  {f1_reduced:.4f}")
diff = f1_reduced - f1_full
if abs(diff) < 0.001:
    print("Difference is negligible - feature count doesn't meaningfully change results.")
elif diff > 0:
    print(f"Removing weak features improved F1 by {diff:.4f}.")
else:
    print(f"Removing weak features hurt F1 by {abs(diff):.4f} - keep all features.")