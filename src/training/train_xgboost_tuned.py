"""
train_xgboost_tuned.py - Runs GridSearchCV over XGBoost hyperparameters
and logs the best model to MLflow, so it can be compared against the
untuned xgboost_baseline and random_forest_baseline runs.
"""

import mlflow.xgboost
import pandas as pd
from common import log_preprocessing_artifacts
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
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
# STEP 2: Connect to MLflow (same experiment as the other two runs)
# ---------------------------------------------------------------
print("STEP 2: Connecting to MLflow...")
from common import get_tracking_uri

mlflow.set_tracking_uri(get_tracking_uri())
mlflow.set_experiment("network-intrusion-detection")
print("  Using experiment 'network-intrusion-detection'\n")

# ---------------------------------------------------------------
# STEP 3: Set up the hyperparameter grid
# ---------------------------------------------------------------
print("STEP 3: Setting up GridSearchCV...")
# Kept deliberately small (2x2x2 = 8 combinations) so the search
# finishes in a reasonable time for a live demo. The same code scales
# to a wider search by just adding more values to each list below.
n_normal = (y_train == 0).sum()
n_attack = (y_train == 1).sum()
scale_pos_weight = n_normal / n_attack

param_grid = {
    "max_depth": [3, 6],
    "n_estimators": [50, 100],
    "learning_rate": [0.1, 0.2],
}
print(f"  Grid: {param_grid}")
print("  8 combinations x 5-fold CV = 40 models to train\n")

base_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
)

# scoring='f1' because for intrusion detection, missing real attacks
# (recall) and raising false alarms (precision) both matter - f1
# balances the two rather than optimizing for raw accuracy.
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

# ---------------------------------------------------------------
# STEP 4: Run the search
# ---------------------------------------------------------------
print("STEP 4: Running GridSearchCV (this takes a few minutes)...")
grid_search.fit(X_train, y_train)
print("  Search complete.\n")

best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_
print(f"  Best parameters: {best_params}")
print(f"  Best CV F1-score: {best_cv_score:.4f}\n")

# ---------------------------------------------------------------
# STEP 5: Evaluate the best model on the test set
# ---------------------------------------------------------------
print("STEP 5: Evaluating best model on test data...")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-score:  {f1:.4f}\n")

# ---------------------------------------------------------------
# STEP 6: Log everything to MLflow
# ---------------------------------------------------------------
print("STEP 6: Logging to MLflow...")
with mlflow.start_run(run_name="xgboost_gridsearch_tuned"):
    mlflow.log_param("model_type", "XGBoost_GridSearchCV")
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("grid_combinations", 8)

    mlflow.log_metric("best_cv_f1", best_cv_score)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.xgboost.log_model(
        best_model, "model",  # or `model` for the untuned baseline script
        registered_model_name="network-intrusion-detector"
    )
    log_preprocessing_artifacts()
print("  Logged. Run complete.\n")

print("GridSearchCV tuning finished. Compare all three runs in the MLflow UI.")