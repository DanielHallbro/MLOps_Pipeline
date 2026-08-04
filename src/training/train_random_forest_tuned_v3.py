"""
train_random_forest_tuned_v3.py - Third Random Forest GridSearchCV
pass. v1 (see legacy/) overfit with max_depth=None. v2 (see legacy/)
fixed that but used a wide grid (12 combinations x 5-fold = 60 fits)
that made it by far the slowest script in the pipeline, out of step
with every other candidate's runtime. v3 narrows the grid and folds
to bring runtime in line with the rest of the pipeline, while still
demonstrating the same GridSearchCV pattern. Every prior tuning
attempt lost to the untuned baseline anyway, so a narrower search
costs nothing in terms of the actual conclusion.
"""

import mlflow.sklearn
import pandas as pd
from common import get_tracking_uri, log_preprocessing_artifacts
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

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
# STEP 2: Connect to MLflow (same experiment as the other runs)
# ---------------------------------------------------------------
print("STEP 2: Connecting to MLflow...")
mlflow.set_tracking_uri(get_tracking_uri())
mlflow.set_experiment("network-intrusion-detection-v2")
print("  Using experiment 'network-intrusion-detection-v2'\n")

# ---------------------------------------------------------------
# STEP 3: Set up a small, fast hyperparameter grid
# ---------------------------------------------------------------
print("STEP 3: Setting up GridSearchCV...")
# max_depth=None removed (v1 overfit with it). Grid deliberately
# avoids max_depth=15/n_estimators=100 (the exact baseline
# configuration) so the search explores genuinely different territory
# rather than just re-finding the baseline. Grid and CV folds both
# narrowed from v2 to keep runtime comparable to the other candidate
# scripts - 8 combinations x 2-fold = 16 fits, versus v2's 60.
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [10, 12],
    "min_samples_split": [2, 4],
}
print(f"  Grid: {param_grid}")
print("  8 combinations x 2-fold CV = 16 models to train\n")

base_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="f1",
    cv=2,  # was 3
    n_jobs=-1,
    verbose=1,
)

# ---------------------------------------------------------------
# STEP 4: Run the search
# ---------------------------------------------------------------
print("STEP 4: Running GridSearchCV...")
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
# STEP 6: Log everything to MLflow and register the model
# ---------------------------------------------------------------
print("STEP 6: Logging to MLflow...")
with mlflow.start_run(run_name="random_forest_gridsearch_tuned_v3"):
    mlflow.log_param("model_type", "RandomForest_GridSearchCV_v3")
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("cv_folds", 2)
    mlflow.log_param("grid_combinations", 8)
    mlflow.log_param("note", "narrowed grid/folds vs v2 to reduce runtime")

    mlflow.log_metric("best_cv_f1", best_cv_score)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        best_model, "model",
        registered_model_name="network-intrusion-detector"
    )
    log_preprocessing_artifacts()
print("  Logged. Run complete.\n")

print("Random Forest v3 tuning finished. Compare all runs in the MLflow UI.")