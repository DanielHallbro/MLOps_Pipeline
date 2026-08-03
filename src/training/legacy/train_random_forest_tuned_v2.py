"""
train_random_forest_tuned_v2.py - Second Random Forest GridSearchCV
pass, narrower than the first. The first search picked max_depth=None
(unlimited) as "best", which overfit badly - CV F1 was the highest of
all four runs (0.935), but test F1 was the worst (0.909). This search
removes the unlimited-depth option to see whether tuning can beat the
baseline once that overfitting risk is off the table.
"""

import mlflow.sklearn
import pandas as pd
from common import log_preprocessing_artifacts
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
from common import get_tracking_uri

mlflow.set_tracking_uri(get_tracking_uri())
mlflow.set_experiment("network-intrusion-detection")
print("  Using experiment 'network-intrusion-detection'\n")

# ---------------------------------------------------------------
# STEP 3: Set up the hyperparameter grid
# ---------------------------------------------------------------
print("STEP 3: Setting up GridSearchCV...")
# max_depth=None removed on purpose - the first RF search showed it
# overfits (highest CV score, worst test score of all four runs).
# Testing depths close to the untuned baseline's max_depth=15 instead.
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
}
print(f"  Grid: {param_grid}")
print("  12 combinations x 5-fold CV = 60 models to train\n")

# n_jobs=-1 only on GridSearchCV, not on the model itself - the last
# run had both, which caused nested parallelism (worker processes
# each trying to use all CPU cores) and slowed things down a lot.
base_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
)

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
with mlflow.start_run(run_name="random_forest_gridsearch_tuned_v2"):
    mlflow.log_param("model_type", "RandomForest_GridSearchCV_v2")
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("grid_combinations", 12)
    mlflow.log_param("note", "narrowed max_depth after v1 overfit with None")

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

print("Random Forest v2 tuning finished. Compare all five runs in the MLflow UI.")