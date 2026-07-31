"""
retrain_pipeline.py - Retrains all candidate models and promotes the
best one to champion in MLflow.

The four candidate scripts are independent experiments (no data
dependency between them), so they run in parallel. promote_champion
only runs once ALL four have finished, since it needs to compare every
candidate's F1 score to pick a winner - that dependency is enforced
explicitly below, not just assumed.
"""

import os
import requests
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

PROJECT_DIR = "/opt/airflow/project"


def notify_failure(context):
    """
    Runs automatically if any task in this DAG fails. Always logs to
    the task's own Airflow logs; additionally posts to Slack if
    SLACK_WEBHOOK_URL is configured. The webhook call is wrapped in
    its own try/except so a Slack outage or bad URL never masks the
    real failure or crashes the callback itself.
    """
    task_instance = context["task_instance"]
    message = (
        f"Task '{task_instance.task_id}' failed in DAG "
        f"'{task_instance.dag_id}' at {context['logical_date']}"
    )
    print(f"TASK FAILED: {message}")

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook_url:
        try:
            requests.post(webhook_url, json={"text": f":x: {message}"}, timeout=5)
        except Exception as e:
            print(f"Slack notification failed (non-fatal): {e}")


default_args = {
    "owner": "daniel",
    "retries": 0,
    "on_failure_callback": notify_failure,
}

with DAG(
    dag_id="network_intrusion_retraining",
    description="Retrains all candidate models and promotes the best one to champion",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    train_random_forest = BashOperator(
        task_id="train_random_forest",
        bash_command=f"cd {PROJECT_DIR} && python3 src/training/train_random_forest.py",
    )

    train_xgboost = BashOperator(
        task_id="train_xgboost",
        bash_command=f"cd {PROJECT_DIR} && python3 src/training/train_xgboost.py",
    )

    train_xgboost_tuned = BashOperator(
        task_id="train_xgboost_tuned",
        bash_command=f"cd {PROJECT_DIR} && python3 src/training/train_xgboost_tuned.py",
    )

    train_random_forest_tuned_v2 = BashOperator(
        task_id="train_random_forest_tuned_v2",
        bash_command=f"cd {PROJECT_DIR} && python3 src/training/train_random_forest_tuned_v2.py",
    )

    promote_champion = BashOperator(
        task_id="promote_champion",
        bash_command=f"cd {PROJECT_DIR} && python3 src/training/promote_champion.py",
    )

    [
        train_random_forest,
        train_xgboost,
        train_xgboost_tuned,
        train_random_forest_tuned_v2,
    ] >> promote_champion