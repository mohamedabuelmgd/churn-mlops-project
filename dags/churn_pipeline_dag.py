from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="churn_mlops_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="cd /c/Users/zizo/Documents/churn-mlops-project && dvc repro preprocess"
    )

    train = BashOperator(
        task_id="train",
        bash_command="cd /c/Users/zizo/Documents/churn-mlops-project && dvc repro train"
    )

    preprocess >> train
