from prefect import flow, task, get_run_logger
import subprocess
@task(name="Run DVC Preprocessing")
def run_preprocess():
    logger = get_run_logger()
    logger.info("Starting preprocessing...")
    result = subprocess.run(
        ["dvc", "repro", "preprocess"],
        capture_output=True,
        text=True,
        check=True
    )
    logger.info(result.stdout)
    if result.stderr:
        logger.error(result.stderr)
    logger.info("Preprocessing finished")

@task(name="Run DVC Training")
def run_training():
    logger = get_run_logger()
    logger.info("Starting training...")
    result = subprocess.run(
        ["dvc", "repro", "train"],
        capture_output=True,
        text=True,
        check=True
    )
    logger.info(result.stdout)
    if result.stderr:
        logger.error(result.stderr)
    logger.info("Training finished")

@task(name="Run DVC Evaluation")
def run_evaluation():
    logger = get_run_logger()
    logger.info("Starting evaluation...")
    result = subprocess.run(
        ["dvc", "repro", "evaluate"],
        capture_output=True,
        text=True,
        check=True
    )
    logger.info(result.stdout)
    if result.stderr:
        logger.error(result.stderr)
    logger.info("Evaluation finished")

@flow(name="Churn MLOps Pipeline")
def churn_pipeline():
    run_preprocess()
    run_training()
    run_evaluation()

if __name__ == "__main__":
    churn_pipeline()
