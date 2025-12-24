import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import yaml

PROCESSED_DATA_DIR = "data/processed"
PREPROCESSOR_PATH = "models/preprocessor.joblib"


def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_DIR}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{PROCESSED_DATA_DIR}/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


def build_model():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    model_params = params["model"]
    model = RandomForestClassifier(
         n_estimators=model_params["n_estimators"],
         max_depth=model_params["max_depth"],
         random_state=42
    )
    return model


def train_model(model, preprocessor, X_train, y_train):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }


if __name__ == "__main__":
    mlflow.set_experiment("churn_prediction")

    X_train, X_test, y_train, y_test = load_data()
    preprocessor = load_preprocessor()
    model = build_model()

    with mlflow.start_run():
        pipeline = train_model(model, preprocessor, X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 10,
            "model_type": "RandomForest"
        })

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model",registered_model_name="ChurnPredictionModel")
        joblib.dump(pipeline, "models/model.joblib")

        print("Training & MLflow logging completed")
