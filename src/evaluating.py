import joblib
import pandas as pd
from sklearn.metrics import classification_report


def load():
    pipeline = joblib.load("models/model.joblib")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return pipeline, X_test, y_test


if __name__ == "__main__":
    pipeline, X_test, y_test = load()
    preds = pipeline.predict(X_test)

    report = classification_report(y_test, preds)

    with open("reports/metrics.txt", "w") as f:
        f.write(report)

    print(report)
    print("Evaluation report saved to reports/metrics.txt") 