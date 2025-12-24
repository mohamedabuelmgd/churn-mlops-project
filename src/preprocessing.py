import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


RAW_DATA_PATH = "data/raw/Telco_Customer_Churn.csv"
PROCESSED_DATA_DIR = "data/processed"
PREPROCESSOR_PATH = "models/preprocessor.joblib"


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df


def preprocess_data(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numerical_features),
            ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    return X, y, preprocessor


def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def save_artifacts(X_train, X_test, y_train, y_test, preprocessor):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train.to_csv(f"{PROCESSED_DATA_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DATA_DIR}/y_test.csv", index=False)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)


if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    save_artifacts(X_train, X_test, y_train, y_test, preprocessor)

    print(" Data preprocessing completed successfully")
