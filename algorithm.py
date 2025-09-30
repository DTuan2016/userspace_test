#!/usr/bin/env python3
import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import joblib
import pandas as pd


def train_knn(X: np.ndarray, k: int = 3, model_path: str = "knn_model.pkl"):
    """
    Train KNN (NearestNeighbors) model.
    X: numpy array (n_samples, n_features)
    k: number of neighbors
    Save .pkl to model path
    """
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    knn.fit(X)
    joblib.dump(knn, model_path)
    print(f"[+] KNN trained and saved to {model_path}")
    return knn


def train_lof(X: np.ndarray, n_neighbors: int = 20, contamination: float = 0.1, model_path: str = "lof_model.pkl"):
    """
    Train Local Outlier Factor (unsupervised anomaly detection).
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True 
    )
    lof.fit(X)
    joblib.dump(lof, model_path)
    print(f"[+] LOF trained and saved to {model_path}")
    return lof

def process_lof(data_csv, n_neighbors=20, contamination=0.1, model_path="lof_model.pkl"):
    """
    Đọc CSV, lấy dữ liệu numeric và train LOF.
    """
    # Đọc CSV
    df = pd.read_csv(data_csv)

    # Lấy toàn bộ cột số (loại bỏ text)
    X = df.select_dtypes(include=["int64", "float64"]).values

    if X.shape[0] == 0:
        raise ValueError("Không có dữ liệu numeric trong CSV để train")

    # Train LOF
    lof = train_lof(X, n_neighbors=n_neighbors,
                              contamination=contamination,
                              model_path=model_path)
    return lof


def train_isolation_forest(X: np.ndarray, contamination: float = 0.1, model_path: str = "isoforest_model.pkl"):
    """
    Train Isolation Forest.
    """
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42
    )
    iso.fit(X)
    joblib.dump(iso, model_path)
    print(f"[+] Isolation Forest trained and saved to {model_path}")
    return iso


def load_model(path: str):
    """Load model from file joblib"""
    return joblib.load(path)


def predict(model, X: np.ndarray):
    """
    Run predict to anomaly detection.
    - With LOF and IsolationForest: model.predict(X) return {1: normal, -1: anomaly}
    - With KNN: Define threshold.
    """
    if hasattr(model, "predict"):
        return model.predict(X)
    elif isinstance(model, NearestNeighbors):
        distances, _ = model.kneighbors(X)
        return distances
    else:
        raise ValueError("Unsupported model type")
