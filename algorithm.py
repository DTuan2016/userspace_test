#!/usr/bin/env python3
import numpy as np
import pandas as pd
import joblib, os
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class SimpleMLP(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(num_features, num_classes)  # 1 lớp Linear duy nhất
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input shape: (batch, num_features)
        x = self.relu(self.fc1(x))
        return self.softmax(x)
    
# ======== TRAINING ========
def train_mlp_torch(X: np.ndarray, y: np.ndarray,
                    model_path: str = "model/mlp_torch.pth",
                    epochs: int = 20,
                    batch_size: int = 32,
                    lr: float = 1e-3,
                    device: str = "cpu"):
    """
    Train MLP bằng PyTorch.
    - Chỉ nhận X, y đã được xử lý (ví dụ: log2 transform).
    - Lưu state_dict vào model_path, và lưu label classes vào model_path + ".labels.pkl".
    - Trả về object model (nn.Module) đã train (trên CPU).
    """
    if torch is None:
        raise RuntimeError("PyTorch chưa được cài trong môi trường này.")

    # encode labels
    y_enc, classes = pd.factorize(y)
    num_classes = len(classes)
    num_features = X.shape[1]

    # tensors: shape (N, num_features)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_enc, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleMLP(num_features=num_features, num_classes=num_classes)
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f"[MLP-Torch] Epoch {ep}/{epochs} - loss: {avg_loss:.4f}")

    # save
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(classes, model_path + ".labels.pkl")
    print(f"[+] MLP (PyTorch) trained and saved to {model_path} (+ labels.pkl)")

    # return model on CPU
    model_cpu = SimpleMLP(num_features=num_features, num_classes=num_classes)
    model_cpu.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_cpu.eval()
    return model_cpu


# ======== TIỀN XỬ LÝ + TRAIN ========
def process_mlp_torch(data_csv, label_col="Label",
                      model_path="model/mlp_torch.pth",
                      epochs: int = 20, batch_size: int = 32, lr: float = 1e-3,
                      device: str = "cpu"):
    """
    Read CSV, pick feature columns, log2 transform, then call train_mlp_torch().
    Returns trained model (nn.Module on CPU).
    """
    df = pd.read_csv(data_csv)
    feature_columns = [
        "FlowDuration",
        "FlowIATMean",
        "FlowPktsPerSec",
        "FlowBytesPerSec",
        "PktLenMean"
    ]
    if not all(col in df.columns for col in feature_columns + [label_col]):
        missing = [c for c in feature_columns + [label_col] if c not in df.columns]
        raise ValueError(f"[ERROR] CSV missing columns: {missing}")

    X = df.loc[:, feature_columns].values.astype(float)
    y = df[label_col].values
    if X.shape[0] == 0:
        raise ValueError("No numeric data in CSV to train")

    X_log = np.log2(X + 1)
    model = train_mlp_torch(X_log, y, model_path=model_path,
                            epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    return model


# ======== DỰ ĐOÁN ========
def predict_mlp_torch(model: nn.Module | str, X: np.ndarray, model_path: str = None, device: str = "cpu"):
    """
    Predict with a trained PyTorch MLP.
    - model: either nn.Module instance or None (if you pass model_path)
    - X: features (same preprocessing as training)
    Returns (labels, scores)
    """
    if torch is None:
        raise RuntimeError("PyTorch chưa được cài trong môi trường này.")

    loaded_here = False
    if not isinstance(model, nn.Module):
        if model_path is None:
            raise ValueError("If model object not provided, set model_path")
        classes = joblib.load(model_path + ".labels.pkl")
        num_classes = len(classes)
        num_features = X.shape[1]
        model_local = SimpleMLP(num_features=num_features, num_classes=num_classes)
        model_local.load_state_dict(torch.load(model_path, map_location="cpu"))
        model_local.eval()
        model = model_local
        loaded_here = True
    else:
        classes = None
        if model_path and os.path.exists(model_path + ".labels.pkl"):
            classes = joblib.load(model_path + ".labels.pkl")

    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)  # (N, F)
        probs = model(X_t)
        probs_np = probs.cpu().numpy()
        preds_idx = probs_np.argmax(axis=1)
        scores = probs_np.max(axis=1)

    if 'classes' in locals() and classes is not None:
        labels = [classes[i] for i in preds_idx]
    else:
        labels = preds_idx.tolist()

    return np.array(labels, dtype=object), np.array(scores, dtype=float)

# ===========================
# TRAIN FUNCTIONS
# ===========================
def train_knn(X: np.ndarray, k: int = 5, model_path: str = "model/knn_model.pkl"):
    """
    Train KNN (NearestNeighbors).
    Chỉ dùng cho anomaly detection qua khoảng cách lân cận.
    """
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    knn.fit(X)
    joblib.dump(knn, model_path)
    print(f"[+] KNN trained and saved to {model_path}")
    return knn


def train_lof(X: np.ndarray, n_neighbors: int = 20, contamination: float = 0.1,
              model_path: str = "model/lof_model.pkl"):
    """
    Train LOF (Local Outlier Factor).
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True  # Cho phép dùng predict sau khi train
    )
    lof.fit(X)
    joblib.dump(lof, model_path)
    print(f"[+] LOF trained and saved to {model_path}")
    return lof


def train_isolation_forest(X: np.ndarray, contamination: float = 0.1,
                           model_path: str = "model/isoforest_model.pkl"):
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
    print(f"[+] IsolationForest trained and saved to {model_path}")
    return iso


def train_random_forest(X: np.ndarray, y: np.ndarray,
                        model_path: str = "model/randforest_model.pkl"):
    """
    Train Random Forest supervised classification.
    y phải là label (vd: 0/1 hoặc BENIGN/PORTMAP).
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, model_path)
    print(f"[+] RandomForest trained and saved to {model_path}")
    return rf

def train_linear_svm(X: np.ndarray, y: np.ndarray,
                     C: float = 1.0,
                     max_iter: int = 5000,
                     model_path: str = "model/linear_svm.pkl"):

    model = LinearSVC(C=C, max_iter=max_iter, random_state=42)
    model.fit(X, y)

    # os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[+] LinearSVM trained and saved to {model_path}")
    return model

# def train_cnn(X: np.ndarray, y: np.ndarray, model_path="model/cnn_model.h5",
#               epochs=20, batch_size=32, learning_rate=0.001):
#     """
#     Training simple 1D CNN cho classification
#     """

def process_linear_svm(data_csv, label_col="Label", model_path="model/linear_svm.pkl"):
    """
    Read CSV, get feature and label, train LinearSVM.
    """
    df = pd.read_csv(data_csv)
    feature_columns = [
        "FlowDuration",
        "FlowIATMean",
        "FlowPktsPerSec",
        "FlowBytesPerSec",
        "PktLenMean"
    ]

    if not all(col in df.columns for col in feature_columns + [label_col]):
        missing = [c for c in feature_columns + [label_col] if c not in df.columns]
        raise ValueError(f"[ERROR] CSV missing columns: {missing}")

    X = df.loc[:, feature_columns].values.astype(float)
    y = df[label_col].values
    X_log = np.log2(X + 1)

    return train_linear_svm(X_log, y, model_path=model_path)

def process_isolation_forest(data_csv, contamination=0.1, model_path="model/isoforest_model.pkl"):
    """
    Đọc CSV, lấy feature numeric, train Isolation Forest.
    """
    df = pd.read_csv(data_csv)

    feature_columns = [
        "FlowDuration",
        "FlowIATMean",
        "FlowPktsPerSec",
        "FlowBytesPerSec",
        "PktLenMean"
    ]

    if not all(col in df.columns for col in feature_columns):
        missing = [c for c in feature_columns if c not in df.columns]
        raise ValueError(f"[ERROR] CSV thiếu cột: {missing}")

    X = df.loc[:, feature_columns].values.astype(float)
    if X.shape[0] == 0:
        raise ValueError("Không có dữ liệu numeric trong CSV để train")

    X_log = np.log2(X + 1)

    iso = train_isolation_forest(X_log, contamination=contamination, model_path=model_path)
    return iso


def process_random_forest(data_csv, label_col="Label", model_path="model/randforest_model.pkl"):
    """
    Đọc CSV, lấy feature + label, train Random Forest (supervised classification).
    label_col: tên cột chứa nhãn (vd: "Label" với BENIGN / PORTMAP).
    """
    df = pd.read_csv(data_csv)

    feature_columns = [
        "FlowDuration",
        "FlowIATMean",
        "FlowPktsPerSec",
        "FlowBytesPerSec",
        "PktLenMean"
    ]

    if not all(col in df.columns for col in feature_columns + [label_col]):
        missing = [c for c in feature_columns + [label_col] if c not in df.columns]
        raise ValueError(f"[ERROR] CSV thiếu cột: {missing}")

    X = df.loc[:, feature_columns].values.astype(float)
    y = df[label_col].values

    if X.shape[0] == 0:
        raise ValueError("Không có dữ liệu numeric trong CSV để train")

    X_log = np.log2(X + 1)

    rf = train_random_forest(X_log, y, model_path=model_path)
    return rf

# ===========================
# PREDICT FUNCTIONS
# ===========================
def predict_with_score(model, X: np.ndarray, threshold: float = None):
    """
    Trả về (labels, scores) cho mọi model.
    - LOF & IsolationForest: dùng decision_function
    - KNN: dùng khoảng cách trung bình
    - RandomForest: dùng predict_proba nếu có
    """
    # --- LOF ---
    if isinstance(model, LocalOutlierFactor):
        labels = model.predict(X)
        scores = model.decision_function(X)
        return labels, scores

    # --- IsolationForest ---
    elif isinstance(model, IsolationForest):
        labels = model.predict(X)
        scores = model.decision_function(X)
        return labels, scores

    # --- KNN ---
    elif isinstance(model, NearestNeighbors):
        distances, _ = model.kneighbors(X)
        scores = distances.mean(axis=1)  # score = khoảng cách trung bình
        if threshold is None:
            threshold = np.percentile(scores, 95)  # lấy ngưỡng 95%
        labels = np.where(scores > threshold, -1, 1)
        return labels, scores

    # --- RandomForest (supervised) ---
    elif isinstance(model, RandomForestClassifier):
        probs = model.predict_proba(X)
        labels = model.predict(X)
        # Score = xác suất của class dự đoán
        scores = np.max(probs, axis=1)
        return labels, scores
    
    # --- Support Vector Machine ---
    elif isinstance(model, LinearSVC):
        scores = model.decision_function(X)
        labels = model.predict(X)
        # Ép kiểu về mảng float 1D (tránh lỗi khi in f"{score:.4f}")
        scores = np.array(scores, dtype=float).reshape(-1)
        return labels, scores
    
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

# ===========================
# UTILITIES
# ===========================
def load_model(path: str):
    """Load model từ file .pkl"""
    return joblib.load(path)


def process_lof(data_csv, n_neighbors=20, contamination=0.1,
                model_path="model/lof_model.pkl"):
    """
    Đọc CSV, lấy feature, train LOF (dùng log2 transform).
    """
    df = pd.read_csv(data_csv)
    feature_columns = [
        "FlowDuration",
        "FlowIATMean",
        "FlowPktsPerSec",
        "FlowBytesPerSec",
        "PktLenMean"
    ]

    if not all(col in df.columns for col in feature_columns):
        missing = [c for c in feature_columns if c not in df.columns]
        raise ValueError(f"[ERROR] CSV thiếu cột: {missing}")

    X = df.loc[:, feature_columns].values.astype(float)
    if X.shape[0] == 0:
        raise ValueError("Không có dữ liệu numeric trong CSV để train")

    X_log = np.log2(X + 1)

    lof = train_lof(X_log, n_neighbors=n_neighbors,
                    contamination=contamination, model_path=model_path)
    return lof
