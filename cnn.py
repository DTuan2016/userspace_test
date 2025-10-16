import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os, joblib, csv
import numpy as np
import pandas as pd

# ========================
# Định nghĩa mô hình CNN
# ========================
class Simple1DCNN(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear((num_features // 2) * 64, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return self.softmax(x)

# ========================
# Train CNN
# ========================
def train_cnn_torch(X: np.ndarray, y: np.ndarray,
                    model_path: str = "model/cnn_torch.pth",
                    epochs: int = 20, batch_size: int = 32, lr: float = 1e-3,
                    device: str = "cpu"):
    if torch is None:
        raise RuntimeError("PyTorch chưa được cài đặt.")

    # Encode nhãn
    y_enc, classes = pd.factorize(y)
    num_classes = len(classes)
    num_features = X.shape[1]

    # Tensor
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_enc, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = Simple1DCNN(num_features=num_features, num_classes=num_classes)
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model.to(device)

    # Huấn luyện
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[CNN-Torch] Epoch {ep}/{epochs} - loss: {total_loss / len(loader):.4f}")

    # Lưu model
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(classes, model_path + ".labels.pkl")

    # Trả model về CPU
    model_cpu = Simple1DCNN(num_features=num_features, num_classes=num_classes)
    model_cpu.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_cpu.eval()

    print(f"[+] CNN trained & saved to {model_path}")
    return model_cpu

# ========================
# Xử lý CSV đầu vào
# ========================
def process_cnn_torch(data_csv, label_col="Label",
                      model_path="model/cnn_torch.pth",
                      epochs=20, batch_size=32, lr=1e-3, device="cpu"):
    df = pd.read_csv(data_csv)
    features = ["FlowDuration", "FlowIATMean", "FlowPktsPerSec", "FlowBytesPerSec", "PktLenMean"]

    if not all(c in df.columns for c in features + [label_col]):
        missing = [c for c in features + [label_col] if c not in df.columns]
        raise ValueError(f"Thiếu cột: {missing}")

    X = df[features].values.astype(float)
    y = df[label_col].values
    X_log = np.log2(X + 1)

    model = train_cnn_torch(X_log, y, model_path, epochs, batch_size, lr, device)
    export_cnn_weights_to_csv(model, model_path.replace(".pth", "_weights.csv"))
    return model

# ========================
# Dự đoán
# ========================
def predict_cnn_torch(model: nn.Module | str, X: np.ndarray, model_path=None, device="cpu"):
    if torch is None:
        raise RuntimeError("PyTorch chưa được cài đặt.")

    # Load model nếu cần
    if not isinstance(model, nn.Module):
        if model_path is None:
            raise ValueError("Phải truyền model_path nếu model chưa được khởi tạo.")
        classes = joblib.load(model_path + ".labels.pkl")
        num_classes = len(classes)
        num_features = X.shape[1]
        model = Simple1DCNN(num_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    else:
        classes = joblib.load(model_path + ".labels.pkl") if model_path else None

    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model.to(device).eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        probs = model(X_t)
        probs_np = probs.cpu().numpy()
        preds_idx = probs_np.argmax(axis=1)
        scores = probs_np.max(axis=1)

    labels = [classes[i] for i in preds_idx] if classes is not None else preds_idx.tolist()
    return np.array(labels, dtype=object), np.array(scores, dtype=float)

# ========================
# Xuất trọng số ra CSV
# ========================
def export_cnn_weights_to_csv(model: torch.nn.Module, output_csv: str):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param_name", "param_shape", "param_values"])
        for name, param in model.state_dict().items():
            data = param.detach().cpu().numpy().flatten()
            shape = list(param.shape)
            values = " ".join(map(str, data.tolist()))
            writer.writerow([name, str(shape), values])
    print(f"[+] Exported weights to {output_csv}")

# ========================
# CLI
# ========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and export CNN (PyTorch)")
    parser.add_argument("--csv", type=str, required=True, help="Đường dẫn tới file CSV huấn luyện")
    parser.add_argument("--model", type=str, default="model/cnn_torch.pth", help="Đường dẫn lưu model")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch huấn luyện")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Thiết bị train")
    args = parser.parse_args()

    print("=== [1] Training CNN (PyTorch) ===")
    model = process_cnn_torch(args.csv, model_path=args.model, epochs=args.epochs, device=args.device)

    print("\n=== [2] Predict 5 mẫu ngẫu nhiên ===")
    df = pd.read_csv(args.csv)
    feats = ["FlowDuration", "FlowIATMean", "FlowPktsPerSec", "FlowBytesPerSec", "PktLenMean"]
    X = np.log2(df[feats].values.astype(float) + 1)
    X_sample = X[np.random.choice(len(X), size=min(5, len(X)), replace=False)]
    labels, scores = predict_cnn_torch(model, X_sample, model_path=args.model)
    for i, (lbl, score) in enumerate(zip(labels, scores)):
        print(f"Sample {i+1}: {lbl} (score={score:.4f})")

    print("\n[+] Done.")
