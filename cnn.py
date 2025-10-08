import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os, joblib
import numpy as np
import pandas as pd

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
        # x: (batch, 1, num_features)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)           # (batch, 64, floor(num_features/2))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return self.softmax(x)
    
def train_cnn_torch(X: np.ndarray, y: np.ndarray,
                    model_path: str = "model/cnn_torch.pth",
                    epochs: int = 20,
                    batch_size: int = 32,
                    lr: float = 1e-3,
                    device: str = "cpu"):
    """
    Train CNN 1D bằng PyTorch.
    - Chỉ nhận X, y đã được xử lý (ví dụ: log2 transform).
    - Lưu state_dict vào model_path, và lưu label classes vào model_path + ".labels.pkl".
    - Trả về object model (nn.Module) đã train (trên CPU).
    """
    if torch is None:
        raise RuntimeError("PyTorch không được cài đặt trong môi trường này.")

    import pandas as pd

    # encode labels -> integers, giữ lớp để decode sau
    y_enc, classes = pd.factorize(y)
    num_classes = len(classes)
    num_features = X.shape[1]

    # tensors: shape (N, 1, num_features)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_enc, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Simple1DCNN(num_features=num_features, num_classes=num_classes)
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
        print(f"[CNN-Torch] Epoch {ep}/{epochs} - loss: {avg_loss:.4f}")

    # save
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    # save state_dict
    torch.save(model.state_dict(), model_path)
    # save label classes
    joblib.dump(classes, model_path + ".labels.pkl")
    print(f"[+] CNN (PyTorch) trained and saved to {model_path} (+ labels at .labels.pkl)")

    # return model loaded on cpu for prediction convenience
    model_cpu = Simple1DCNN(num_features=num_features, num_classes=num_classes)
    model_cpu.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_cpu.eval()
    return model_cpu    

def process_cnn_torch(data_csv, label_col="Label",
                      model_path="model/cnn_torch.pth",
                      epochs: int = 20, batch_size: int = 32, lr: float = 1e-3,
                      device: str = "cpu"):
    """
    Read CSV, pick feature columns, log2 transform, then call train_cnn_torch().
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
    model = train_cnn_torch(X_log, y, model_path=model_path,
                            epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    return model

def predict_cnn_torch(model: nn.Module | str, X: np.ndarray, model_path: str = None, device: str = "cpu"):
    """
    Predict with a trained PyTorch CNN.
    - model: either a nn.Module instance (loaded) OR None (if you pass model_path), then model_path used.
    - X: raw features (should be same preprocessing as training, e.g. log2)
    Returns (labels, scores) where:
      - labels: decoded class labels (strings) if classes file exists, else integer indices
      - scores: max probability per sample (float ndarray)
    """
    if torch is None:
        raise RuntimeError("PyTorch không được cài đặt trong môi trường này.")

    # load model from path if needed
    loaded_here = False
    if not isinstance(model, nn.Module):
        if model_path is None:
            raise ValueError("If model object not provided you must set model_path")
        # need num_features and num_classes to re-create model; load label file to get classes
        classes = joblib.load(model_path + ".labels.pkl")
        num_classes = len(classes)
        num_features = X.shape[1]
        model_local = Simple1DCNN(num_features=num_features, num_classes=num_classes)
        model_local.load_state_dict(torch.load(model_path, map_location="cpu"))
        model_local.eval()
        model = model_local
        loaded_here = True
    else:
        # try load classes if path provided
        classes = None
        if model_path and os.path.exists(model_path + ".labels.pkl"):
            classes = joblib.load(model_path + ".labels.pkl")

    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)  # (N,1,F)
        probs = model(X_t)  # (N, num_classes)
        probs_np = probs.cpu().numpy()
        preds_idx = probs_np.argmax(axis=1)
        scores = probs_np.max(axis=1)

    # decode labels if classes available
    if 'classes' in locals() and classes is not None:
        labels = [classes[i] for i in preds_idx]
    else:
        labels = preds_idx.tolist()

    return np.array(labels, dtype=object), np.array(scores, dtype=float)