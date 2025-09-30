import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
from get_data import get_data, DataPoint

def extract_features(dp: DataPoint):
    return [
        dp.flow_duration,
        dp.flow_IAT_mean,
        dp.flow_pkts_per_s,
        dp.flow_bytes_per_s,
        dp.pkts_len_mean,
    ]

# --- Training phase ---
train_flows = get_data('/sys/fs/bpf/eno3/xdp_flow_tracking')  # hoặc load nhiều batch từ map/lưu trữ
X = np.array([extract_features(dp) for _, dp in train_flows], dtype=float)
X_log = np.log2(X + 1)  # log2 scale

# fit KNN
k = 3
knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
knn.fit(X_log)

# optional: lưu model KNN để dùng sau
joblib.dump(knn, "knn_model.pkl")
print("Training done, KNN model saved.")
