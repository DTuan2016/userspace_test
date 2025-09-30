import numpy as np
from sklearn.neighbors import LocalOutlierFactor
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

# --- LOF model ---
# Nếu bạn muốn semi-supervised, hãy dùng novelty=True
lof = LocalOutlierFactor(n_neighbors=2, novelty=True)
lof.fit(X_log)

# Lưu model LOF
joblib.dump(lof, "lof_model.pkl")
print("Training done, LOF model saved.")
