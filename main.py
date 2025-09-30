import numpy as np
from read_map import read_flows
import algorithm

map_path = "/sys/fs/bpf/enp1s0f0/xdp_flow_tracking"
flows = read_flows(map_path)

# build dataset
X = np.array([
    [
        dp.flow_duration,
        dp.flow_IAT_mean,
        dp.flow_pkts_per_s,
        dp.flow_bytes_per_s,
        dp.pkts_len_mean
    ]
    for _, dp in flows
], dtype=float)

X_log = np.log2(X + 1)  # log scale

# train LOF
lof = algorithm.train_lof(X_log, n_neighbors=1, contamination=0.1)

# predict lại chính dataset
labels = algorithm.predict(lof, X_log)
print(labels)
