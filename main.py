import numpy as np
from read_map import read_flows
import algorithm
import time

def main():
    map_path = "/sys/fs/bpf/enp1s0f0/xdp_flow_tracking"

    # Train LOF 1 lần từ CSV
    lof = algorithm.process_lof("test.csv", n_neighbors=5, contamination=0.1)

    while True:
        flows = read_flows(map_path)
        if not flows:
            print("[MAIN] EMPTY")
            time.sleep(2)
            continue

        # build dataset từ map
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

        X_log = np.log2(X + 1)

        # inference
        labels = algorithm.predict(lof, X_log)

        # in kết quả
        print("\n=== LOF Prediction Results ===")
        for (flow, _), label in zip(flows, labels):
            print(f"{flow:40s} -> {'Outlier' if label == -1 else 'Normal'}")

        time.sleep(0.5)  # đọc lại map mỗi 2 giây

if __name__ == "__main__":
    main()
