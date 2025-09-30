#!/usr/bin/env python3
import numpy as np
import joblib
from get_data import get_data, DataPoint
import time
import csv
import socket
import struct

def extract_features(dp: DataPoint):
    """Trích 5 feature từ DataPoint"""
    return [
        dp.flow_duration,
        dp.flow_IAT_mean,
        dp.flow_pkts_per_s,
        dp.flow_bytes_per_s,
        dp.pkts_len_mean,
    ]

#!/usr/bin/env python3
import numpy as np
import joblib
from get_data import get_data, DataPoint
import time
import csv
import socket
import struct

def extract_features(dp: DataPoint):
    """Trích 5 feature từ DataPoint"""
    return [
        dp.flow_duration,
        dp.flow_IAT_mean,
        dp.flow_pkts_per_s,
        dp.flow_bytes_per_s,
        dp.pkts_len_mean,
    ]

def fix_flow_str(flow_str):
    """
    Chuyển IP từ số nguyên sang dạng x.x.x.x
    flow_str = "ip:port" từ get_data()
    """
    try:
        ip_str, port_str = flow_str.split(":")
        ip_int = int(ip_str)
        ip_fixed = socket.inet_ntoa(struct.pack(">I", ip_int))  # big-endian
        port = int(port_str)
        return f"{ip_fixed}:{port}"
    except ValueError:
        # nếu đã là string dạng "x.x.x.x", giữ nguyên
        return flow_str

def main():
    # load KNN model đã train
    knn = joblib.load("knn_model.pkl")
    threshold = 1.0

    results = []  # lưu toàn bộ batch để xuất CSV

    try:
        while True:
            flows = get_data()  # list of (flow_str, DataPoint)
            if not flows:
                time.sleep(1)
                continue

            # tạo ma trận feature
            X = np.array([extract_features(dp) for _, dp in flows], dtype=float)
            X_log = np.log2(X + 1)  # log2 scale

            # tính KNN so với tập train
            distances, indices = knn.kneighbors(X_log)
            min_dist = distances.min(axis=1)

            print("\n=== Outlier Detection ===")
            print(f"{'STT':<4} | {'Flow':<21} | {'Dist':<10} | {'Outlier'}")

            for i, (flow_str, dp) in enumerate(flows, 1):
                flow_fixed = fix_flow_str(flow_str)
                is_outlier = min_dist[i-1] > threshold

                print(f"{i:<4d} | {flow_fixed:<21} | {min_dist[i-1]:<10.2f} | {is_outlier}")

                # lưu kết quả vào danh sách
                results.append({
                    "STT": i,
                    "Flow": flow_fixed,
                    "Dist": min_dist[i-1],
                    "Outlier": is_outlier,
                    "flow_duration": dp.flow_duration,
                    "flow_IAT_mean": dp.flow_IAT_mean,
                    "flow_pkts_per_s": dp.flow_pkts_per_s,
                    "flow_bytes_per_s": dp.flow_bytes_per_s,
                    "pkts_len_mean": dp.pkts_len_mean
                })

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Saving results to CSV...")

        if results:
            csv_file = "outlier_results.csv"
            with open(csv_file, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
            print(f"[INFO] Results saved to {csv_file}")
        else:
            print("[INFO] No results to save.")

def main():
    # load KNN model đã train
    knn = joblib.load("model/knn_model.pkl")
    threshold = 1.0

    results = []  # lưu toàn bộ batch để xuất CSV

    try:
        while True:
            flows = get_data('/sys/fs/bpf/eno3/xdp_flow_tracking')  # list of (flow_str, DataPoint)
            if not flows:
                time.sleep(1)
                continue

            # tạo ma trận feature
            X = np.array([extract_features(dp) for _, dp in flows], dtype=float)
            X_log = np.log2(X + 1)  # log2 scale

            # tính KNN so với tập train
            distances, indices = knn.kneighbors(X_log)
            min_dist = distances.min(axis=1)

            print("\n=== Outlier Detection ===")
            print(f"{'STT':<4} | {'Flow':<21} | {'Dist':<10} | {'Outlier'}")

            for i, (flow_str, dp) in enumerate(flows, 1):
                flow_fixed = fix_flow_str(flow_str)
                is_outlier = min_dist[i-1] > threshold

                print(f"{i:<4d} | {flow_fixed:<21} | {min_dist[i-1]:<10.2f} | {is_outlier}")

                # lưu kết quả vào danh sách
                results.append({
                    "STT": i,
                    "Flow": flow_fixed,
                    "Dist": min_dist[i-1],
                    "Outlier": is_outlier,
                    "flow_duration": dp.flow_duration,
                    "flow_IAT_mean": dp.flow_IAT_mean,
                    "flow_pkts_per_s": dp.flow_pkts_per_s,
                    "flow_bytes_per_s": dp.flow_bytes_per_s,
                    "pkts_len_mean": dp.pkts_len_mean
                })

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Saving results to CSV...")

        if results:
            csv_file = "outlier_results.csv"
            with open(csv_file, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
            print(f"[INFO] Results saved to {csv_file}")
        else:
            print("[INFO] No results to save.")

if __name__ == "__main__":
    main()
