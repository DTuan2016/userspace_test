import numpy as np
import pandas as pd
from read_map import read_flows
import algorithm

import time
import argparse

def lof_run():
    map_path = "/sys/fs/bpf/eno3/xdp_flow_tracking"

    buffer_flows = []       # lưu features
    seen_flows = set()      # lưu key flow đã gặp
    lof = None

    print("[DEBUG] Waiting for 100 unique flows to train...")

    while True:
        try:
            flows = read_flows(map_path)
        except Exception as e:
            print(f"[ERROR] Exception in read_flows: {e}")
            time.sleep(1)
            continue

        if not flows:
            print("[DEBUG] No flows in map.")
            time.sleep(1)
            continue

        temp_data = []
        for flow, dp in flows:
            if lof is None:
                # lọc flow trùng
                if flow in seen_flows:
                    continue
                seen_flows.add(flow)

            features = [
                dp.flow_duration,
                dp.flow_IAT_mean,
                dp.flow_pkts_per_s,
                dp.flow_bytes_per_s,
                dp.pkts_len_mean
            ]
            temp_data.append((flow, features))

        # giai đoạn training
        if lof is None:
            for _, feat in temp_data:
                buffer_flows.append(feat)

            if len(buffer_flows) >= 100:
                print(f"[DEBUG] Collected {len(buffer_flows)} unique flows. Training LOF model...")
                X_train = np.array(buffer_flows[:100], dtype=float)
                X_train_log = np.log2(X_train + 1)

                lof = algorithm.train_lof(X_train_log, n_neighbors=20, contamination=0.1)
                print("[DEBUG] LOF training complete. Now predicting new flows...")
            else:
                print(f"[DEBUG] Collected {len(buffer_flows)}/100 unique flows so far...")
                time.sleep(1)
                continue

        # giai đoạn predict
        else:
            if not temp_data:
                time.sleep(1)
                continue

            X = np.array([feat for _, feat in temp_data], dtype=float)
            X_log = np.log2(X + 1)

            labels, scores = algorithm.predict_with_score(lof, X_log)

            print("\n=== LOF Prediction Results ===")
            for (flow, _), label, score in zip(temp_data, labels, scores):
                print(f"{flow:40s} -> {'Outlier' if label == -1 else 'Normal'} (score={score:.4f})")
            print("=== END ===\n")

        time.sleep(0.1)

def knn_run():
    map_path = "/sys/fs/bpf/eno3/xdp_flow_tracking"

    buffer_flows = []
    seen_flows = set()
    knn = None

    print("[DEBUG] Waiting for 100 unique flows to train (KNN)...")

    while True:
        try:
            flows = read_flows(map_path)
        except Exception as e:
            print(f"[ERROR] Exception in read_flows: {e}")
            time.sleep(1)
            continue

        if not flows:
            print("[DEBUG] No flows in map.")
            time.sleep(1)
            continue

        temp_data = []
        for flow, dp in flows:
            if knn is None:
                if flow in seen_flows:
                    continue
                seen_flows.add(flow)

            features = [
                dp.flow_duration,
                dp.flow_IAT_mean,
                dp.flow_pkts_per_s,
                dp.flow_bytes_per_s,
                dp.pkts_len_mean
            ]
            temp_data.append((flow, features))

        if knn is None:
            for _, feat in temp_data:
                buffer_flows.append(feat)

            if len(buffer_flows) >= 100:
                print(f"[DEBUG] Collected {len(buffer_flows)} unique flows. Training KNN model...")
                X_train = np.array(buffer_flows[:100], dtype=float)
                X_train_log = np.log2(X_train + 1)

                knn = algorithm.train_knn(X_train_log, k=5)
                print("[DEBUG] KNN training complete. Now predicting new flows...")
            else:
                print(f"[DEBUG] Collected {len(buffer_flows)}/100 unique flows so far...")
                time.sleep(1)
                continue
        else:
            if not temp_data:
                time.sleep(1)
                continue

            X = np.array([feat for _, feat in temp_data], dtype=float)
            X_log = np.log2(X + 1)

            labels, scores = algorithm.predict_with_score(knn, X_log)

            print("\n=== KNN Prediction Results ===")
            for (flow, _), label, score in zip(temp_data, labels, scores):
                print(f"{flow:40s} -> {'Outlier' if label == -1 else 'Normal'} (score={score:.4f})")
            print("=== END ===\n")

        time.sleep(0.1)


def isoforest_run(csv_path="train_data.csv"):
    map_path = "/sys/fs/bpf/eno3/xdp_flow_tracking"

    # Chọn đúng cột feature numeric
    feature_columns = [
        "FlowDuration",
        "FlowIATMean",
        "FlowPktsPerSec",
        "FlowBytesPerSec",
        "PktLenMean"
    ]

    df = pd.read_csv(csv_path)

    # Lọc chỉ lấy cột numeric
    X_train = df[feature_columns].astype(float).values
    X_train_log = np.log2(X_train + 1)

    iso = algorithm.train_isolation_forest(X_train_log, contamination=0.1)
    print("[DEBUG] IsolationForest training complete from CSV.")

    # Loop predict real-time flows
    while True:
        try:
            flows = read_flows(map_path)
        except Exception as e:
            print(f"[ERROR] Exception in read_flows: {e}")
            time.sleep(1)
            continue

        if not flows:
            print("[DEBUG] No flows in map.")
            time.sleep(1)
            continue

        temp_data = []
        for flow, dp in flows:
            features = [
                dp.flow_duration,
                dp.flow_IAT_mean,
                dp.flow_pkts_per_s,
                dp.flow_bytes_per_s,
                dp.pkts_len_mean
            ]
            temp_data.append((flow, features))

        X = np.array([feat for _, feat in temp_data], dtype=float)
        X_log = np.log2(X + 1)

        labels, scores = algorithm.predict_with_score(iso, X_log)

        print("\n=== IsolationForest Prediction Results ===")
        for (flow, _), label, score in zip(temp_data, labels, scores):
            print(f"{flow:40s} -> {'Outlier' if label == -1 else 'Normal'} (score={score:.4f})")
        print("=== END ===\n")

        time.sleep(0.1)


def randforest_run(csv_path="train_data.csv"):
    map_path = "/sys/fs/bpf/eno3/xdp_flow_tracking"

    feature_columns = [
        "FlowDuration",
        "FlowIATMean",
        "FlowPktsPerSec",
        "FlowBytesPerSec",
        "PktLenMean"
    ]
    label_column = "Label"
    df = pd.read_csv(csv_path)
    X_train = df[feature_columns].astype(float).values
    y_train = df[label_column]
    X_train_log = np.log2(X_train + 1)

    rf = algorithm.train_random_forest(X_train_log, y_train)
    print("[DEBUG] RandomForest training complete from CSV.")

    # Loop predict real-time flows
    while True:
        try:
            flows = read_flows(map_path)
        except Exception as e:
            print(f"[ERROR] Exception in read_flows: {e}")
            time.sleep(1)
            continue

        if not flows:
            print("[DEBUG] No flows in map.")
            time.sleep(1)
            continue

        temp_data = []
        for flow, dp in flows:
            features = [
                dp.flow_duration,
                dp.flow_IAT_mean,
                dp.flow_pkts_per_s,
                dp.flow_bytes_per_s,
                dp.pkts_len_mean
            ]
            temp_data.append((flow, features))

        X = np.array([feat for _, feat in temp_data], dtype=float)
        X_log = np.log2(X + 1)

        labels, scores = algorithm.predict_with_score(rf, X_log)

        print("\n=== RandomForest Prediction Results ===")
        for (flow, _), label, score in zip(temp_data, labels, scores):
            print(f"{flow:40s} -> {'Outlier' if label == -1 else 'Normal'} (score={score:.4f})")
        print("=== END ===\n")

        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["lof", "knn", "isoforest", "randforest"],
                        help="Choose one algorithm")
    parser.add_argument("--train_csv", type=str, default="/home/dongtv/dtuan/training_isolation/data.csv",
                        help="Choose file CSV to Train isoforest or randforest")
    
    args = parser.parse_args()
    if args.model == "lof":
        lof_run()
    elif args.model == "knn":
        knn_run()
    elif args.model == "isoforest":
        isoforest_run(args.train_csv)
    elif args.model == "randforest":
        randforest_run(args.train_csv)
    else:
        raise ValueError("Unknown model")
    
if __name__ == "__main__":
    main()