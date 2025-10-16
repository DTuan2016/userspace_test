import numpy as np
import pandas as pd
from read_map import read_flows
import time
import argparse
import sys
from algorithm import (
    train_isolation_forest,
    train_random_forest,
    train_linear_svm,
    train_lof,
    train_knn,
    predict_cnn_torch,
    predict_with_score,
    process_cnn_torch,
)

def parse_flow(flow_str):
    """
    Chuyển flow string kiểu '192.168.50.6:54821 -> 72.21.91.29:80 proto=6'
    thành tuple: (src_ip, src_port, dst_ip, dst_port, proto)
    """
    try:
        parts = flow_str.split("->")
        src_part = parts[0].strip()
        dst_part_proto = parts[1].strip()

        if "proto=" in dst_part_proto:
            dst_part, proto_part = dst_part_proto.split("proto=")
            proto = int(proto_part.strip())
        else:
            dst_part = dst_part_proto
            proto = None

        src_ip, src_port = src_part.split(":")
        dst_ip, dst_port = dst_part.split(":")

        return (src_ip.strip(), int(src_port), dst_ip.strip(), int(dst_port), proto)
    except Exception as e:
        print(f"[ERROR] Cannot parse flow '{flow_str}': {e}")
        return None

def run_model(model_name, train_csv=None):
    """
    Hàm tổng quát cho tất cả các model, bao gồm CNN (PyTorch)
    """
    print(f"[DEBUG] run_model() called with model={model_name}, train_csv={train_csv}")

    map_path = "/sys/fs/bpf/enp1s0f0/xdp_flow_tracking"
    buffer_flows, seen_flows, printed_flows, results = [], set(), set(), []
    model = None

    # --- Khối train cho các model có sẵn ---
    if model_name in ["isoforest", "randforest", "svm", "cnn"]:
        if train_csv is None:
            raise ValueError("train_csv is required for isoforest, randforest, svm, or cnn")

        df = pd.read_csv(train_csv)
        feature_columns = ["FlowDuration", "FlowIATMean", "FlowPktsPerSec", "FlowBytesPerSec", "PktLenMean"]
        X_train = df[feature_columns].astype(float).values
        X_train_log = np.log2(X_train + 1)

        if model_name == "isoforest":
            model = train_isolation_forest(X_train_log, contamination=0.01)
            print("[DEBUG] IsolationForest training complete from CSV.")
        elif model_name == "randforest":
            y_train = df["Label"]
            model = train_random_forest(X_train_log, y_train)
            print("[DEBUG] RandomForest training complete from CSV.")
        elif model_name == "svm":
            y_train = df["Label"]
            model = train_linear_svm(X_train_log, y_train)
            print("[DEBUG] Linear SVM training complete from CSV.")
        elif model_name == "cnn":
            model_path = "model/cnn.pth"
            print("[DEBUG] Training CNN (PyTorch) from CSV...")
            model = process_cnn_torch(
                data_csv=train_csv,
                model_path=model_path,
                epochs=10,
                device="cpu"
            )
            print("[DEBUG] CNN (PyTorch) training complete.")
    else:
        print(f"[DEBUG] Waiting for 100 unique flows to train ({model_name.upper()})...")

    try:
        while True:
            try:
                flows = read_flows(map_path)
            except Exception as e:
                print(f"[ERROR] Exception in read_flows: {e}")
                time.sleep(1)
                continue

            if not flows:
                time.sleep(1)
                continue

            temp_data = []
            for flow_str, dp in flows:
                flow_tuple = parse_flow(flow_str)
                if flow_tuple is None:
                    continue

                if model_name in ["lof", "knn"] and model is None:
                    if flow_tuple in seen_flows:
                        continue
                    seen_flows.add(flow_tuple)

                features = [
                    dp.flow_duration,
                    dp.flow_IAT_mean,
                    dp.flow_pkts_per_s,
                    dp.flow_bytes_per_s,
                    dp.pkts_len_mean
                ]
                temp_data.append((flow_tuple, features))

            # Training phase for LOF/KNN
            if model_name in ["lof", "knn"] and model is None:
                for _, feat in temp_data:
                    buffer_flows.append(feat)

                if len(buffer_flows) >= 100:
                    X_train = np.array(buffer_flows[:100], dtype=float)
                    X_train_log = np.log2(X_train + 1)
                    if model_name == "lof":
                        model = train_lof(X_train_log, n_neighbors=5, contamination=0.01)
                    else:
                        model = train_knn(X_train_log, k=5)
                    print(f"[DEBUG] {model_name.upper()} training complete. Now predicting new flows...")
                else:
                    print(f"[DEBUG] Collected {len(buffer_flows)}/100 unique flows so far...")
                    time.sleep(1)
                    continue

            if not temp_data:
                time.sleep(1)
                continue

            # --- Prediction ---
            X = np.array([feat for _, feat in temp_data], dtype=float)
            X_log = np.log2(X + 1)

            if model_name == "cnn":
                labels, scores = predict_cnn_torch(model, X_log, model_path="model/cnn.pth")
            else:
                labels, scores = predict_with_score(model, X_log)

            print(f"\n=== {model_name.upper()} Prediction Results ===")
            for (flow_tuple, _), label, score in zip(temp_data, labels, scores):
                if flow_tuple not in printed_flows:
                    src_ip, src_port, dst_ip, dst_port, proto = flow_tuple
                    lbl_str = "Outlier" if str(label).lower() in ["-1", "outlier"] else "Normal"
                    print(f"{src_ip}:{src_port} -> {dst_ip}:{dst_port} proto={proto} -> {lbl_str} (score={float(np.mean(score)):.4f})")
                    printed_flows.add(flow_tuple)
                    results.append({
                        "Flow": f"{src_ip},{src_port},{dst_ip},{dst_port},{proto}",
                        "Label": lbl_str,
                        "Score": float(np.mean(score))
                    })
            print("=== END ===\n")
            time.sleep(1)

    except KeyboardInterrupt:
        out_file = f"results/{model_name}_results.csv"
        print(f"\n[INFO] Ctrl+C detected, saving results to {out_file} ...")
        pd.DataFrame(results).to_csv(out_file, index=False)
        print(f"[INFO] Saved {len(results)} predictions to {out_file}")
        sys.exit(0)


def main():
    print("[DEBUG] main() started") 
    parser = argparse.ArgumentParser(description="Run anomaly detection models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["lof", "knn", "isoforest", "randforest", "svm", "cnn"],
                        help="Choose one algorithm")
    parser.add_argument("--train_csv", type=str,
                        default="/home/lanforge/userspace_test/data/data.csv",
                        help="CSV file for training (required for isoforest, randforest, svm, cnn)")

    args = parser.parse_args()
    run_model(args.model, args.train_csv if args.model in ["isoforest", "randforest", "svm", "cnn"] else None)
    
if __name__ == "__main__":
    main()