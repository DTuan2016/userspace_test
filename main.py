import numpy as np
import pandas as pd
from read_map import read_flows
import time
import argparse
import sys, os, joblib
from sklearn.preprocessing import MinMaxScaler
import torch
from algorithm import (
    train_isolation_forest,
    train_random_forest,
    train_linear_svm,
    train_lof,
    train_knn,
    predict_mlp_torch,
    predict_with_score,
    train_mlp_torch,
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

def dump_mlp_to_csv(model_path="model/mlp.pth", output_csv="model/mlp_weights.csv", device="cpu"):
    """
    Dump MLP (1 layer Linear + Softmax) ra CSV:
      out_idx,in_idx,weight
    Tự động xác định số features & classes từ checkpoint.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file {model_path} not found.")
        return

    # Load raw state_dict
    state_dict = torch.load(model_path, map_location=device)

    # Lấy weight & bias trực tiếp
    weight_tensor = state_dict["fc1.weight"]
    bias_tensor = state_dict["fc1.bias"]

    num_classes, num_features = weight_tensor.shape
    print(f"[INFO] Loaded MLP with {num_features} input features → {num_classes} output classes")

    weights = weight_tensor.detach().cpu().numpy()
    bias = bias_tensor.detach().cpu().numpy()

    rows = []
    for out_idx in range(num_classes):
        for in_idx in range(num_features):
            rows.append({
                "out_idx": out_idx,
                "in_idx": in_idx,
                "weight": weights[out_idx, in_idx]
            })
        rows.append({
            "out_idx": out_idx,
            "in_idx": -1,  # bias
            "weight": bias[out_idx]
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[+] Dumped {len(df)} weights (including bias) to {output_csv}")
    return df

def dump_random_forest_to_csv(
    model_path="model/randforest.pkl",
    output_csv="model/randforest_nodes.csv"
):
    """
    Xuất toàn bộ node trong RandomForestClassifier ra CSV.
    - Nếu node là lá: label = 1 (BENIGN) hoặc 0 (Portmap)
    - Nếu node không phải lá: label = -1
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")

    model = joblib.load(model_path)

    if not hasattr(model, "estimators_"):
        raise ValueError("Model không có thuộc tính estimators_ — có thể chưa train xong hoặc không phải RandomForestClassifier.")

    rows = []
    n_trees = len(model.estimators_)
    class_labels = list(model.classes_)  # ví dụ: ['BENIGN', 'Portmap']

    # Ánh xạ nhãn: BENIGN=1, Portmap=0
    label_map = {"BENIGN": 1, "Portmap": 0}

    for tree_idx, tree in enumerate(model.estimators_):
        tree_struct = tree.tree_
        n_nodes = tree_struct.node_count

        for node_idx in range(n_nodes):
            feature = int(tree_struct.feature[node_idx])
            threshold = float(tree_struct.threshold[node_idx])
            left_child = int(tree_struct.children_left[node_idx])
            right_child = int(tree_struct.children_right[node_idx])
            is_leaf = int(left_child == -1 and right_child == -1)

            label = -1
            if is_leaf:
                value = tree_struct.value[node_idx].flatten()
                if value.sum() > 0:
                    majority_label = class_labels[int(np.argmax(value))]
                    label = label_map.get(majority_label, -1)

            rows.append({
                "tree_idx": tree_idx,
                "node_idx": node_idx,
                "feature_idx": feature if feature >= 0 else -1,
                "split_value": threshold if threshold != -2 else -1.0,
                "left_child": left_child,
                "right_child": right_child,
                "is_leaf": is_leaf,
                "label": label
            })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[+] Dumped {len(df)} nodes from {n_trees} trees to {output_csv}")
    return df

def export_single_linear_svm_weight_to_csv(model_path: str, csv_path: str):
    """
    Dump ONE weight vector + bias from a trained Linear SVM to CSV.
    Output: one row [w0, w1, ..., wN, bias]
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    svm = joblib.load(model_path)

    if not hasattr(svm, "coef_") or not hasattr(svm, "intercept_"):
        raise TypeError("Provided model is not a linear SVM (missing coef_ / intercept_)")

    # Take the first (and usually only) weight vector
    weights = svm.coef_[0]
    bias = svm.intercept_[0]

    # Combine into one array [w0, w1, ..., bias]
    export_vec = np.append(weights, bias)

    # Write to CSV as a single line
    np.savetxt(csv_path, [export_vec], delimiter=",", fmt="%.6f")

    print(f"[INFO] Exported 1 weight vector with {len(weights)} features + bias to {csv_path}")

# ============================================================
# ================  TRAINING STAGE ===========================
# ============================================================
def train_all_models(train_csv):
    print("[INFO] === TRAINING ALL MODELS ===")
    df = pd.read_csv(train_csv)
    # Lọc chỉ giữ 2 nhãn
    df = df[df["Label"].isin(["BENIGN", "Portmap"])]

    # Nếu rỗng hoặc chỉ có 1 loại -> bỏ qua model
    if df["Label"].nunique() < 2:
        print("[WARN] Không đủ 2 loại nhãn (BENIGN, PortMap) → bỏ qua training.")
        return

    X = df[["FlowDuration", "FlowPktsPerSec", "FlowBytesPerSec", "FlowIATMean", "PktLenMean"]].values
    y = df["Label"].values

    X_train_log = np.log2(X + 1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train_log)
    df_log = pd.DataFrame(X_train_log, columns=["FlowDuration", "FlowPktsPerSec", "FlowBytesPerSec", "FlowIATMean", "PktLenMean"])
    df_scaled = pd.DataFrame(X_scaled, columns=["FlowDuration", "FlowPktsPerSec", "FlowBytesPerSec", "FlowIATMean", "PktLenMean"])

    os.makedirs("model/debug_features", exist_ok=True)
    df_log.to_csv("model/debug_features/X_train_log.csv", index=False)
    df_scaled.to_csv("model/debug_features/X_scaled.csv", index=False)
    print("[+] Saved log-transformed features to model/debug_features/X_train_log.csv")
    print("[+] Saved min-max scaled features to model/debug_features/X_scaled.csv")

    os.makedirs("model", exist_ok=True)
    scaler_path = "model/minmax_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    print(f"[+] Scaler saved to {scaler_path}")
    # Isolation Forest
    print("[TRAIN] Isolation Forest ...")
    isof = train_isolation_forest(X_train_log, contamination=0.01)
    os.makedirs("model", exist_ok=True)
    # import joblib
    joblib.dump(isof, "model/isoforest.pkl")

    # Random Forest
    print("[TRAIN] Random Forest ...")
    rf = train_random_forest(X_train_log, y)
    joblib.dump(rf, "model/randforest.pkl")
    dump_random_forest_to_csv()

    # Linear SVM
    print("[TRAIN] Linear SVM ...")
    svm = train_linear_svm(X_scaled, y)
    joblib.dump(svm, "model/svm.pkl")
    export_single_linear_svm_weight_to_csv("model/svm.pkl", "model/svm_weight.csv")

    # LOF
    print("[TRAIN] Local Outlier Factor ...")
    lof = train_lof(X_train_log, n_neighbors=5, contamination=0.01)
    joblib.dump(lof, "model/lof.pkl")

    # KNN
    print("[TRAIN] KNN ...")
    knn = train_knn(X_train_log, k=5)
    joblib.dump(knn, "model/knn.pkl")

    # MLP
    print("[TRAIN] MLP (PyTorch) ...")
    train_mlp_torch(X_train_log, y, model_path="model/mlp.pth", epochs=25, device="cpu")
    dump_mlp_to_csv("model/mlp.pth", "model/mlp_weights.csv")
        # === Dump min-max scaler ra CSV ===
    min_vals = scaler.data_min_
    max_vals = scaler.data_max_
    df_mm = pd.DataFrame({
        "feature_idx": np.arange(len(min_vals)),
        "min_val": min_vals,
        "max_val": max_vals
    })
    df_mm.to_csv("model/minmax_params.csv", index=False)
    print("[+] MinMax parameters saved to model/minmax_params.csv")

    print("[INFO] === TRAINING COMPLETED ===")

def run_inference(model_name):
    """
    Chạy inference realtime từ eBPF map hoặc dump CSV.
    """
    print(f"[INFO] Running inference for model={model_name}")
    map_path = "/sys/fs/bpf/eno3/xdp_flow_tracking"
    results, printed_flows = [], set()
    scaler = joblib.load("model/minmax_scaler.pkl")
    # Load model
    # import joblib
    if model_name == "mlp":
        model_path = "model/mlp.pth"
    else:
        model_path = f"model/{model_name}.pkl"
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file {model_path} not found. Please run --train first.")
            sys.exit(1)
        model = joblib.load(model_path)

    try:
        while True:
            from read_map import read_flows
            flows = read_flows(map_path)
            if not flows:
                time.sleep(1)
                continue

            temp_data = []
            for flow_str, dp in flows:
                flow_tuple = parse_flow(flow_str)
                if not flow_tuple:
                    continue
                features = [
                    dp.flow_duration,
                    dp.flow_IAT_mean,
                    dp.flow_pkts_per_s,
                    dp.flow_bytes_per_s,
                    dp.pkts_len_mean
                ]
                temp_data.append((flow_tuple, features))

            X = np.array([feat for _, feat in temp_data], dtype=float)
            X_log = np.log2(X + 1)
            X_scaled = scaler.transform(X_log)

            if model_name == "mlp":
                labels, scores = predict_mlp_torch(None, X_scaled, model_path="model/mlp.pth")
            else:
                labels, scores = predict_with_score(model, X_scaled)

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
        os.makedirs("results", exist_ok=True)
        print(f"\n[INFO] Ctrl+C detected, saving results to {out_file} ...")
        pd.DataFrame(results).to_csv(out_file, index=False)
        print(f"[INFO] Saved {len(results)} predictions to {out_file}")
        sys.exit(0)


# ============================================================
# ================  MAIN ENTRY ===============================
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train or run anomaly detection models")
    parser.add_argument("--train", action="store_true", help="Train all models from CSV")
    parser.add_argument("--model", type=str, choices=["lof", "knn", "isoforest", "randforest", "svm", "mlp"],
                        help="Run inference with a specific model")
    parser.add_argument("--train_csv", type=str,
                        default="/home/dongtv/dtuan/training_isolation/data.csv",
                        help="CSV file for training")

    args = parser.parse_args()

    if args.train:
        train_all_models(args.train_csv)
    elif args.model:
        run_inference(args.model)
    else:
        print("Usage:")
        print("  python3 main.py --train")
        print("  python3 main.py --model <lof|knn|isoforest|randforest|svm|mlp>")


if __name__ == "__main__":
    main()