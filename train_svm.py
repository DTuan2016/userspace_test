import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# Đọc dữ liệu train
df = pd.read_csv("/home/dongtv/dtuan/training_isolation/data.csv")
# Chỉ lấy các cột số đã chọn
feature_columns = [
    "FlowDuration",
    "FlowPktsPerSec",
    "FlowBytesPerSec",
    "FlowIATMean",
    "PktLenMean"
]

X = df[feature_columns].astype(float).values 
y = df["Label"].apply(lambda x: 1 if x == "BENIGN" else -1).values         

print("Label distribution:\n", pd.Series(y).value_counts())

# Log2 scale
X = np.log2(X + 1)

# In ra giá trị sau khi log2
df_log2 = pd.DataFrame(X, columns=feature_columns)
df_log2["Label"] = y

df_log2.to_csv("data_log2.csv", index=False)
print("Saved log2 scaled features to data_log2.csv")

model = LinearSVC(C=1.0, max_iter=5000)
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]

print("Trained w:", w)
print("Trained b:", b)

SCALE = 1000000
w_q = (w * SCALE).astype(np.int32)
b_q = int(b * SCALE)

with open("weights.csv", "w") as f:
    for val in w_q:
        f.write(f"{val}\n")
    f.write(f"{b_q}\n")

print("Saved quantized weights to weights.csv")