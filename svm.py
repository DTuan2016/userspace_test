import os
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Đường dẫn CSV
csv_file = "/home/dongtv/dtuan/training_isolation/data.csv"

# 2. Kiểm tra file
if not os.path.isfile(csv_file):
    raise FileNotFoundError(f"CSV file not found: {csv_file}")

# 3. Đọc dữ liệu
df = pd.read_csv(csv_file)

# 4. Xử lý label: BENIGN=1, else=0
if 'Label' not in df.columns:
    raise ValueError("Column 'Label' not found in CSV")

df['Label'] = df['Label'].apply(lambda x: 1 if str(x).strip().upper() == "BENIGN" else 0)

# 5. Chọn features
feature_cols = ["FlowDuration", "FlowPktsPerSec", "FlowBytesPerSec", "FlowIATMean", "PktLenMean"]
for col in feature_cols:
    if col not in df.columns:
        raise ValueError(f"Feature column '{col}' not found in CSV")

X = df[feature_cols].astype(float).values
y = df['Label'].values

print("Label distribution:\n", pd.Series(y).value_counts())

# 6. Log2 scale để dữ liệu dễ hội tụ
X = np.log2(X + 1)

# 7. Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

# 8. Train Linear SVM
model = LinearSVC(C=1.0, max_iter=5000)
model.fit(X_train, y_train)

# 9. Test
y_pred = model.predict(X_test)

# 10. Đánh giá
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Attack','BENIGN']))
